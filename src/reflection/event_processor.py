from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

import asyncpg
import redis.asyncio as redis
import structlog

from src.reflection.llm_analyzer import LLMAnalyzer
from src.reflection.schemas import ReflectionInput

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)

EVENT_BUFFER_KEY = "reflection:event_buffer"


class EventBufferProcessor:
    def __init__(
        self,
        redis_client: redis.Redis,
        db_pool: asyncpg.Pool,
        llm_analyzer: LLMAnalyzer,
        flush_interval_secs: int = 60,
        batch_size: int = 50,
    ):
        self.redis = redis_client
        self.db = db_pool
        self.llm = llm_analyzer
        self.flush_interval = flush_interval_secs
        self.batch_size = batch_size
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._periodic_flush())
        logger.info("EventBufferProcessor запущен")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def add_event(self, event: ReflectionInput) -> None:
        await self.redis.lpush(EVENT_BUFFER_KEY, event.model_dump_json())
        await self.redis.expire(EVENT_BUFFER_KEY, 3600)

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(self.flush_interval)
            try:
                await self._flush_events()
            except Exception as e:
                logger.error(f"Ошибка сброса событий: {e}")

    async def _flush_events(self) -> None:
        events: List[Dict[str, Any]] = []
        while len(events) < self.batch_size:
            raw = await self.redis.rpop(EVENT_BUFFER_KEY)
            if raw is None:
                break
            events.append(json.loads(raw))

        if not events:
            return

        logger.info(f"Обработка {len(events)} событий рефлексии...")

        campaign_ids = {e["campaign_id"] for e in events}
        for cid in campaign_ids:
            group_events = [e for e in events if e["campaign_id"] == cid]
            analysis = await self.llm.analyze(group_events)

            insight_id = str(uuid4())
            root_cause = analysis.priority
            script_tweak = analysis.summary[:500]
            confidence = 0.7 if analysis.sentiment_trend != "neutral" else 0.5
            await self.db.execute(
                """
                INSERT INTO reflection_insights
                    (id, campaign_id, root_cause, suggested_script_tweak, confidence_score, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
                """,
                insight_id,
                cid,
                root_cause,
                script_tweak,
                confidence,
                datetime.now(timezone.utc),
            )
            logger.info(f"Сохранён insight {insight_id} для кампании {cid}")
