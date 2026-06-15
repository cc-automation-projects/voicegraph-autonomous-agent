from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import asyncpg
import httpx
from croniter import croniter

from src.reflection.prompts import WEEKLY_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

WEEKLY_LLM_URL = "http://vllm-service:8000/v1/chat/completions"


class WeeklyAggregator:
    def __init__(self, db_pool: asyncpg.Pool, cron_expr: str = "0 9 * * 1"):
        self.db = db_pool
        if not croniter.is_valid(cron_expr):
            raise ValueError(f"Невалидное cron-выражение: {cron_expr}")
        self.cron_expr = cron_expr

    async def generate_weekly_report(self) -> str:
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        rows = await self.db.fetch(
            """
            SELECT campaign_id, summary, action_items, sentiment_trend, priority, created_at
            FROM reflection_insights
            WHERE created_at >= $1
            ORDER BY created_at DESC
            """,
            week_ago,
        )

        if not rows:
            return "Нет данных за последнюю неделю."

        data = [
            {
                "campaign_id": row["campaign_id"],
                "summary": row["summary"],
                "action_items": row["action_items"],
                "sentiment_trend": row["sentiment_trend"],
                "priority": row["priority"],
            }
            for row in rows
        ]

        prompt = WEEKLY_SUMMARY_PROMPT.format(data=json.dumps(data, ensure_ascii=False, default=str))

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    WEEKLY_LLM_URL,
                    json={
                        "model": "qwen2.5-72b",
                        "messages": [
                            {"role": "system", "content": "Ты — аналитик VoiceGraph."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2000,
                    },
                )
                response.raise_for_status()
                report = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Ошибка генерации еженедельного отчёта: {e}")
            report = "Ошибка генерации отчёта."

        return report
