from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as redis

from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)


class RedisCheckpointer:
    def __init__(self, redis_url: str = "redis://redis-checkpointer:6379/0"):
        self.redis: redis.Redis = redis.from_url(redis_url, decode_responses=True)

    @staticmethod
    def _serialize(state: AgentState) -> str:
        def default_serializer(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        return json.dumps(state.model_dump(), default=default_serializer, ensure_ascii=False)

    @staticmethod
    def _deserialize(data: str) -> AgentState:
        obj = json.loads(data)
        def convert_iso(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_iso(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_iso(i) for i in obj]
            if isinstance(obj, str):
                try:
                    return datetime.fromisoformat(obj.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    return obj
            return obj
        converted = convert_iso(obj)
        return AgentState(**converted)

    async def save(self, thread_id: str, state: AgentState) -> None:
        key = f"campaign_state:{thread_id}"
        try:
            serialized = self._serialize(state)
            await self.redis.set(key, serialized, ex=86400 * 7)
            logger.debug(f"State for {thread_id} saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state for {thread_id}: {e}")
            raise

    async def load(self, thread_id: str) -> Optional[AgentState]:
        key = f"campaign_state:{thread_id}"
        data = await self.redis.get(key)
        if data:
            try:
                return self._deserialize(data)
            except Exception as e:
                logger.error(f"Failed to load state for {thread_id}: {e}")
                return None
        return None

    async def delete(self, thread_id: str) -> None:
        key = f"campaign_state:{thread_id}"
        await self.redis.delete(key)
