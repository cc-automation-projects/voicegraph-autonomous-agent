from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import redis.asyncio as redis

logger = logging.getLogger(__name__)

RETRY_QUEUE_KEY = "integrations:retry_queue"


class RetryWorker:
    def __init__(self, redis_client: redis.Redis, max_retries: int = 3, base_delay_sec: float = 1.0):
        self.redis = redis_client
        self.max_retries = max_retries
        self.base_delay = base_delay_sec

    async def enqueue(self, task_type: str, payload: dict) -> None:
        task = {
            "task_type": task_type,
            "payload": payload,
            "retry_count": 0,
            "max_retries": self.max_retries,
        }
        await self.redis.lpush(RETRY_QUEUE_KEY, json.dumps(task))
        logger.info(f"Задача {task_type} добавлена в очередь повторов")

    async def process_queue(self, handler: Callable[[str, dict], Awaitable[None]]) -> None:
        while True:
            try:
                raw = await self.redis.rpop(RETRY_QUEUE_KEY)
                if raw is None:
                    await asyncio.sleep(1)
                    continue

                task = json.loads(raw)
                task_type = task["task_type"]
                payload = task["payload"]
                retry_count = task.get("retry_count", 0)

                try:
                    await handler(task_type, payload)
                    logger.info(f"Задача {task_type} выполнена успешно")
                except Exception as e:
                    retry_count += 1
                    if retry_count < task.get("max_retries", self.max_retries):
                        delay = self.base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Повтор {task_type} через {delay:.1f}s (попытка {retry_count})")
                        task["retry_count"] = retry_count
                        await asyncio.sleep(delay)
                        await self.redis.lpush(RETRY_QUEUE_KEY, json.dumps(task))
                    else:
                        logger.error(f"Задача {task_type} исчерпала лимит повторов: {e}")
            except Exception as e:
                logger.error(f"Ошибка в цикле RetryWorker: {e}")
                await asyncio.sleep(5)
