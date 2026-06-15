from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: int = 30,
        half_open_max_calls: int = 3,
        name: str = "default",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_sec)
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: datetime | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if (
                    self._last_failure_time
                    and datetime.now(timezone.utc) - self._last_failure_time > self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"CircuitBreaker[{self.name}] перешёл в HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"CircuitBreaker[{self.name}] OPEN")

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(f"CircuitBreaker[{self.name}] HALF_OPEN лимит исчерпан")
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_calls = 0
                    logger.info(f"CircuitBreaker[{self.name}] восстановлен -> CLOSED")

            return result

        except Exception:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = datetime.now(timezone.utc)

                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.OPEN
                    logger.warning(f"CircuitBreaker[{self.name}] HALF_OPEN неудача -> OPEN")

                if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"CircuitBreaker[{self.name}] превышен порог {self.failure_threshold} -> OPEN")

            raise


class CircuitBreakerOpenError(Exception):
    pass
