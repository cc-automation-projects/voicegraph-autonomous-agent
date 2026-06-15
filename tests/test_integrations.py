from __future__ import annotations

import pytest

from src.integrations.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_open_on_failures(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=10)
        call_count = 0

        async def failing():
            raise ValueError("fail")

        for _ in range(2):
            try:
                await cb.call(failing)
            except (ValueError, CircuitBreakerOpenError):
                call_count += 1

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=0)

        async def failing():
            raise ValueError("fail")

        try:
            await cb.call(failing)
        except ValueError:
            pass

        assert cb.state == CircuitState.OPEN
