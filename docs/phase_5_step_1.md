Это критически важный мост между выполнением звонка и интеллектуальным анализом. Система должна мгновенно и надежно фиксировать факт завершения звонка, гарантировать, что в базу попадает *только* замаскированный транскрипт, и асинхронно триггерить процесс рефлексии только для тех случаев, где это действительно необходимо (отказы, негативные эмоции), экономя вычислительные ресурсы LLM.

---

# 🚀 ЭТАП 5.1: Сбор мета-данных и триггер рефлексии

## Шаг 1: Зависимости и подготовка окружения

Создаем выделенный микросервис (или модуль в составе оркестратора) для обработки пост-колл событий. Используем `asyncpg` для максимально быстрого асинхронного взаимодействия с PostgreSQL.

```toml
# Добавить в pyproject.toml модуля reflection-processor
dependencies = [
    "asyncpg>=0.29.0",      # Высокопроизводительный асинхронный драйвер PostgreSQL
    "redis>=5.0.0",         # Асинхронный клиент для Redis Streams
    "pydantic>=2.7.1",      # Валидация входящих событий
    "structlog>=24.1.0"     # Структурированное логирование для трейсинга
]
```

---

## Шаг 2: Схема данных события завершения звонка (Pydantic)

Определяем строгий контракт для события, которое приходит от `VoiceAgentWorker` или вебхука LiveKit. Это предотвращает попадание "мусорных" данных в очередь и базу.

**Файл: `src/reflection/schemas.py`**

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import re

class CallEndedEvent(BaseModel):
    """Строгая схема события завершения звонка."""
    session_id: str = Field(description="Уникальный ID сессии LiveKit/LangGraph")
    campaign_id: str = Field(description="ID кампании")
    user_id: str = Field(description="ID пользователя")
    script_id: str = Field(description="ID использованного скрипта")
    duration_sec: int = Field(ge=0, description="Длительность звонка в секундах")
    
    outcome: Literal["SUCCESS", "REFUSAL", "HANGUP", "ERROR", "NO_ANSWER"] = Field(
        description="Итоговый статус звонка"
    )
    
    max_sentiment_score: Literal["CALM", "ANNOYED", "CONFUSED", "ANGRY", "UNKNOWN"] = Field(
        default="UNKNOWN",
        description="Максимально негативная эмоция, зафиксированная за звонок"
    )
    
    # ВАЖНО: Это поле ДОЛЖНО приходить уже замаскированным от VoiceWorker
    transcript_masked: str = Field(description="Полный транскрипт звонка с замененными PII-токенами")

    @field_validator('transcript_masked')
    @classmethod
    def ensure_pii_masked(cls, v: str) -> str:
        """Fail-fast проверка: если в транскрипте есть явные номера карт или паспортов, отклоняем событие."""
        # Простая эвристика для предотвращения утечек до попадания в БД
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', v):
            raise ValueError("Критическая ошибка: обнаружен незамаскированный номер карты в транскрипте!")
        if re.search(r'\b\d{4}\s?\d{6}\b', v):
            raise ValueError("Критическая ошибка: обнаружен незамаскированный номер паспорта в транскрипте!")
        return v
```

---

## Шаг 3: Асинхронный сервис логирования и триггеринга (Core Logic)

Реализуем класс, который атомарно (в рамках бизнес-логики) выполняет две задачи:
1. Записывает метаданные в PostgreSQL (таблица `call_logs` из `data_api_contracts.md`).
2. Если выполнены условия триггера, отправляет задачу в Redis Stream `reflection_queue`.

**Файл: `src/reflection/event_processor.py`**

```python
import asyncpg
import redis.asyncio as redis
import logging
import json
from typing import Dict, Any

from src.reflection.schemas import CallEndedEvent

logger = logging.getLogger(__name__)

class CallEventProcessor:
    """
    Обрабатывает события завершения звонка: логирует в БД и триггерит рефлексию при необходимости.
    """
    def __init__(self, db_dsn: str, redis_url: str):
        self.db_dsn = db_dsn
        self.redis_url = redis_url
        self._db_pool: asyncpg.Pool | None = None
        self._redis: redis.Redis | None = None

    async def connect(self):
        self._db_pool = await asyncpg.create_pool(self.db_dsn, min_size=2, max_size=10)
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        logger.info("✅ CallEventProcessor подключен к PostgreSQL и Redis")

    async def close(self):
        if self._db_pool:
            await self._db_pool.close()
        if self._redis:
            await self._redis.close()

    async def process_event(self, event: CallEndedEvent) -> Dict[str, Any]:
        """
        Основная логика обработки: запись в БД + условный триггер в очередь.
        """
        try:
            # 1. Асинхронная запись в PostgreSQL (таблица call_logs)
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO call_logs 
                    (session_id, campaign_id, user_id, script_id, duration_sec, outcome, max_sentiment_score, transcript_masked)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (session_id) DO UPDATE SET
                        outcome = EXCLUDED.outcome,
                        duration_sec = EXCLUDED.duration_sec,
                        max_sentiment_score = EXCLUDED.max_sentiment_score,
                        transcript_masked = EXCLUDED.transcript_masked
                """, 
                event.session_id, event.campaign_id, event.user_id, event.script_id, 
                event.duration_sec, event.outcome, event.max_sentiment_score, event.transcript_masked)
            
            logger.info(f"💾 Событие звонка {event.session_id} сохранено в БД (outcome={event.outcome})")

            # 2. Логика триггера рефлексии
            # Триггерим, если звонок НЕ успешен ИЛИ клиент был зол
            needs_reflection = (event.outcome != "SUCCESS") or (event.max_sentiment_score == "ANGRY")
            
            if needs_reflection:
                # Формируем payload для очереди. Добавляем timestamp для сортировки/дебага
                queue_payload = {
                    "session_id": event.session_id,
                    "campaign_id": event.campaign_id,
                    "user_id": event.user_id,
                    "script_id": event.script_id,
                    "outcome": event.outcome,
                    "max_sentiment_score": event.max_sentiment_score,
                    "transcript_masked": event.transcript_masked,
                    "trigger_reason": "OUTCOME_FAILURE" if event.outcome != "SUCCESS" else "ANGRY_SENTIMENT"
                }
                
                # Отправка в Redis Stream. maxlen=100000 предотвращает раздувание памяти
                await self._redis.xadd(
                    "reflection_queue", 
                    queue_payload, 
                    maxlen=100000, 
                    approximate=True
                )
                logger.info(f"🚀 Задача на рефлексию добавлена в очередь для session_id={event.session_id}")
                
                return {"status": "processed", "reflection_triggered": True}
            else:
                return {"status": "processed", "reflection_triggered": False}

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке события звонка {event.session_id}: {e}")
            # В продакшене здесь можно отправить событие в Dead Letter Queue (DLQ)
            raise
```

---

## Шаг 4: Интеграция с VoiceAgentWorker (Точка вызова)

Показываем, как `VoiceAgentWorker` (из Этапа 1.2) инициирует этот процесс сразу после завершения WebRTC-сессии.

**Файл: `src/voice_worker/lifecycle.py`** (Фрагмент)

```python
import httpx
import logging
from src.reflection.schemas import CallEndedEvent

logger = logging.getLogger(__name__)

async def notify_call_ended(
    session_id: str,
    campaign_id: str,
    user_id: str,
    script_id: str,
    duration_sec: int,
    outcome: str,
    max_sentiment: str,
    transcript_masked: str
):
    """
    Отправляет событие завершения звонка в Reflection Processor.
    Использует HTTP или прямой вызов, если они в одном кластере.
    """
    event = CallEndedEvent(
        session_id=session_id,
        campaign_id=campaign_id,
        user_id=user_id,
        script_id=script_id,
        duration_sec=duration_sec,
        outcome=outcome,
        max_sentiment_score=max_sentiment,
        transcript_masked=transcript_masked # Убеждаемся, что передаем ТОЛЬКО замаскированную версию
    )
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "http://reflection-processor:8000/api/v1/events/call_ended",
                json=event.model_dump()
            )
            response.raise_for_status()
            logger.info(f"✅ Уведомление о завершении звонка {session_id} успешно отправлено.")
    except Exception as e:
        logger.error(f"⚠️ Не удалось отправить событие завершения звонка {session_id}: {e}")
        # Graceful degradation: не роняем звонок, просто логируем ошибку. 
        # Данные можно восстановить из сырых логов LiveKit/MinIO позже.
```

---

## Шаг 5: Модульное тестирование (Shift-Left Testing)

Критически важно проверить, что триггер срабатывает *только* при нужных условиях и что защита от PII работает.

**Файл: `tests/reflection/test_event_processor.py`**

```python
import pytest
import asyncpg
import redis.asyncio as redis
from unittest.mock import AsyncMock, patch
from src.reflection.schemas import CallEndedEvent
from src.reflection.event_processor import CallEventProcessor

@pytest.fixture
def mock_db_and_redis():
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_conn.execute = AsyncMock()
    
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock()
    
    return mock_pool, mock_redis

@pytest.mark.asyncio
async def test_successful_call_no_reflection_trigger(mock_db_and_redis):
    mock_pool, mock_redis = mock_db_and_redis
    
    processor = CallEventProcessor("postgresql://test", "redis://test")
    processor._db_pool = mock_pool
    processor._redis = mock_redis
    
    event = CallEndedEvent(
        session_id="sess-123", campaign_id="camp-1", user_id="u-1", script_id="v1",
        duration_sec=45, outcome="SUCCESS", max_sentiment_score="CALM",
        transcript_masked="Здравствуйте, все отлично, спасибо. Оценка 10."
    )
    
    result = await processor.process_event(event)
    
    # Проверка записи в БД
    mock_pool.acquire.return_value.__aenter__.return_value.execute.assert_called_once()
    
    # Проверка, что в Redis Stream НИЧЕГО не было отправлено
    mock_redis.xadd.assert_not_called()
    assert result["reflection_triggered"] is False

@pytest.mark.asyncio
async def test_refusal_call_triggers_reflection(mock_db_and_redis):
    mock_pool, mock_redis = mock_db_and_redis
    
    processor = CallEventProcessor("postgresql://test", "redis://test")
    processor._db_pool = mock_pool
    processor._redis = mock_redis
    
    event = CallEndedEvent(
        session_id="sess-456", campaign_id="camp-1", user_id="u-2", script_id="v2",
        duration_sec=15, outcome="REFUSAL", max_sentiment_score="ANNOYED",
        transcript_masked="Мне это не интересно, отстаньте. Моя карта [CARD_NUMBER_REDACTED] вам не нужна."
    )
    
    result = await processor.process_event(event)
    
    # Проверка, что задача добавлена в очередь
    mock_redis.xadd.assert_called_once()
    call_args = mock_redis.xadd.call_args[0]
    assert call_args[0] == "reflection_queue"
    assert call_args[1]["trigger_reason"] == "OUTCOME_FAILURE"
    assert result["reflection_triggered"] is True

@pytest.mark.asyncio
async def test_pii_leak_prevention():
    """Проверка того, что Pydantic-валидация отклоняет незамаскированные данные."""
    with pytest.raises(ValueError, match="Критическая ошибка: обнаружен незамаскированный номер карты"):
        CallEndedEvent(
            session_id="sess-789", campaign_id="camp-1", user_id="u-3", script_id="v1",
            duration_sec=10, outcome="REFUSAL", max_sentiment_score="ANGRY",
            transcript_masked="Не звоните мне, моя карта 4276 5500 1234 9988 не будет использоваться." # Ошибка: должно быть [CARD_NUMBER_REDACTED]
        )
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 5.1)

Прежде чем переходить к **Подзадаче 5.2 (LLM-анализ и кластеризация инсайтов)**, убедитесь, что:

- [ ] Схема `CallEndedEvent` строго валидирует входящие данные и содержит `@field_validator` для предотвращения утечек PII (fail-fast).
- [ ] Сервис `CallEventProcessor` успешно записывает метаданные в таблицу `call_logs` PostgreSQL с использованием `asyncpg` (проверено через интеграционный тест или мок).
- [ ] Логика триггера корректно определяет необходимость рефлексии: `outcome != "SUCCESS"` ИЛИ `max_sentiment_score == "ANGRY"`.
- [ ] Задачи успешно добавляются в Redis Stream `reflection_queue` с корректным `maxlen` для предотвращения утечек памяти.
- [ ] Успешные звонки (`SUCCESS` + `CALM`) **не** попадают в очередь рефлексии, экономя токены LLM.
- [ ] Юнит-тесты покрывают как позитивные сценарии, так и сценарии отклонения из-за нарушения PII.
