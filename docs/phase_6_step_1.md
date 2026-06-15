Это финальный штрих в обеспечении автономности системы. Агент не просто "разговаривает", он совершает реальные бизнес-действия. Ключевой вызов здесь — обеспечить **отказоустойчивость (Resilience)**. CRM-системы (Битрикс24, amoCRM) часто имеют строгие лимиты (rate limits) или временные сбои. Мы реализуем паттерн **Circuit Breaker** с асинхронной очередью повторных попыток (Dead Letter Queue / Retry Queue), чтобы сбои CRM никогда не прерывали голосовой конвейер.

---

# 🚀 ЭТАП 6.1: Интеграция через Composio и Circuit Breaker

## Шаг 1: Зависимости и подготовка окружения

Используем официальную интеграцию Composio для LangChain/LangGraph, которая автоматически преобразует действия CRM в инструменты с корректными JSON-схемами.

```toml
# Добавить в pyproject.toml модуля orchestrator / voice-worker
dependencies = [
    "composio-core>=0.5.20",
    "composio-langchain>=0.5.20", # Интеграция с LangGraph
    "redis>=5.0.0",
    "pydantic>=2.7.1"
]
```

*Предварительное действие (Setup):* 
1. Зарегистрировать приложение в [Composio Dashboard](https://app.composio.dev/).
2. Подключить интеграцию `BITRIX24` или `AMOCRM` через OAuth2.
3. Получить `COMPOSIO_API_KEY` и сохранить в секретах Kubernetes.

---

## Шаг 2: Реализация Circuit Breaker на базе Redis

Поскольку у нас распределенная система (несколько подов LangGraph), стандартный `pybreaker` в памяти не подойдет. Реализуем легкий асинхронный Circuit Breaker на базе Redis со скользящим окном.

**Файл: `src/integrations/circuit_breaker.py`**

```python
import redis.asyncio as redis
import logging
import time
from typing import Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class RedisCircuitBreaker:
    """
    Распределенный Circuit Breaker на базе Redis.
    Открывается, если получено > max_failures ошибок 5xx за window_seconds.
    """
    def __init__(self, name: str, redis_client: redis.Redis, max_failures: int = 5, window_seconds: int = 60, reset_timeout: int = 120):
        self.name = name
        self.redis = redis_client
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.reset_timeout = reset_timeout
        self.error_key = f"cb:{name}:errors"
        self.state_key = f"cb:{name}:state"

    async def get_state(self) -> CircuitState:
        state = await self.redis.get(self.state_key)
        if state is None:
            return CircuitState.CLOSED
        
        state_str = state.decode('utf-8')
        if state_str == CircuitState.OPEN.value:
            # Проверка, истек ли таймаут сброса
            ttl = await self.redis.ttl(self.state_key)
            if ttl <= 0:
                await self.redis.set(self.state_key, CircuitState.HALF_OPEN.value, ex=self.reset_timeout)
                return CircuitState.HALF_OPEN
            return CircuitState.OPEN
            
        return CircuitState(state_str)

    async def record_success(self):
        if await self.get_state() == CircuitState.HALF_OPEN:
            await self.redis.set(self.state_key, CircuitState.CLOSED.value)
        # Очистка счетчика ошибок при успехе
        await self.redis.delete(self.error_key)

    async def record_failure(self):
        current_state = await self.get_state()
        if current_state == CircuitState.OPEN:
            raise Exception("Circuit breaker is OPEN")

        # Инкремент счетчика ошибок в скользящем окне
        pipe = self.redis.pipeline()
        pipe.incr(self.error_key)
        pipe.expire(self.error_key, self.window_seconds)
        await pipe.execute()

        error_count = int(await self.redis.get(self.error_key) or 0)
        if error_count >= self.max_failures:
            logger.warning(f"⚠️ Circuit Breaker '{self.name}' OPENED из-за {error_count} ошибок за {self.window_seconds}с")
            await self.redis.set(self.state_key, CircuitState.OPEN.value, ex=self.reset_timeout)
            raise Exception("Circuit breaker is OPEN")

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        state = await self.get_state()
        if state == CircuitState.OPEN:
            raise Exception("Circuit breaker is OPEN. Request rejected.")
        
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            # Определяем, является ли ошибка 5xx (серверной)
            if "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e):
                await self.record_failure()
            raise e
```

---

## Шаг 3: Реализация инструментов LangGraph с защитой

Создаем инструменты, которые используют Composio, но обернуты в Circuit Breaker. При сбое данные сохраняются в Redis для последующей синхронизации.

**Файл: `src/integrations/crm_tools.py`**

```python
import json
import logging
import redis.asyncio as redis
from langchain_core.tools import tool
from composio_langchain import ComposioToolSet, App, Action
from src.schemas import UpdateCRMRecordInput # Из data_api_contracts.md
from src.integrations.circuit_breaker import RedisCircuitBreaker

logger = logging.getLogger(__name__)

# Инициализация Composio (глобально при старте приложения)
toolset = ComposioToolSet(api_key="YOUR_COMPOSIO_API_KEY")
bx24_tools = toolset.get_tools(app=App.BITRIX24, actions=[Action.BITRIX24_CRM_DEAL_ADD, Action.BITRIX24_TASK_ADD])

# Инициализация Circuit Breaker
redis_client = redis.from_url("redis://redis-checkpointer:6379/0")
crm_breaker = RedisCircuitBreaker(name="bitrix24_api", redis_client=redis_client, max_failures=5, window_seconds=60)

async def fallback_to_retry_queue(payload: dict, action_type: str):
    """Сохраняет задачу в очередь для последующей синхронизации при сбое CRM."""
    retry_payload = {
        "action_type": action_type,
        "payload": payload,
        "retry_count": 0,
        "timestamp": int(time.time())
    }
    await redis_client.lpush("crm_sync_retry_queue", json.dumps(retry_payload))
    logger.warning(f"💾 Действие {action_type} сохранено в очередь ретрая из-за сбоя CRM.")

@tool(args_schema=UpdateCRMRecordInput)
async def composio_bx24_create_deal(
    user_id: str, 
    nps_score: int, 
    notes_masked: str, 
    idempotency_key: str
) -> str:
    """Создает сделку в Битрикс24 при успешном NPS или продаже. Требует замаскированные данные."""
    try:
        async def _execute_deal_creation():
            # Вызов инструмента Composio
            result = await toolset.execute_action(
                action=Action.BITRIX24_CRM_DEAL_ADD,
                params={
                    "fields": {
                        "TITLE": f"NPS Опрос: {user_id}",
                        "COMMENTS": notes_masked, # Уже проверено на PII через Pydantic
                        "UF_CRM_NPS_SCORE": nps_score # Кастомное поле
                    }
                }
            )
            return result

        response = await crm_breaker.execute(_execute_deal_creation)
        return f"✅ Сделка успешно создана. ID: {response.get('id')}"
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания сделки: {e}")
        payload = {"user_id": user_id, "nps_score": nps_score, "notes_masked": notes_masked, "idempotency_key": idempotency_key}
        await fallback_to_retry_queue(payload, "CREATE_DEAL")
        return "⚠️ Временная проблема с CRM. Данные сохранены и будут синхронизированы позже."

@tool(args_schema=UpdateCRMRecordInput)
async def composio_bx24_add_task(
    user_id: str, 
    notes_masked: str, 
    idempotency_key: str
) -> str:
    """Создает задачу 'Перезвонить' с высоким приоритетом при сложном вопросе или жалобе."""
    try:
        async def _execute_task_creation():
            result = await toolset.execute_action(
                action=Action.BITRIX24_TASK_ADD,
                params={
                    "fields": {
                        "TITLE": f"Перезвонить клиенту: {user_id}",
                        "DESCRIPTION": notes_masked,
                        "PRIORITY": "2", # 2 = High в Битрикс24
                        "DEADLINE": "tomorrow"
                    }
                }
            )
            return result

        response = await crm_breaker.execute(_execute_task_creation)
        return f"✅ Задача успешно создана. ID: {response.get('id')}"
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания задачи: {e}")
        payload = {"user_id": user_id, "notes_masked": notes_masked, "idempotency_key": idempotency_key}
        await fallback_to_retry_queue(payload, "CREATE_TASK")
        return "⚠️ Временная проблема с CRM. Задача поставлена в очередь на синхронизацию."
```

---

## Шаг 4: Интеграция инструментов в LangGraph

Теперь передаем эти инструменты агенту. LangGraph автоматически обработает вызовы, а наши обертки обеспечат безопасность.

**Файл: `src/orchestrator/graph_builder.py`** (Фрагмент обновления)

```python
from langgraph.prebuilt import create_react_agent
from src.integrations.crm_tools import composio_bx24_create_deal, composio_bx24_add_task

def build_agent_with_tools():
    # Собираем инструменты
    tools = [composio_bx24_create_deal, composio_bx24_add_task]
    
    # Создаем агента с привязкой к инструментам
    # LLM будет сам решать, когда вызвать создание сделки или задачи на основе диалога
    agent = create_react_agent(
        model=vllm_client, # Ваш настроенный клиент vLLM
        tools=tools,
        state_modifier=state_modifier_function # Функция, добавляющая MEMORY_CONTEXT и Emotion
    )
    return agent
```

---

## Шаг 5: Фоновый воркер синхронизации (Retry Worker)

Этот асинхронный процесс постоянно мониторит очередь `crm_sync_retry_queue` и пытается выполнить отложенные действия, когда Circuit Breaker переходит в состояние `HALF_OPEN` или `CLOSED`.

**Файл: `src/integrations/retry_worker.py`**

```python
import asyncio
import json
import logging
import redis.asyncio as redis
from src.integrations.circuit_breaker import crm_breaker

logger = logging.getLogger(__name__)

async def crm_sync_retry_worker():
    """Фоновая задача для обработки отложенных действий CRM."""
    redis_client = redis.from_url("redis://redis-checkpointer:6379/0")
    
    logger.info("🔄 Запуск воркера синхронизации CRM...")
    
    while True:
        try:
            # Блокирующее чтение из очереди (timeout=5 сек)
            result = await redis_client.brpop("crm_sync_retry_queue", timeout=5)
            if not result:
                continue
                
            _, message = result
            task = json.loads(message)
            
            # Проверка состояния Circuit Breaker перед попыткой
            state = await crm_breaker.get_state()
            if state.value == "OPEN":
                # Возвращаем задачу в начало очереди, так как CRM все еще недоступна
                await redis_client.lpush("crm_sync_retry_queue", message)
                await asyncio.sleep(10) # Ждем перед следующей проверкой
                continue
            
            # Попытка выполнить действие (логика выполнения аналогична tools, но с извлечением из task['payload'])
            # Для краткости здесь псевдокод:
            # success = await execute_crm_action(task['action_type'], task['payload'])
            success = True # Эмуляция успеха
            
            if success:
                logger.info(f"✅ Успешно синхронизировано: {task['action_type']}")
            else:
                # Если снова ошибка, увеличиваем retry_count и возвращаем в очередь с задержкой (exponential backoff)
                task['retry_count'] += 1
                if task['retry_count'] < 5:
                    await asyncio.sleep(2 ** task['retry_count']) # Exponential backoff
                    await redis_client.lpush("crm_sync_retry_queue", json.dumps(task))
                else:
                    logger.error(f"🚨 Задача {task['action_type']} окончательно провалена после 5 попыток. Перемещена в DLQ.")
                    # await redis_client.lpush("crm_sync_dlq", json.dumps(task))

        except Exception as e:
            logger.error(f"❌ Ошибка в воркере синхронизации: {e}")
            await asyncio.sleep(5)

# Запуск воркера как отдельной asyncio задачи при старте приложения
# asyncio.create_task(crm_sync_retry_worker())
```

---

## Шаг 6: Модульное тестирование (Shift-Left Testing)

Проверяем, что Circuit Breaker корректно открывается и задачи попадают в очередь.

**Файл: `tests/integrations/test_crm_tools.py`**

```python
import pytest
import json
import redis.asyncio as redis
from unittest.mock import AsyncMock, patch
from src.integrations.crm_tools import composio_bx24_create_deal, crm_breaker

@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_fallbacks_to_queue():
    # 1. Настраиваем мок Composio на выброс 500-й ошибки
    with patch("src.integrations.crm_tools.toolset.execute_action") as mock_execute:
        mock_execute.side_effect = Exception("HTTP 500 Internal Server Error")
        
        redis_client = redis.from_url("redis://localhost:6379/0")
        
        # 2. Имитируем 5 последовательных сбоев, чтобы открыть Circuit Breaker
        for _ in range(5):
            with pytest.raises(Exception, match="Circuit breaker is OPEN"):
                await composio_bx24_create_deal(
                    user_id="u-1", nps_score=9, notes_masked="Все отлично", idempotency_key="key-1"
                )
        
        # 3. Проверяем, что задача ушла в очередь ретрая
        queue_length = await redis_client.llen("crm_sync_retry_queue")
        assert queue_length >= 1, "Задача не была сохранена в очередь ретрая при открытии Circuit Breaker"
        
        # 4. Проверяем содержимое очереди
        raw_task = await redis_client.lindex("crm_sync_retry_queue", 0)
        task = json.loads(raw_task)
        assert task["action_type"] == "CREATE_DEAL"
        assert task["payload"]["user_id"] == "u-1"
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 6.1)

Прежде чем переходить к **Подзадаче 6.2 (Observability и дашборды)**, убедитесь, что:

- [ ] Приложение Composio настроено, OAuth-токен для Битрикс24/amoCRM получен и валиден.
- [ ] Инструменты `composio_bx24_create_deal` и `composio_bx24_add_task` успешно регистрируются в LangGraph и имеют корректные Pydantic-схемы.
- [ ] Pydantic-валидатор `check_pii` в `UpdateCRMRecordInput` активно блокирует передачу незамаскированных данных в инструменты.
- [ ] `RedisCircuitBreaker` корректно переходит в состояние `OPEN` после 5 ошибок 5xx в течение 60 секунд.
- [ ] При открытом Circuit Breaker вызовы инструментов не падают с исключением, а возвращают понятное сообщение и сохраняют payload в список `crm_sync_retry_queue`.
- [ ] Фоновый `crm_sync_retry_worker` успешно извлекает задачи из очереди и имитирует их повторное выполнение при закрытии Circuit Breaker.
- [ ] Юнит-тесты подтверждают переходы состояний Circuit Breaker и корректность fallback-логики.
