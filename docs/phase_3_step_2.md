Это "мозг" оркестратора. Здесь мы превращаем абстрактную бизнес-логику в конкретные, тестируемые асинхронные функции LangGraph. Мы строго соблюдаем контракт `CampaignState`, используем асинхронные вызовы для неблокирующей работы и реализуем механизм `interrupt` для Human-in-the-Loop.

---

# 🚀 ЭТАП 3.2: Проектирование узлов графа

## Шаг 1: Зависимости и вспомогательные утилиты

Для работы узлов нам понадобятся асинхронный HTTP-клиент, утилиты для работы с Redis Streams и алгоритм Томпсона.

**Файл: `src/orchestrator/utils.py`**
```python
import numpy as np
from typing import Dict, List
import redis.asyncio as redis

async def push_to_voice_queue(redis_client: redis.Redis, task: dict):
    """Отправляет задачу на обзвон в Redis Stream."""
    await redis_client.xadd("voice_tasks_stream", task, maxlen=10000, approximate=True)

def sample_thompson_script(bandit_weights: Dict[str, Dict[str, float]]) -> str:
    """
    Реализация Thompson Sampling для выбора скрипта.
    Возвращает script_id с наибольшей сэмплированной вероятностью успеха.
    """
    samples = {}
    for script_id, weights in bandit_weights.items():
        # Сэмплируем из Beta-распределения
        samples[script_id] = np.random.beta(weights["alpha"], weights["beta"])
    
    # Возвращаем script_id с максимальным значением
    return max(samples, key=samples.get)
```

---

## Шаг 2: Реализация узлов графа (Nodes)

Каждый узел принимает текущее `CampaignState` и возвращает **словарь с обновлениями** (deltas), которые LangGraph автоматически сольет с текущим состоянием.

**Файл: `src/orchestrator/nodes.py`**

```python
import logging
import httpx
import redis.asyncio as redis
from langgraph.types import interrupt, Command
from typing import Dict, Any, List

from src.orchestrator.state import CampaignState, ApprovalStatus
from src.orchestrator.utils import push_to_voice_queue, sample_thompson_script
from src.schemas import ReflectionInsight # Из data_api_contracts.md

logger = logging.getLogger(__name__)

# --- 1. PLANNER NODE ---
async def planner_node(state: CampaignState) -> Dict[str, Any]:
    """Генерирует пул кандидатов и варианты скриптов."""
    logger.info(f"[planner_node] Планирование кампании: {state['campaign_id']}")
    
    try:
        # 1. Получение candidate_pool из Propensity API
        # В реальном сценарии здесь берется список всех доступных user_id для кампании
        mock_user_ids = ["user-1", "user-2", "user-3"] 
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://propensity-service:8000/api/v1/ml/score/batch",
                json={"campaign_id": state["campaign_id"], "user_ids": mock_user_ids}
            )
            response.raise_for_status()
            candidate_pool = response.json()["scored_users"]
            
        # 2. Генерация скриптов через LLM (упрощенная эмуляция вызова vLLM)
        # В реальности: await call_vllm(prompt=planning_prompt.format(...))
        active_scripts = [
            {"script_id": "v1_direct", "text": "Здравствуйте! Это контроль качества. Оцените доставку от 1 до 10.", "tone": "direct"},
            {"script_id": "v2_empathic", "text": "Здравствуйте! Видим, что была задержка. Всё ли в порядке сейчас?", "tone": "empathic"},
            {"script_id": "v3_benefit", "text": "Здравствуйте! У нас для вас особое предложение по доставке.", "tone": "benefit"}
        ]
        
        # Инициализация весов Bandit для новых скриптов (alpha=1, beta=1)
        bandit_weights = {s["script_id"]: {"alpha": 1.0, "beta": 1.0} for s in active_scripts}

        return {
            "candidate_pool": candidate_pool,
            "active_scripts": active_scripts,
            "bandit_weights": bandit_weights,
            "human_approval_status": ApprovalStatus.PENDING,
            "current_user_index": 0
        }
    except Exception as e:
        logger.error(f"[planner_node] Ошибка: {e}")
        return {"error_message": str(e)}

# --- 2. HUMAN APPROVAL NODE ---
async def human_approval_node(state: CampaignState) -> Dict[str, Any]:
    """Приостанавливает граф и ждет подтверждения супервайзера."""
    logger.info("[human_approval_node] Ожидание подтверждения супервайзера...")
    
    # Эмуляция отправки уведомления в Telegram-бот
    # telegram_bot.send_message(f"Кампания {state['campaign_id']} готова. Аудитория: {len(state['candidate_pool'])}. Утвердить?")
    
    # Ключевая фича LangGraph: прерывание выполнения
    # Граф сохранит состояние в Redis и остановится здесь
    interrupt_value = interrupt("approval_required")
    
    # Когда супервайзер нажмет "Approve" в Telegram, бот отправит команду на возобновление:
    # Command(resume={"human_approval_status": "APPROVED"})
    # LangGraph передаст это значение в `interrupt_value`
    
    if isinstance(interrupt_value, dict) and "human_approval_status" in interrupt_value:
        new_status = interrupt_value["human_approval_status"]
        logger.info(f"[human_approval_node] Получен статус: {new_status}")
        return {"human_approval_status": new_status}
    
    # Если возобновили без явного статуса (fallback)
    return {"human_approval_status": state["human_approval_status"]}

# --- 3. DIALER NODE ---
async def dialer_node(state: CampaignState) -> Dict[str, Any]:
    """Инициирует звонок для следующего пользователя из пула."""
    idx = state["current_user_index"]
    pool = state["candidate_pool"]
    
    if idx >= len(pool):
        logger.info("[dialer_node] Пул кандидатов исчерпан. Переход к рефлексии.")
        return {"current_user_index": idx} # Сигнал для перехода к следующему узлу

    user = pool[idx]
    user_id = user["user_id"]
    
    # Выбор скрипта через Thompson Sampling
    script_id = sample_thompson_script(state["bandit_weights"])
    
    # Формирование задачи для VoiceWorker
    task = {
        "session_id": f"sess-{user_id}-{idx}",
        "user_id": user_id,
        "campaign_id": state["campaign_id"],
        "script_id": script_id,
        "priority_score": user["priority_score"]
    }
    
    # Отправка в Redis Stream (VoiceWorker слушает эту очередь)
    redis_client = redis.from_url("redis://redis-checkpointer:6379/0")
    await push_to_voice_queue(redis_client, task)
    
    logger.info(f"[dialer_node] Задача отправлена: user={user_id}, script={script_id}")
    
    # Возвращаем обновленный индекс для следующей итерации
    return {"current_user_index": idx + 1}

# --- 4. OPTIMIZING NODE ---
async def optimizing_node(state: CampaignState) -> Dict[str, Any]:
    """Обновляет веса Bandit на основе результатов последних звонков."""
    # В реальной системе этот узел читает агрегированные результаты из PostgreSQL 
    # или получает события из Redis Streams от VoiceWorker.
    # Для демонстрации эмулируем, что последний звонок (script_id = 'v1_direct') был успешным (SUCCESS).
    
    last_script_id = "v1_direct" # Эмуляция: берем из контекста последнего завершенного звонка
    outcome = "SUCCESS" # Эмуляция
    
    weights = state["bandit_weights"].copy()
    
    if last_script_id in weights:
        if outcome == "SUCCESS":
            weights[last_script_id]["alpha"] += 1.0
            logger.info(f"[optimizing_node] Успех! Увеличиваем alpha для {last_script_id}")
        else:
            weights[last_script_id]["beta"] += 1.0
            logger.info(f"[optimizing_node] Неудача. Увеличиваем beta для {last_script_id}")
            
    return {"bandit_weights": weights}

# --- 5. REFLECTING NODE ---
async def reflecting_node(state: CampaignState) -> Dict[str, Any]:
    """Анализирует отказы и генерирует инсайты для улучшения будущих кампаний."""
    logger.info("[reflecting_node] Анализ результатов и генерация инсайтов...")
    
    # Эмуляция вызова LLM для анализа транскрипта отказа
    # prompt = reflection_prompt.format(transcript_masked="...", outcome="REFUSAL")
    # llm_response = await call_vllm(prompt, response_format=ReflectionInsight)
    
    mock_insight = ReflectionInsight(
        session_id="sess-mock-123",
        root_cause="WRONG_TIMING",
        suggested_script_tweak="Сдвинуть время звонков на вечер, так как утром клиенты раздражены.",
        confidence_score=0.85,
        direct_quote_from_client="Я же сказал, что занят утром!"
    )
    
    insight_text = f"[{mock_insight.root_cause}] {mock_insight.suggested_script_tweak} (Confidence: {mock_insight.confidence_score})"
    
    # Благодаря Annotated[..., operator.add] в CampaignState, этот инсайт ДОБАВИТСЯ к существующим,
    # а не перезапишет список!
    return {"reflection_insights": [insight_text]}
```

---

## Шаг 3: Сборка графа с условными переходами

Теперь мы связываем узлы в единый граф, определяя логику переходов (особенно для узла одобрения).

**Файл: `src/orchestrator/graph_builder.py`** (Обновленная версия)

```python
from langgraph.graph import StateGraph, END
from src.orchestrator.state import CampaignState, ApprovalStatus
from src.orchestrator.checkpointer import AsyncRedisSaver
from src.orchestrator.nodes import (
    planner_node, human_approval_node, dialer_node, 
    optimizing_node, reflecting_node
)
import os

def build_campaign_graph() -> StateGraph:
    redis_url = os.getenv("REDIS_URL", "redis://redis-checkpointer:6379/0")
    checkpointer = AsyncRedisSaver(redis_url=redis_url)

    builder = StateGraph(CampaignState)

    # 1. Регистрация узлов
    builder.add_node("planner", planner_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("dialer", dialer_node)
    builder.add_node("optimizing", optimizing_node)
    builder.add_node("reflecting", reflecting_node)

    # 2. Определение ребер
    builder.set_entry_point("planner")
    builder.add_edge("planner", "human_approval")

    # Условный переход после human_approval
    def route_after_approval(state: CampaignState) -> str:
        if state["human_approval_status"] == ApprovalStatus.APPROVED:
            return "dialer"
        elif state["human_approval_status"] == ApprovalStatus.REJECTED:
            return END
        else:
            return "human_approval" # Остаемся в ожидании

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"dialer": "dialer", "human_approval": "human_approval", END: END}
    )

    # Цикл обзвона: dialer -> optimizing -> (если есть еще кандидаты) -> dialer
    # Для упрощения сделаем линейный переход, а условие исчерпания пула проверим внутри dialer_node
    # или добавим условное ребро из dialer:
    
    def route_after_dialer(state: CampaignState) -> str:
        if state["current_user_index"] >= len(state["candidate_pool"]):
            return "reflecting" # Пул закончен, идем анализировать
        return "optimizing" # Иначе оптимизируем веса и продолжаем

    builder.add_conditional_edges("dialer", route_after_dialer, {"optimizing": "optimizing", "reflecting": "reflecting"})
    
    # После оптимизации возвращаемся к dialer для следующего звонка
    builder.add_edge("optimizing", "dialer")
    
    # Завершение
    builder.add_edge("reflecting", END)

    # 3. Компиляция с чекпоинтером
    return builder.compile(checkpointer=checkpointer)
```

---

## Шаг 4: Модульное тестирование узлов (Unit Tests)

Критически важно протестировать узлы изолированно, мокая внешние вызовы (Propensity API, LLM, Redis).

**Файл: `tests/orchestrator/test_nodes.py`**

```python
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from src.orchestrator.state import CampaignState, ApprovalStatus
from src.orchestrator.nodes import planner_node, human_approval_node, dialer_node, optimizing_node

@pytest.fixture
def initial_state() -> CampaignState:
    return {
        "campaign_id": "camp-test-001",
        "target_goal": "NPS_Optimization",
        "candidate_pool": [],
        "active_scripts": [],
        "bandit_weights": {},
        "human_approval_status": ApprovalStatus.PENDING,
        "current_user_index": 0,
        "reflection_insights": [],
        "error_message": None
    }

@pytest.mark.asyncio
@patch("src.orchestrator.nodes.httpx.AsyncClient")
async def test_planner_node_success(mock_client_class, initial_state):
    # 1. Настройка мока для Propensity API
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "scored_users": [
            {"user_id": "u1", "priority_score": 0.9, "recommended_call_window": "18:00"},
            {"user_id": "u2", "priority_score": 0.7, "recommended_call_window": "19:00"}
        ]
    }
    mock_response.raise_for_status = AsyncMock()
    
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client_class.return_value.__aenter__.return_value = mock_client

    # 2. Вызов узла
    result = await planner_node(initial_state)

    # 3. Проверка обновлений состояния
    assert result["human_approval_status"] == ApprovalStatus.PENDING
    assert len(result["candidate_pool"]) == 2
    assert len(result["active_scripts"]) == 3
    assert "v1_direct" in result["bandit_weights"]
    assert result["error_message"] is None

@pytest.mark.asyncio
async def test_human_approval_node_interrupt(initial_state):
    # Подготавливаем состояние, как будто оно пришло от planner_node
    state = {**initial_state, "candidate_pool": [{"user_id": "u1", "priority_score": 0.9}]}
    
    # При первом вызове должен сработать interrupt
    # В pytest мы не можем легко перехватить interrupt без обертки графа, 
    # поэтому тестируем логику возобновления напрямую.
    
    # Эмуляция того, что LangGraph передает результат interrupt обратно в функцию
    # при возобновлении через Command(resume={"human_approval_status": "APPROVED"})
    with patch("src.orchestrator.nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"human_approval_status": "APPROVED"}
        
        result = await human_approval_node(state)
        
        assert result["human_approval_status"] == "APPROVED"
        mock_interrupt.assert_called_once_with("approval_required")

@pytest.mark.asyncio
@patch("src.orchestrator.nodes.push_to_voice_queue")
async def test_dialer_node_pushes_task(mock_push, initial_state):
    state = {
        **initial_state,
        "candidate_pool": [{"user_id": "u1", "priority_score": 0.9}],
        "bandit_weights": {"v1_direct": {"alpha": 1.0, "beta": 1.0}},
        "current_user_index": 0
    }
    
    result = await dialer_node(state)
    
    # Проверка, что задача была отправлена
    mock_push.assert_called_once()
    call_args = mock_push.call_args[0][1] # Второй аргумент - это task dict
    assert call_args["user_id"] == "u1"
    assert call_args["script_id"] == "v1_direct"
    
    # Проверка инкремента индекса
    assert result["current_user_index"] == 1

@pytest.mark.asyncio
async def test_optimizing_node_updates_weights(initial_state):
    state = {
        **initial_state,
        "bandit_weights": {"v1_direct": {"alpha": 1.0, "beta": 1.0}}
    }
    
    # Эмулируем, что внутри узла был зафиксирован SUCCESS для v1_direct
    with patch("src.orchestrator.nodes.sample_thompson_script", return_value="v1_direct"):
        # В реальной логике узла мы бы передали outcome, здесь тестируем саму логику обновления
        # Для теста вызовем логику напрямую или через мок последнего звонка
        pass # (Упрощено для примера, в проде тестируется через интеграционный тест графа)
        
    result = await optimizing_node(state)
    assert result["bandit_weights"]["v1_direct"]["alpha"] == 2.0 # Увеличилось на 1
    assert result["bandit_weights"]["v1_direct"]["beta"] == 1.0
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 3.2)

Прежде чем переходить к **Подзадаче 3.3 (Реализация Thompson Sampling как отдельного изолированного модуля или его интеграция)**, убедитесь, что:

- [ ] Все 5 узлов (`planner`, `human_approval`, `dialer`, `optimizing`, `reflecting`) реализованы как асинхронные функции, возвращающие корректные словари обновлений для `CampaignState`.
- [ ] Узел `human_approval` корректно использует `langgraph.types.interrupt` и обрабатывает возобновление с новым статусом.
- [ ] Узел `dialer` успешно формирует задачу и вызывает mock-функцию отправки в Redis Stream.
- [ ] Юнит-тесты (`test_nodes.py`) проходят на 100%, покрывая как успешные сценарии, так и обработку ошибок (например, падение Propensity API).
- [ ] Граф, собранный через `build_campaign_graph()`, компилируется без ошибок валидации типов LangGraph.
