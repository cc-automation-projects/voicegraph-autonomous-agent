Это ключевой момент, где "сырая" векторная база данных превращается в осмысленную гиперперсонализацию. Мы реализуем механизм, который перед началом диалога извлекает релевантные факты, применяет к ним штраф за устаревание (Decay) и аккуратно вплетает их в системный промпт LLM, а также механизм сохранения новых фактов после звонка.

---

# 🚀 ЭТАП 4.2: Auto-Recall и инъекция в контекст

## Шаг 1: Формирователь контекста памяти (Context Builder)

Сырые данные из Qdrant нужно превратить в компактный, легко читаемый для LLM формат. Создадим утилиту, которая генерирует блок `<MEMORY_CONTEXT>`.

**Файл: `src/memory/context_builder.py`**

```python
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def build_memory_context(facts: List[Dict[str, Any]]) -> str:
    """
    Преобразует список фактов из Mem0 в структурированный блок для системного промпта LLM.
    Факты уже отсортированы по final_score (с учетом decay_factor) в методе get_facts.
    """
    if not facts:
        return "<MEMORY_CONTEXT>\nНет предыдущих взаимодействий с этим клиентом.\n</MEMORY_CONTEXT>"

    context_lines = ["<MEMORY_CONTEXT>", "История взаимодействий и предпочтения клиента:"]
    
    for i, fact in enumerate(facts, 1):
        category = fact.get("category", "INFO").upper()
        text = fact.get("fact", "")
        score = fact.get("final_score", 0.0)
        
        # Добавляем факт только если его итоговый скор выше порога релевантности (например, 0.3)
        if score > 0.3:
            context_lines.append(f"{i}. [{category}] {text} (Релевантность: {score:.2f})")
            
    context_lines.append("</MEMORY_CONTEXT>")
    
    return "\n".join(context_lines)
```

---

## Шаг 2: Интеграция Auto-Recall в VoiceAgentWorker

В начале звонка, сразу после соединения и идентификации `user_id`, агент должен запросить память. Чтобы не блокировать асинхронный цикл обработки аудио, вызов к Mem0 выполняется через `asyncio.to_thread` (или нативный async-клиент, если используется).

**Файл: `src/voice_worker/session_manager.py`** (Новый или обновленный модуль)

```python
import asyncio
import logging
from src.memory.manager import memory_manager
from src.memory.context_builder import build_memory_context

logger = logging.getLogger(__name__)

async def initialize_session_with_memory(user_id: str, campaign_goal: str) -> str:
    """
    Инициализирует сессию звонка: извлекает память и формирует системный промпт.
    """
    logger.info(f"🧠 Загрузка эпизодической памяти для user_id={user_id}...")
    
    try:
        # Асинхронный вызов синхронного метода memory_manager через thread pool, 
        # чтобы не блокировать event loop VoiceWorker
        loop = asyncio.get_running_loop()
        facts = await loop.run_in_executor(
            None, 
            memory_manager.get_facts, 
            user_id, 
            5 # limit
        )
        
        memory_context = build_memory_context(facts)
        logger.debug(f"Сформированный контекст памяти:\n{memory_context}")
        
        # Формируем динамический системный промпт, подставляя контекст
        # (В реальном коде это загружается из prompts/voicegraph_v1.0.yaml)
        system_prompt = f"""Ты — автономный AI-менеджер кампаний VoiceGraph.
Твоя цель: {campaign_goal}.

## Правила безопасности (Compliance):
1. 152-ФЗ: Никогда не запрашивай и не храни полные паспортные данные, номера карт или СНИЛС.
2. 38-ФЗ: Проверяй наличие согласия перед совершением целевого действия.
3. Тон: Будь вежлив, эмпатичен, но не назойлив.
4. Если клиент просит человека: Немедленно предложи перевод на оператора.

## Контекст памяти:
{memory_context}

Используй эти факты для персонализации, но не упоминай их напрямую в стиле "В базе записано...". 
Например, если есть факт о жалобе на доставку, скажи: "Я вижу, у нас были вопросы по доставке, сегодня все пришло вовремя?".

Отвечай кратко, по существу.
"""
        return system_prompt
        
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке памяти для {user_id}: {e}. Использую промпт по умолчанию.")
        # Fallback: возвращаем базовый промпт без персонализации, чтобы не ломать звонок
        return f"Ты — вежливый AI-ассистент VoiceGraph. Твоя цель: {campaign_goal}."
```

*Интеграция в `VoiceAgentWorker` (из Этапа 1.2):*
```python
# Внутри entrypoint(ctx: JobContext) или при обработке события participant_connected:
# user_id извлекается из метаданных комнаты LiveKit или передается через SIP-заголовки
system_prompt = await initialize_session_with_memory(user_id="user-123", campaign_goal="NPS_Optimization")
agent = Agent(instructions=system_prompt)
```

---

## Шаг 3: Интеграция сохранения фактов в Reflection Agent

После завершения звонка `Reflection Agent` должен не только проанализировать отказ, но и выделить новые факты для сохранения в долговременную память.

**Файл: `src/orchestrator/nodes.py`** (Обновление `reflecting_node`)

```python
import logging
import asyncio
from typing import Dict, Any
from src.orchestrator.state import CampaignState
from src.schemas import ReflectionInsight, MemoryFact
from src.memory.manager import memory_manager

logger = logging.getLogger(__name__)

async def reflecting_node(state: CampaignState) -> Dict[str, Any]:
    """Анализирует транскрипт, генерирует инсайты и обновляет эпизодическую память."""
    logger.info("[reflecting_node] Анализ результатов и обновление памяти...")
    
    # Эмуляция получения данных последнего звонка из состояния или БД
    last_transcript_masked = state.get("last_transcript_masked", "")
    last_user_id = state.get("last_user_id", "unknown")
    last_outcome = state.get("last_call_outcome", "UNKNOWN")
    
    if not last_transcript_masked or last_outcome == "SUCCESS":
        logger.info("[reflecting_node] Звонок успешен или нет транскрипта. Пропуск глубокого анализа.")
        return {}

    # 1. LLM-анализ для извлечения инсайта (упрощенно)
    # В реальности: llm_response = await call_vllm(reflection_prompt, response_format=ReflectionInsight)
    mock_insight = ReflectionInsight(
        session_id=state.get("last_session_id", "sess-000"),
        root_cause="WRONG_TIMING",
        suggested_script_tweak="Сдвинуть время звонков на вечер.",
        confidence_score=0.85,
        direct_quote_from_client="Я же сказал, что занят утром!"
    )
    
    # 2. Выделение нового факта для памяти (если он есть)
    # Эвристика или отдельный вызов LLM для извлечения факта из транскрипта
    new_fact_text = None
    if "утром" in last_transcript_masked.lower() and "занят" in last_transcript_masked.lower():
        new_fact_text = "Клиент просит не звонить ему в утренние часы, он занят."

    if new_fact_text:
        logger.info(f"💾 Обнаружен новый факт для сохранения: {new_fact_text}")
        
        memory_fact = MemoryFact(
            user_id=last_user_id,
            fact=new_fact_text,
            category="PREFERENCE",
            confidence=mock_insight.confidence_score
        )
        
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                memory_manager.add_fact,
                memory_fact.user_id,
                memory_fact.fact,
                memory_fact.category,
                memory_fact.confidence
            )
            logger.info("✅ Факт успешно сохранен в Mem0/Qdrant.")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении факта в память: {e}")

    # 3. Возврат инсайта в состояние графа (Annotated[..., operator.add] добавит его в список)
    insight_text = f"[{mock_insight.root_cause}] {mock_insight.suggested_script_tweak} (Confidence: {mock_insight.confidence_score})"
    
    return {"reflection_insights": [insight_text]}
```

---

## Шаг 4: Модульное тестирование (Shift-Left Testing)

Проверяем корректность форматирования контекста и вызовов памяти.

**Файл: `tests/memory/test_context_and_recall.py`**

```python
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.memory.context_builder import build_memory_context
from src.orchestrator.nodes import reflecting_node

def test_build_memory_context_empty():
    context = build_memory_context([])
    assert "Нет предыдущих взаимодействий" in context

def test_build_memory_context_populated():
    facts = [
        {
            "fact": "Клиент просит звонить только после 18:00",
            "category": "PREFERENCE",
            "final_score": 0.85
        },
        {
            "fact": "Жаловался на грубость курьера месяц назад",
            "category": "COMPLAINT",
            "final_score": 0.45
        },
        {
            "fact": "Старый неактуальный факт",
            "category": "INFO",
            "final_score": 0.20 # Ниже порога 0.3, должен быть отфильтрован
        }
    ]
    context = build_memory_context(facts)
    
    assert "[PREFERENCE] Клиент просит звонить только после 18:00" in context
    assert "[COMPLAINT] Жаловался на грубость курьера месяц назад" in context
    assert "Старый неактуальный факт" not in context # Проверка фильтрации по скорy
    assert context.startswith("<MEMORY_CONTEXT>")
    assert context.endswith("</MEMORY_CONTEXT>")

@pytest.mark.asyncio
@patch("src.orchestrator.nodes.memory_manager")
async def test_reflecting_node_adds_fact(mock_memory_manager):
    # Настраиваем мок для add_fact
    mock_memory_manager.add_fact = MagicMock()
    
    state = {
        "last_transcript_masked": "Алло, я же говорил, что утром занят, не звоните мне до обеда!",
        "last_user_id": "user-999",
        "last_call_outcome": "REFUSAL",
        "last_session_id": "sess-abc"
    }
    
    result = await reflecting_node(state)
    
    # Проверяем, что add_fact был вызван с правильными аргументами
    mock_memory_manager.add_fact.assert_called_once()
    args = mock_memory_manager.add_fact.call_args[0]
    assert args[0] == "user-999"
    assert "утром занят" in args[1].lower()
    assert args[2] == "PREFERENCE"
    
    # Проверяем, что инсайт добавлен в состояние
    assert len(result["reflection_insights"]) == 1
    assert "WRONG_TIMING" in result["reflection_insights"][0]
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 4.2)

Прежде чем переходить к **ЭТАПУ 5 (Reflection Agent и самообучение)**, убедитесь, что:

- [ ] Функция `build_memory_context` корректно фильтрует факты с `final_score < 0.3` и оборачивает результат в теги `<MEMORY_CONTEXT>`.
- [ ] `VoiceAgentWorker` (или узел инициализации) успешно вызывает `memory_manager.get_facts` асинхронно, не блокируя основной поток обработки аудио.
- [ ] Сгенерированный `<MEMORY_CONTEXT>` успешно подставляется в `system_prompt` и передается в vLLM.
- [ ] `reflecting_node` успешно извлекает новые факты из замаскированного транскрипта и вызывает `memory_manager.add_fact` с корректной категоризацией.
- [ ] Юнит-тесты подтверждают, что старые факты отсеиваются, а новые успешно добавляются в мок-объект Mem0.
