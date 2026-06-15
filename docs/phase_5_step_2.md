Это этап, где система переходит от простого логирования к настоящему **самообучению**. Мы реализуем асинхронного потребителя, который превращает неструктурированные тексты отказов в строгие данные, и еженедельный джоб, который находит паттерны и автоматически "прокачивает" промпты для будущих кампаний.

---

# 🚀 ЭТАП 5.2: LLM-анализ и кластеризация инсайтов

## Шаг 1: Зависимости и подготовка окружения

Для гарантированного получения валидного JSON от LLM мы будем использовать библиотеку `instructor` (или нативный `response_format` OpenAI API, который полностью поддерживается vLLM). Это стандарт индустрии для Structured Output.

```toml
# Добавить в pyproject.toml модуля reflection-processor
dependencies = [
    "instructor>=1.3.0",      # Гарантированный Pydantic Structured Output для LLM
    "httpx>=0.27.0",
    "scikit-learn>=1.5.0",    # Для простой кластеризации инсайтов (опционально, если не хватает группировки по root_cause)
    "croniter>=2.0.0"         # Для планирования еженедельных джобов
]
```

---

## Шаг 2: Асинхронный потребитель Redis Streams (Reflection Worker)

Этот сервис постоянно слушает очередь `reflection_queue`, извлекает замаскированный транскрипт, отправляет его в vLLM со строгой схемой и сохраняет результат в PostgreSQL.

**Файл: `src/reflection/llm_analyzer.py`**

```python
import logging
import asyncpg
import redis.asyncio as redis
import instructor
import httpx
from pydantic import ValidationError
from typing import Dict, Any

from src.schemas import ReflectionInsight # Из data_api_contracts.md
from src.reflection.prompts import REFLECTION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class ReflectionAnalyzer:
    def __init__(self, db_dsn: str, redis_url: str, vllm_api_url: str):
        self.db_dsn = db_dsn
        self.redis_url = redis_url
        self.vllm_api_url = vllm_api_url
        self._db_pool: asyncpg.Pool | None = None
        self._redis: redis.Redis | None = None
        
        # Инициализация клиента LLM с поддержкой Structured Output (через instructor или нативный API)
        # vLLM поддерживает формат OpenAI, поэтому используем стандартный httpx + json schema
        self.client = httpx.AsyncClient(base_url=vllm_api_url, timeout=30.0)

    async def connect(self):
        self._db_pool = await asyncpg.create_pool(self.db_dsn)
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        logger.info("✅ ReflectionAnalyzer подключен к БД, Redis и vLLM")

    async def process_queue(self, group_name: str = "reflection_group", consumer_name: str = "worker_1"):
        """Основной цикл потребления задач из Redis Streams."""
        logger.info(f"👂 Слушаю очередь reflection_queue как {consumer_name}...")
        
        # Создание consumer group, если не существует
        try:
            await self._redis.xgroup_create("reflection_queue", group_name, id="0", mkstream=True)
        except redis.ResponseError:
            pass # Группа уже существует

        while True:
            try:
                # Чтение новых сообщений (блокирующий вызов на 2 секунды)
                messages = await self._redis.xreadgroup(
                    group_name, consumer_name, {"reflection_queue": ">"}, count=1, block=2000
                )
                
                if not messages:
                    continue

                stream_name, msg_list = messages[0]
                msg_id, msg_data = msg_list[0]
                
                await self._analyze_and_save(msg_id, msg_data)
                
                # Подтверждение обработки (ACK)
                await self._redis.xack("reflection_queue", group_name, msg_id)

            except Exception as e:
                logger.error(f"❌ Ошибка в цикле обработки очереди: {e}")
                await asyncio.sleep(5) # Backoff при сбоях

    async def _analyze_and_save(self, msg_id: str, data: Dict[str, str]):
        logger.info(f"🔍 Анализ инсайта для session_id={data['session_id']}")
        
        # 1. Формирование промпта
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            transcript_masked=data['transcript_masked'],
            outcome=data['outcome'],
            max_sentiment_score=data['max_sentiment_score'],
            script_id=data['script_id']
        )

        # 2. Вызов LLM со строгой схемой (vLLM поддерживает response_format)
        payload = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, # Низкая температура для детерминированного JSON
            "response_format": {
                "type": "json_object",
                "schema": ReflectionInsight.model_json_schema()
            }
        }

        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            llm_output = response.json()["choices"][0]["message"]["content"]
            
            # 3. Валидация через Pydantic (гарантия структуры)
            insight = ReflectionInsight.model_validate_json(llm_output)
            
            # 4. Сохранение в PostgreSQL
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO reflection_insights 
                    (campaign_id, root_cause, suggested_script_tweak, confidence_score, direct_quote_from_client)
                    VALUES ($1, $2, $3, $4, $5)
                """, data['campaign_id'], insight.root_cause, insight.suggested_script_tweak, 
                   insight.confidence_score, insight.direct_quote_from_client)
            
            logger.info(f"✅ Инсайт успешно сохранен: {insight.root_cause}")

        except ValidationError as e:
            logger.error(f"⚠️ LLM вернул невалидный JSON: {e}. Сырой вывод: {llm_output}")
            # В продакшене: отправка в Dead Letter Queue (DLQ) для ручного разбора
        except Exception as e:
            logger.error(f"❌ Ошибка при анализе или сохранении: {e}")
```

---

## Шаг 3: Шаблон промпта для Reflection (Prompt Library)

Выносим промпт в отдельный файл для удобного версионирования и A/B тестирования.

**Файл: `src/reflection/prompts.py`**

```python
REFLECTION_PROMPT_TEMPLATE = """
Ты — эксперт по анализу качества диалогов в контакт-центре VoiceGraph.
Проанализируй транскрипт неудачного звонка и извлеки структурированный инсайт для улучшения будущих кампаний.

Транскрипт (PII-замаскированный):
"""
{transcript_masked}
"""

Метаданные:
- Outcome: {outcome}
- Emotion: {max_sentiment_score}
- Script Used: {script_id}

Инструкция:
1. Определи корневую причину отказа (Root Cause). Выбери строго одну из: PRICING, WRONG_TIMING, AGENT_TONE, TECHNICAL_ISSUE, UNKNOWN.
2. Если причина в скрипте или тоне, предложи 1 конкретное, атомарное изменение текста или логики.
3. Оцени свою уверенность в этом выводе от 0.0 до 1.0.
4. Приведи короткую прямую цитату клиента (уже замаскированную), которая подтверждает твой вывод.

Верни результат СТРОГО в формате JSON, соответствующем схеме ReflectionInsight. Никакого текста вне JSON.
"""
```

---

## Шаг 4: Еженедельный джоб агрегации и обновления Few-Shot примеров

Этот скрипт запускается по расписанию (например, каждое воскресенье в 03:00 через Kubernetes CronJob). Он находит общие паттерны в инсайтах и обновляет конфигурацию, которую читает `planner_node`.

**Файл: `src/reflection/weekly_aggregator.py`**

```python
import asyncpg
import logging
import json
from collections import Counter
from typing import List, Dict

logger = logging.getLogger(__name__)

async def aggregate_and_update_prompts(db_dsn: str):
    """Агрегирует инсайты за неделю и обновляет few-shot примеры для planner_node."""
    logger.info("🚀 Запуск еженедельной агрегации инсайтов...")
    
    pool = await asyncpg.create_pool(db_dsn)
    
    async with pool.acquire() as conn:
        # 1. Получаем необработанные инсайты за последние 7 дней
        records = await conn.fetch("""
            SELECT root_cause, suggested_script_tweak, confidence_score 
            FROM reflection_insights 
            WHERE created_at >= NOW() - INTERVAL '7 days' 
              AND applied_to_campaign = FALSE
            ORDER BY confidence_score DESC
        """)
        
        if not records:
            logger.info("ℹ️ Нет новых инсайтов для обработки.")
            return

        # 2. Кластеризация / Группировка по root_cause
        cause_counts = Counter([r['root_cause'] for r in records])
        total = len(records)
        
        # 3. Формирование обновленных Few-Shot примеров
        # Берем топ-1 самый частый root_cause и топ-2 лучших совета (по confidence) для него
        top_cause = cause_counts.most_common(1)[0][0]
        top_tweaks = [
            r['suggested_script_tweak'] 
            for r in records if r['root_cause'] == top_cause
        ][:2]
        
        new_few_shot_example = {
            "scenario": f"Клиенты часто отказываются по причине: {top_cause} ({int((cause_counts[top_cause]/total)*100)}% случаев за неделю).",
            "bad_example": "Вы не хотите оформлять доставку? (Слишком прямо, вызывает отторжение)",
            "good_example": top_tweaks[0] if top_tweaks else "Скажите, удобно ли будет обсудить доставку в другое время?",
            "reasoning": f"Анализ показал, что {top_cause} является основным барьером. Предложенная формулировка снижает сопротивление."
        }

        # 4. Сохранение обновленного шаблона промпта в БД (или Redis)
        # Предполагается наличие таблицы prompt_templates (id, node_name, version, few_shot_examples JSONB)
        await conn.execute("""
            INSERT INTO prompt_templates (node_name, version, few_shot_examples, created_at)
            VALUES ('planner_node', 'auto_updated', $1, NOW())
            ON CONFLICT (node_name) 
            DO UPDATE SET few_shot_examples = EXCLUDED.few_shot_examples, 
                          version = prompt_templates.version + 1,
                          created_at = NOW()
        """, json.dumps([new_few_shot_example]))

        # 5. Помечаем инсайты как обработанные
        await conn.execute("""
            UPDATE reflection_insights 
            SET applied_to_campaign = TRUE 
            WHERE created_at >= NOW() - INTERVAL '7 days' AND applied_to_campaign = FALSE
        """)
        
        logger.info(f"✅ Промпт planner_node успешно обновлен. Обработано инсайтов: {total}")

    await pool.close()
```

---

## Шаг 5: Интеграция обновленных промптов в `planner_node`

Теперь модифицируем `planner_node` (из Этапа 3.2), чтобы он динамически подгружал эти улучшенные few-shot примеры.

**Файл: `src/orchestrator/nodes.py`** (Фрагмент обновления)

```python
async def planner_node(state: CampaignState) -> Dict[str, Any]:
    # ... (получение candidate_pool из Propensity API) ...
    
    # 1. Загрузка актуальных few-shot примеров из БД/Redis
    async with db_pool.acquire() as conn:
        prompt_config = await conn.fetchrow(
            "SELECT few_shot_examples FROM prompt_templates WHERE node_name = 'planner_node' ORDER BY version DESC LIMIT 1"
        )
        few_shots = prompt_config['few_shot_examples'] if prompt_config else []

    # 2. Формирование промпта с динамическими примерами
    few_shots_text = "\n".join([
        f"Сценарий: {ex['scenario']}\nПлохо: {ex['bad_example']}\nХорошо: {ex['good_example']}\nПричина: {ex['reasoning']}"
        for ex in few_shots
    ])
    
    final_planning_prompt = PLANNING_PROMPT_TEMPLATE.format(
        campaign_name=state["target_goal"],
        target_segment_description="...",
        product_context="...",
        few_shot_examples=few_shots_text # <-- Вставка самообученных знаний
    )
    
    # 3. Вызов LLM для генерации скриптов
    # active_scripts = await call_vllm(final_planning_prompt, response_format=ScriptVariants)
    
    return {
        "candidate_pool": candidate_pool,
        "active_scripts": active_scripts,
        # ...
    }
```

---

## Шаг 6: Модульное тестирование (Shift-Left Testing)

**Файл: `tests/reflection/test_llm_analyzer.py`**

```python
import pytest
import json
from unittest.mock import AsyncMock, patch
from src.schemas import ReflectionInsight

@pytest.mark.asyncio
@patch("src.reflection.llm_analyzer.httpx.AsyncClient.post")
async def test_reflection_analyzer_valid_json(mock_post):
    # 1. Мок ответа vLLM (строго валидный JSON)
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "session_id": "sess-123",
                    "root_cause": "WRONG_TIMING",
                    "suggested_script_tweak": "Перенести обзвон на вечернее время",
                    "confidence_score": 0.92,
                    "direct_quote_from_client": "Я же сказал, что утром занят!"
                })
            }
        }]
    }
    mock_response.raise_for_status = AsyncMock()
    mock_post.return_value = mock_response

    # 2. Имитация данных из Redis
    msg_data = {
        "session_id": "sess-123",
        "campaign_id": "camp-1",
        "transcript_masked": "Алло? Я же сказал, что утром занят! Не звоните.",
        "outcome": "REFUSAL",
        "max_sentiment_score": "ANNOYED",
        "script_id": "v1_direct"
    }

    # 3. Инициализация и вызов (упрощенно, без реального БД для теста)
    analyzer = ReflectionAnalyzer("postgresql://test", "redis://test", "http://vllm:8000")
    
    # Проверка валидации Pydantic на мок-ответе
    llm_content = mock_response.json()["choices"][0]["message"]["content"]
    insight = ReflectionInsight.model_validate_json(llm_content)
    
    assert insight.root_cause == "WRONG_TIMING"
    assert insight.confidence_score == 0.92
    assert "утром занят" in insight.direct_quote_from_client
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 5.2)

Прежде чем переходить к **ЭТАПУ 6 (Интеграции, Observability и сдача)**, убедитесь, что:

- [ ] `ReflectionAnalyzer` успешно потребляет сообщения из `reflection_queue`, отправляет запрос в vLLM и парсит ответ через `ReflectionInsight.model_validate_json` без ошибок.
- [ ] В случае, если LLM возвращает невалидный JSON (например, с лишним текстом вокруг), система логирует ошибку и не крашится (устойчивость к галлюцинациям).
- [ ] Инсайты корректно сохраняются в таблицу `reflection_insights` PostgreSQL.
- [ ] Скрипт `weekly_aggregator.py` успешно группирует инсайты, вычисляет проценты и обновляет запись в `prompt_templates`.
- [ ] `planner_node` успешно подгружает обновленные `few_shot_examples` и включает их в промпт при генерации новых скриптов.
- [ ] Юнит-тесты покрывают сценарии успешного парсинга и обработки невалидного ответа LLM.
