# Артефакты ИИ и ML (AI/ML Specifications)

## 2.1. Библиотека системных промптов (Prompt Library)

**Цель:** Централизованное хранение, версионирование и управление промптами. Промпты — это "логика" агента. Они хранятся в Git и загружаются в LangGraph как конфигурация.

### Структура файла: `prompts/voicegraph_v1.0.yaml`

```yaml
version: "1.0.0"
created_at: "2026-06-12"

# --- Базовый системный промпт (Применяется ко всем узлам) ---
system_core: |
  Ты — автономный AI-менеджер кампаний VoiceGraph.
  Твоя цель: {campaign_goal}.
  
  ## Правила безопасности (Compliance):
  1. 152-ФЗ: Никогда не запрашивай и не храни полные паспортные данные, номера карт или СНИЛС.
  2. 38-ФЗ: Проверяй наличие согласия перед совершением целевого действия.
  3. Тон: Будь вежлив, эмпатичен, но не назойлив. Избегай слов "обязан", "должен".
  4. Если клиент просит человека: Немедленно предложи перевод на оператора, не пытайся удержать.

  ## Контекст памяти (Episodic Memory):
  {mem0_facts}
  Используй эти факты для персонализации, но не упоминай их напрямую, если это смущает клиента.
  Пример: Если есть факт "жаловался на доставку", скажи: "Я вижу, у нас были вопросы по доставке, все ли сейчас в порядке?", а не "В базе записано, что вы жаловались...".

  ## Формат ответа:
  Строго следуй JSON-схеме, определенной в инструментах. Никакого текста вне JSON при вызове инструментов.

# --- Промпт узла PLANNING (Генерация скриптов) ---
planning_node: |
  Твоя задача: Сгенерировать 3-5 вариантов сценария (Script Variants) для кампании "{campaign_name}".
  
  Целевая аудитория: {target_segment_description}
  Продукт/Услуга: {product_context}
  
  Требования к скриптам:
  1. Variant A (Direct): Краткий, прямой вопрос (для сегмента LTV=High).
  2. Variant B (Empathic): Мягкое начало, упоминание контекста из памяти (для тех, кто недавно жаловался).
  3. Variant C (Benefit): Акцент на выгоде (для "холодной" базы).
  
  Ограничения:
  - Длительность вступления не более 15 секунд.
  - Обязательно включать фразу о записи разговора в начале.
  - Каждый вариант должен иметь уникальный `script_id`.

# --- Промпт узла REFLECTION (Анализ отказов) ---
reflection_node: |
  Проанализируй транскрипт неудачного звонка и извлеки инсайт для улучшения кампании.
  
  Транскрипт (PII-замаскированный):
  """
  {transcript_masked}
  """
  
  Метаданные:
  - Outcome: {outcome}
  - Emotion: {max_sentiment_score}
  - Script Used: {script_id}
  
  Инструкция:
  1. Определи корневую причину отказа (Root Cause).
  2. Если причина в скрипте (например, вопрос задан слишком рано), предложи конкретную правку текста.
  3. Оцени уверенность (0.0 - 1.0).
  
  Верни результат строго в формате JSON модели `ReflectionInsight`.
```

---

## 2.2. Feature Dictionary для модели CatBoost

**Цель:** Единый реестр признаков (Feature Store Definition) для ML-инженеров. Описывает, откуда брать данные, как их трансформировать и какие типы использовать.

### Таблица признаков (Feature Registry)

| Категория | Имя признака (Feature Name) | Тип данных | Логика трансформации | Источник данных | Важность (Expected) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Time Context** | `hour_of_day` | Int | Час начала звонка (0-23) | Текущее время | 🌟🌟🌟🌟🌟 |
| | `is_weekend` | Bool | 1 если Сб/Вс, иначе 0 | Текущее время | 🌟🌟🌟 |
| | `days_since_last_call` | Float | `NOW() - MAX(call_logs.created_at)` | `call_logs` | 🌟🌟🌟🌟🌟 |
| **Historical** | `total_calls_30d` | Int | Count звонков за последние 30 дней | `call_logs` | 🌟🌟🌟 |
| | `success_rate_90d` | Float | Доля `outcome='SUCCESS'` за 90 дней | `call_logs` | 🌟🌟🌟🌟 |
| | `avg_duration_sec` | Float | Средняя длительность успешных звонков | `call_logs` | 🌟🌟 |
| **Behavioral** | `barge_in_ratio_30d` | Float | Отношение кол-ва прерываний к общему числу звонков | `call_logs` (event logs) | 🌟🌟🌟🌟 |
| | `last_sentiment` | Cat | Последняя эмоция: `CALM`, `ANGRY`, etc. | `call_logs` | 🌟🌟🌟 |
| **Demographic** | `ltv_segment` | Cat | `users.ltv_segment` (Premium/Standard/Low) | `users` | 🌟🌟🌟 |
| | `region_code` | Int | Код региона из номера телефона | `users` (derived from phone) | 🌟🌟 |
| **Campaign** | `script_variation_id` | Cat | ID скрипта (для A/B/n теста) | `campaigns` state | 🌟🌟🌟🌟🌟 |
| | `consent_age_days` | Int | Сколько дней прошло с момента получения согласия | `consents` table | 🌟🌟🌟🌟 |

### Пример SQL-запроса для генерации Feature Set:
```sql
-- Запрос для обучения модели (Feature Engineering Pipeline)
SELECT 
    u.id AS user_id,
    EXTRACT(HOUR FROM NOW()) AS hour_of_day,
    EXTRACT(DOW FROM NOW()) IN (0, 6) AS is_weekend,
    COUNT(cl.id) FILTER (WHERE cl.created_at > NOW() - INTERVAL '30 days') AS total_calls_30d,
    AVG(cl.duration_sec) FILTER (WHERE cl.outcome = 'SUCCESS' AND cl.created_at > NOW() - INTERVAL '90 days') AS avg_duration_sec,
    u.ltv_segment,
    -- Целевая переменная
    CASE WHEN cl.outcome = 'SUCCESS' THEN 1 ELSE 0 END AS target_success
FROM users u
LEFT JOIN call_logs cl ON u.id = cl.user_id
WHERE u.consent_to_call = TRUE
GROUP BY u.id, u.ltv_segment, cl.outcome;
```

---

## 2.3. Золотой набор данных (Golden Dataset Structure)

**Цель:** Эталонный набор данных для первичной оценки качества ASR, валидации PII-маскирования и тестирования Reflection Agent. Должен содержать сложные кейсы (шум, перебивания, попытки мошенничества).

### Структура файла: `golden_dataset_v1.csv`

```csv
session_id, audio_file_path, transcript_original, transcript_masked, outcome, sentiment_label, barge_in_count, contains_pii
sess_001_wav, /data/audio/sess_001.wav, "Алло? Да слушаю. Нет, карта 4276 5500 1234 9988 не нужна мне.", "Алло? Да слушаю. Нет, карта [CARD_NUMBER] не нужна мне.", REFUSAL, ANNOYED, 1, TRUE
sess_002_wav, "/data/audio/sess_002.wav", "Здравствуйте. Это опрос. Как вы оцениваете доставку? (Клиент перебивает) Да нормально все, 10 баллов.", "Здравствуйте. Это опрос. Как вы оцениваете доставку? (Клиент перебивает) Да нормально все, 10 баллов.", SUCCESS, CALM, 1, FALSE
sess_003_wav, "/data/audio/sess_003.wav", "... (Шум стройки) ... да я не слышу вас ... переснимите паспорт серии 4500 номер 123456 ...", "... (Шум стройки) ... да я не слышу вас ... переснимите паспорт серии [PASSPORT] ...", ERROR, CONFUSED, 0, TRUE
sess_004_wav, "/data/audio/sess_004.wav", "Кто это? Я уже говорил, что не интересуюсь вашими услугами! Отстаньте!", "Кто это? Я уже говорил, что не интересуюсь вашими услугами! Отстаньте!", REFUSAL, ANGRY, 2, FALSE
```

### Чек-лист качества для Golden Dataset:
1. **Разнообразие:** Минимум 20% записей должны содержать фоновый шум или акцент (для тестов ASR).
2. **PII Coverage:** Минимум 10 записей должны содержать разные типы ПДн (телефон, карта, паспорт, ИНН) для проверки Presidio.
3. **Outcome Balance:** Соотношение Success/Refusal/Error должно быть 40/50/10 (отражает реальность лучше, чем чистые данные).
4. **Разметка эмоций:** Каждая запись должна иметь лейбл `sentiment_label`, проставленный человеком (для валидации OpenSMILE).
