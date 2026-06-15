Ниже представлена полная, производственная (production-ready) структура проекта **VoiceGraph: Autonomous Predictive Outbound Agent**. 

Структура организована как **модульный монорепозиторий (Monorepo)**, что является лучшей практикой для сложных AI-систем: она обеспечивает единое управление зависимостями, сквозное тестирование и простоту CI/CD, при этом четко разделяя ответственность между микросервисами (Voice Worker, Orchestrator, Propensity Model и т.д.).

---

# 📁 Структура проекта VoiceGraph

```text
voicegraph/
├── .env.example                  # Шаблоны переменных окружения (без секретов)
├── .gitignore                    # Исключения для Git (venv, __pycache__, .env)
├── .dvcignore                    # Исключения для DVC (чтобы не версионировать временные файлы)
├── dvc.yaml                      # Пайплайны DVC (этапы ETL и обучения)
├── Makefile                      # Скрипты для быстрой разработки (make dev, make test, make deploy)
├── pyproject.toml                # Корневая конфигурация проекта (зависимости, ruff, mypy, pytest)
│
├── data/                         # Управляемые DVC данные (хранятся в MinIO/S3, в Git только .dvc файлы)
│   ├── raw/
│   │   └── crm_history_12m.csv.dvc
│   ├── processed/
│   │   ├── crm_history_12m_clean.parquet.dvc
│   │   ├── users_latest.parquet.dvc
│   │   └── call_logs_latest.parquet.dvc
│   └── models/
│       └── emotion_classifier.joblib.dvc  # Скомпилированная модель эмоций
│
├── docs/                         # Документация и артефакты спецификаций
│   ├── api/
│   │   └── propensity_openapi.yaml        # Спецификация API для ML-сервиса скоринга
│   └── prompts/
│       └── voicegraph_v1.0.yaml           # Библиотека системных промптов (Planning, Reflection)
│
├── infra/                        # Инфраструктура как код (IaC) и конфигурации окружения
│   ├── asterisk/
│   │   ├── config/
│   │   │   ├── pjsip.conf                   # Настройка SIP-транков и эндпоинтов
│   │   │   └── extensions.conf              # Диалплан с AMD и MixMonitor (SIPREC)
│   │   └── upload_to_minio.sh               # Скрипт-хук для отправки записей в MinIO
│   ├── bridge/                   # Rust-микросервис WebRTC ↔ SIP моста
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs                      # Ядро моста: обработка INVITE, SDP, релей RTP
│   ├── k8s/                      # Kubernetes манифесты (Helm templates или Kustomize)
│   │   ├── vllm-deployment.yaml             # Деплой vLLM (TP=4, prefix caching)
│   │   ├── qdrant-statefulset.yaml          # StatefulSet для векторной БД с PVC
│   │   ├── redis-statefulset.yaml           # Redis с AOF persistence для чекпоинтов
│   │   └── propensity-hpa.yaml              # HorizontalPodAutoscaler для ML-сервиса
│   └── observability/
│       ├── prometheus.yml                   # Конфиг скрейпинга метрик
│       ├── alerting_rules.yml               # Правила алертинга (Latency, WER, Refusal Rate)
│       └── grafana_dashboards/
│           └── voicegraph_realtime_ops.json # JSON-шаблон дашборда Grafana
│
├── src/                          # Исходный код приложения (модульная архитектура)
│   ├── voicegraph/               # Общий пакет: утилиты, конфигурации, схемы
│   │   ├── __init__.py
│   │   ├── config.py                        # Pydantic-settings для управления конфигом
│   │   ├── schemas.py                       # Строгие Pydantic-модели (CampaignState, ReflectionInsight, MemoryFact)
│   │   └── observability/
│   │       ├── metrics.py                   # Prometheus метрики (VOICE_LATENCY_MS, BANDIT_WEIGHTS)
│   │       └── tracing.py                   # Настройка OpenTelemetry
│   │
│   ├── data_pipeline/            # ЭТАП 0.1: Аудит и нормализация данных
│   │   ├── __init__.py
│   │   ├── gx_setup.py                      # Программная инициализация Great Expectations
│   │   └── pipeline.py                      # ETL-скрипт: валидация, фильтрация 38-ФЗ, сохранение Parquet
│   │
│   ├── pii_sanitizer/            # ЭТАП 0.3: Маскирование PII (152-ФЗ)
│   │   ├── __init__.py
│   │   ├── recognizers.py                   # Кастомные паттерны Presidio (RU_PASSPORT, RU_INN)
│   │   └── service.py                       # Singleton PIISanitizer на базе Microsoft Presidio
│   │
│   ├── voice_worker/             # ЭТАП 1: Real-time Voice Pipeline
│   │   ├── __init__.py
│   │   ├── main.py                          # Точка входа LiveKit Agent Worker
│   │   ├── stt/
│   │   │   └── faster_whisper_plugin.py     # Кастомный плагин Streaming ASR
│   │   ├── llm/
│   │   │   └── vllm_plugin.py               # Кастомный плагин Streaming LLM с вызовом PIISanitizer
│   │   ├── tts/
│   │   │   └── silero_plugin.py             # Плагин TTS с интеллектуальной буферизацией по предложениям
│   │   └── emotion/
│   │       ├── detector.py                  # Ядро OpenSMILE eGeMAPSv02 + классификатор
│   │       └── pipeline.py                  # Асинхронный конвейер скользящего окна для аудио
│   │
│   ├── propensity_model/         # ЭТАП 2: Propensity Modeling & MLOps
│   │   ├── __init__.py
│   │   ├── config.py                        # Гиперпараметры CatBoost и пути к данным
│   │   ├── features.py                      # Генератор 50+ признаков (Feature Engineering)
│   │   ├── train.py                         # Скрипт обучения, калибровки (Platt Scaling) и логирования в MLflow
│   │   └── inference_api/
│   │       ├── main.py                      # FastAPI приложение для батчевого скоринга (<10 мс)
│   │       └── cache_manager.py             # Логика In-Memory Feature Cache
│   │
│   ├── orchestrator/             # ЭТАП 3: LangGraph Оркестрация и Бандиты
│   │   ├── __init__.py
│   │   ├── state.py                         # Определение CampaignState (TypedDict + Pydantic)
│   │   ├── checkpointer.py                  # Кастомный AsyncRedisSaver с msgspec сериализацией
│   │   ├── bandit_optimizer.py              # Реализация Thompson Sampling (scipy.stats.beta)
│   │   ├── nodes.py                         # Логика узлов: planner, human_approval, dialer, optimizing, reflecting
│   │   └── graph_builder.py                 # Сборка и компиляция StateGraph
│   │
│   ├── memory/                   # ЭТАП 4: Эпизодическая память (Mem0)
│   │   ├── __init__.py
│   │   ├── qdrant_setup.py                  # Скрипт инициализации коллекции и payload-индексов
│   │   ├── manager.py                       # Обертка над Mem0 (add_fact, get_facts с decay_factor)
│   │   └── context_builder.py               # Формирователь блока <MEMORY_CONTEXT> для промпта
│   │
│   ├── reflection/               # ЭТАП 5: Reflection Agent и самообучение
│   │   ├── __init__.py
│   │   ├── prompts.py                       # Шаблон промпта для анализа отказов
│   │   ├── event_processor.py               # Потребитель Redis Streams, запись в БД, триггер очереди
│   │   ├── llm_analyzer.py                  # Асинхронный воркер: вызов vLLM со Structured Output
│   │   └── weekly_aggregator.py             # Cron-джоб: кластеризация инсайтов и обновление few-shot примеров
│   │
│   ├── integrations/             # ЭТАП 6.1: Интеграции с CRM
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py               # Распределенный Redis Circuit Breaker
│   │   ├── crm_tools.py                     # LangGraph Tools (Composio Битрикс24/amoCRM) с fallback
│   │   └── retry_worker.py                  # Фоновый воркер для обработки crm_sync_retry_queue
│   │
│   └── reporting/                # ЭТАП 6.3: Генерация финального отчета
│       ├── __init__.py
│       ├── data_aggregator.py               # SQL-агрегация метрик кампании из PostgreSQL
│       ├── pdf_generator.py                 # Генерация PDF через WeasyPrint + Matplotlib (Base64 чарты)
│       └── delivery_service.py              # Отправка PDF в Telegram и прикрепление к CRM
│
├── tests/                        # Модульные и интеграционные тесты (зеркало src/)
│   ├── conftest.py               # Глобальные фикстуры pytest (моки БД, Redis, LLM)
│   ├── data_pipeline/
│   │   └── test_pipeline.py
│   ├── pii_sanitizer/
│   │   └── test_pii_sanitizer.py            # Тесты на маскирование карт, паспортов, телефонов
│   ├── voice_worker/
│   │   └── test_emotion_pipeline.py         # Тесты на аудио-файлах из Golden Dataset
│   ├── orchestrator/
│   │   ├── test_checkpointer.py             # Тест восстановления состояния графа после "падения"
│   │   ├── test_nodes.py                    # Тесты узлов с моками внешних API
│   │   └── test_bandit_optimizer.py         # Тест сходимости Thompson Sampling
│   ├── memory/
│   │   └── test_context_and_recall.py
│   ├── reflection/
│   │   ├── test_event_processor.py          # Тест триггера очереди и fail-fast PII валидации
│   │   └── test_llm_analyzer.py             # Тест парсинга Pydantic Structured Output
│   └── integrations/
│       └── test_crm_tools.py                # Тест открытия Circuit Breaker и fallback в очередь
│
└── fixtures/                     # Тестовые данные (Golden Dataset)
    └── audio/
        ├── sess_001_barge_in.wav
        ├── sess_002_heavy_accent.wav
        ├── sess_003_construction_noise.wav
        └── sess_004_pii_leak_attempt.wav
```

---

## 📝 Краткое описание ключевых файлов и их роли

### 1. Конфигурация и Управление (Root)
*   **`pyproject.toml`**: Единый источник истины для зависимостей Python. Использует `uv` или `poetry` для быстрого и детерминированного разрешения зависимостей. Включает конфигурации для `ruff` (линтинг), `mypy` (строгая типизация) и `pytest`.
*   **`Makefile`**: Содержит команды для разработчиков, например: `make dev-up` (поднимает docker-compose), `make test` (запускает pytest с coverage), `make dvc-pull` (загружает данные).

### 2. Инфраструктура (`infra/`)
*   **`infra/bridge/src/main.rs`**: Критически важный компонент. Написан на Rust для гарантии отсутствия сборщика мусора (GC pauses), что критично для задержки < 300 мс при barge-in. Управляет SIP-сигналингом и ретрансляцией RTP-пакетов.
*   **`infra/k8s/vllm-deployment.yaml`**: Содержит специфичные флаги vLLM (`--tensor-parallel-size 4`, `--enable-prefix-caching true`, `--gpu-memory-utilization 0.90`), необходимые для достижения TTFT < 300 мс.

### 3. Ядро Приложения (`src/`)
*   **`src/voicegraph/schemas.py`**: "Контракт" всей системы. Любые данные, передаваемые между микросервисами или в/из LLM, проходят через эти Pydantic-модели. Гарантирует, что LLM не "сломает" граф невалидным JSON.
*   **`src/orchestrator/checkpointer.py`**: Кастомная реализация `BaseCheckpointSaver` для LangGraph. Использует `msgspec` вместо `pickle` или `json` для сверхбыстрой и безопасной сериализации состояний в Redis с поддержкой AOF.
*   **`src/integrations/circuit_breaker.py`**: Защищает систему от каскадных сбоев. Если CRM (Битрикс24) начинает отвечать 5xx ошибками, breaker открывается, и задачи тихо складываются в `retry_queue`, не прерывая голосовой конвейер.

### 4. Данные и ML (`data/`, `src/data_pipeline/`, `src/propensity_model/`)
*   **`dvc.yaml`**: Описывает DAG (Directed Acyclic Graph) пайплайна данных: `raw_csv` -> `gx_validation` -> `clean_parquet` -> `feature_engineering` -> `catboost_model`. Позволяет воспроизвести обучение одной командой `dvc repro`.
*   **`src/propensity_model/inference_api/main.py`**: Высокопроизводительный FastAPI-сервис. Использует In-Memory кэш для признаков, чтобы гарантировать время ответа < 10 мс на батч из 1000 пользователей.

### 5. Тестирование (`tests/`, `fixtures/`)
*   **`fixtures/audio/`**: "Golden Dataset". Набор реальных (анонимизированных) аудиофайлов, покрывающих edge-cases: перебивания, сильный акцент, фоновый шум стройки, попытки диктовки PII-данных. Используется для регрессионного тестирования ASR и Emotion Detector.
*   **`tests/integrations/test_crm_tools.py`**: Доказывает, что при 5 последовательных ошибках API Circuit Breaker переходит в состояние `OPEN`, а данные корректно сохраняются в Redis для последующего ретрая.

---

## 🚀 Как начать работу с этой структурой (Workflow)

1. **Клонирование и инициализация:**
   ```bash
   git clone <repo_url>
   cd voicegraph
   make install  # Создает venv, ставит зависимости, настраивает pre-commit hooks
   ```
2. **Локальная разработка:**
   ```bash
   make dev-up   # Запускает docker-compose.dev.yml (Postgres, Redis, Qdrant, MinIO, Mock vLLM)
   make test     # Запускает весь набор тестов (Shift-Left Testing)
   ```
3. **Работа с данными (Data Scientist):**
   ```bash
   dvc pull      # Загружает актуальные clean.parquet файлы из S3/MinIO
   python -m src.propensity_model.train  # Запускает переобучение и логирует в MLflow
   ```
4. **Деплой (DevOps):**
   ```bash
   # CI/CD пайплайн (GitHub Actions / GitLab CI) автоматически:
   # 1. Запускает ruff/mypy/pytest
   # 2. Собирает Docker-образы
   # 3. Обновляет манифесты в k8s/ и применяет их через ArgoCD или kubectl
   ```

Эта структура полностью соответствует принципам **Production-Ready**, обеспечивает строгое соблюдение 152-ФЗ/38-ФЗ, гарантирует низкую задержку (<800 мс) и позволяет команде из 4-6 инженеров (Backend, ML, DevOps, QA) работать параллельно без конфликтов.