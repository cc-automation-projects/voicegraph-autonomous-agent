Текущая архитектура и план реализации проекта `VoiceGraph: Autonomous Predictive Outbound Agent` находятся на уровне **Senior/Principal Engineer blueprint**. Это отличный фундамент. Однако переход от "working prototype" к "production-hardened enterprise system" требует закрытия нескольких критических зазоров, которые обычно проявляются только под реальной нагрузкой и в условиях регуляторного аудита.

Ниже приведен детальный разбор улучшений, сгруппированный по слоям системы, с конкретными паттернами реализации, конфигурациями и приоритетами.

---
## 🔧 1. Архитектура и Инфраструктура

| Текущее состояние | Риск в Production | Решение (Production-Ready) |
|---|---|---|
| **vLLM на 4×H100 как единый Deployment** | SPOF. При OOM, драйверном крэше или плановом обновлении кластер голосовых агентов встает. | **LLM Router + Multi-Region Fallback**. Использовать `LiteLLM` или кастомный прокси-слой с маршрутизацией: `vLLM (on-prem)` → `YandexGPT Pro (cloud)` → `Qwen-7B (cheap fallback)`. Настроить health-check с `max_fallback_latency=500ms`. |
| **Redis StatefulSet (1 реплика)** | При рестарте пода LangGraph теряет чекпоинты → кампании "разваливаются", состояния графов сбрасываются. | **Redis Cluster + Sentinel** или `DragonflyDB` (drop-in replacement, thread-per-core, совместим с RESP). Для LangGraph чекпоинтов включить `rdbcompression no` и `aof-use-rdb-preamble yes` для быстрого восстановления. |
| **Отсутствие Service Mesh** | Трафик между микросервисами идет в открытом виде. Сложно управлять retry, timeout, circuit breaker на уровне сети. | **Linkerd** (легче Istio, меньше overhead для voice). Включить mTLS по умолчанию, настроить `ServiceProfile` для `/predict` (timeout: `50ms`, retries: `1`, idempotent: `true`). |
| **GPU Scheduling в K8s** | Поды с vLLM могут быть выселены при автоскейлинге нод, causing inference interruptions. | Настроить `tolerations: [{key: "nvidia.com/gpu", effect: "NoSchedule"}]`, `podDisruptionBudget: {minAvailable: 1}`, и `topologySpreadConstraints` по зонам доступности. |

**📦 Конфиг-пример (Linkerd mTLS + Retries):**
```yaml
apiVersion: policy.linkerd.io/v1beta1
kind: HTTPRoute
metadata:
  name: propensity-service-retries
spec:
  parentRefs:
    - kind: Service
      name: propensity-service
  rules:
    - backendRefs:
        - kind: Service
          name: propensity-service
      timeouts:
        request: 50ms
      filters:
        - type: RequestRetry
          requestRetry:
            retries: 1
            retryableStatusCodes: [502, 503, 504]
            backoff: {maxMs: 200, jitter: true}
```

---
## 🎙️ 2. Голосовой конвейер (Real-Time Voice)

| Текущее состояние | Риск в Production | Решение (Production-Ready) |
|---|---|---|
| **Базовый VAD + TTS interrupt** | Race conditions: TTS продолжает буферизацию после VAD-триггера, клиент слышит "эхо" или наложение. | Реализовать **Audio State Machine**: `IDLE` → `TTS_BUFFERING` → `VAD_ACTIVE` → `FLUSH_QUEUE` → `ASR_STREAMING`. Использовать `LiveKit AudioTrack#clearQueue()` + `LLM cancellation token`. |
| **Opus/PCMU без AGC/NS/AEC** | Шум на линии, эхо, разная громкость → WER ASR растет до 40%, эмоции детектируются неверно. | Включить **WebRTC Native AudioProcessing**: `noise_suppression: true`, `echo_cancellation: true`, `auto_gain_control: {target_level: -3dB}`. Добавить `RNNoise` препроцессор на стороне VoiceWorker. |
| **Latency мониторинг только end-to-end** | При деградации невозможно понять, где bottleneck (ASR, LLM, сеть, TTS). | Внедрить **Span-per-Stage OTel**: `voice.asr.chunk_latency`, `voice.llm.ttft`, `voice.tts.first_byte`, `voice.network.rtt`. Агрегировать в Grafana heatmap. |
| **Fallback при потере пакетов > 3%** | Голосовой конвейер "захлебывается", клиент слышит роботизированные артефакты. | Реализовать **Adaptive Codec Switching**: при RTT > 150ms или loss > 3% переключаться на `G.722` (wideband, robust) или снижать ASR chunk size с 400ms до 200ms. |

---
## 🧠 3. AI/ML и LLM-Operations

| Текущее состояние | Риск в Production | Решение (Production-Ready) |
|---|---|---|
| **Промпты хранятся в YAML/Git** | Нет A/B тестирования, rollback, мониторинга "prompt drift". Обновление ломает 100% кампаний. | **Prompt Registry Service** (PostgreSQL + Git sync). Семантическое версионирование (`v1.2.0`), Canary rollout (5% → 20% → 100%), автоматический откат при падении `completion_rate` или росте `toxicity_score`. |
| **CatBoost переобучается каждые 24ч** | Concept drift не детектируется. Модель может деградировать незаметно, пока не упадет конверсия. | Внедрить **Drift Monitoring**: `PSI` для фич, `KL-divergence` для предсказаний. При `PSI > 0.15` → алерт, при `> 0.25` → авто-переобучение на свежем батче. |
| **Structured Output без гарантий** | LLM иногда возвращает JSON с лишними полями, broken syntax → парсер падает, граф зависает. | **Retry + Fallback Pipeline**: 1) Strict JSON schema validation (`pydantic-core`), 2) При fail → retry с `temperature=0.05`, 3) При 3 fail → fallback на deterministic template + алерт в SRE. |
| **Нет мониторинга токенов/стоимости** | Кампании могут незаметно сжечь бюджет на GPU/TTS при росте нагрузки или галлюцинациях. | Ввести **Cost Guardian**: метрики `tokens_in/sec`, `tokens_out/sec`, `tts_minutes`. Авто-пауза кампаний при превышении `daily_budget` или аномальном росте `avg_tokens_per_call`. |

---
## 🛡️ 4. Данные, Compliance и Безопасность

| Текущее состояние | Риск в Production | Решение (Production-Ready) |
|---|---|---|
| **Presidio + Regex для PII** | Пропускает сложные кейсы (разговорные формы, опечатки, смешанные языки). Риск штрафа по 152-ФЗ. | Добавить **Custom NER Model** (fine-tuned `ruBert` на датасете РФ-документов). Внедрить `Human Review Queue` для low-confidence matches. |
| **Audit Logs в PostgreSQL** | Админ БД может изменить/удалить записи. Не проходит юридический аудит. | Перенести логи в **WORM Storage** (MinIO Object Lock / S3 Compliance Mode). Добавить `SHA-256 chain hashing` между записями для tamper-evidence. |
| **Удаление данных по lifecycle** | `DELETE` не гарантирует невозможности восстановления. Риск при комплаенс-аудите. | Реализовать **Crypto-Shredding**: PII шифруется клиентским ключом (`KMS`). При удалении кампании удаляется только ключ → данные становятся криптографически невосстановимыми. |
| **Consent Check на уровне приложения** | Задержки или кэш могут привести к звонку без актуального согласия. | Вынести в **Consent Proxy** (Envoy + Redis cache). SIP-proxy отбрасывает `INVITE`, если `consent_version` отсутствует или `revoked_at` не NULL. |

---
## 🛠️ 5. DevOps, SRE и Production-готовность

| Текущее состояние | Риск в Production | Решение (Production-Ready) |
|---|---|---|
| **Тестирование на синтетических данных** | Реальный мир: акценты, фоновый шум, плохой SIP, пакетные потери, одновременные звонки. | **Chaos Engineering + Synthetic Voice Load**: `SIPp` + `ffmpeg` (генерация аудио с шумами/эхом/перебиваниями). Тесты: 5% packet loss, 200ms jitter, vLLM OOM injection. |
| **Деплой моделей/промптов "big bang"** | Новый промпт может сломать конверсию на 30% за 1 час. | **Argo Rollouts / Flagger** для AI-компонентов. Canary analysis: сравнивать `conversion_rate`, `latency`, `sentiment_distribution` с baseline. Авто-rollback при деградации > 10%. |
| **Отсутствие Runbooks** | При инциденте инженеры тратят часы на диагностику. | Создать **SRE Playbooks**: 1) "Latency > 1s: проверь vLLM cache hit, переключи на fallback LLM", 2) "Circuit Breaker OPEN: очисть retry queue, проверь CRM health", 3) "PII Leak: изолируй сессии, запусти крипто-стирание". |
| **Idempotency не везде** | Ретраи из-за сетевых таймаутов создают дубликаты сделок/задач в CRM. | Внедрить **Idempotency Keys** для всех state-changing вызовов: `X-Idempotency-Key: sha256(user_id+action+timestamp)`. Хранить в Redis с TTL 24h. |

---
## 📊 Приоритизация улучшений (Roadmap to Production)

| Приоритет | Что внедрять | Срок | Влияние |
|:---:|---|:---:|:---:|
| **P0** | Redis Cluster + Sentinel, Idempotency Keys, WebRTC AudioProcessing (NS/AGC/AEC), OTel per-stage spans | 2-3 недели | Предотвращает потерю данных, деградацию голоса, дает видимость bottleneck'ов |
| **P1** | LLM Router/Fallback, Structured Output Retry Pipeline, Prompt Registry + Canary, Drift Monitoring (PSI) | 3-4 недели | Гарантирует доступность AI, стабильность графов, безопасное обновление промптов |
| **P2** | WORM Audit Logs, Crypto-Shredding, Consent Proxy, Chaos Testing Pipeline | 4-6 недель | Проходит комплаенс-аудит, защищает от регуляторных штрафов, проверяет устойчивость |
| **P3** | Cost Guardian, Synthetic Voice Load Testing, Argo Rollouts for AI, SRE Playbooks | 6-8 недель | Оптимизирует бюджет, готовит к масштабированию, стандартизирует реакцию на инциденты |

---
## 💡 Ключевые "подводные камни" реального кода (чего нет в документации)

1. **GPU Memory Fragmentation в vLLM:** При длительной работе `prefix_caching` вызывает фрагментацию VRAM → OOM без видимых причин. *Решение:* `--max-model-len` + периодический `vLLM restart` через `max_age_secs` в K8s CronJob.
2. **LangGraph `interrupt` + Redis AOF:** При высокой нагрузке `fsync everysec` может добавить 50-100ms к latency сохранения состояния. *Решение:* Вынести чекпоинты на отдельный Redis-инстанс с `appendfsync no` (accept loss on crash, но сохранять бизнес-логику в БД).
3. **TTS Voice Consistency:** Silero меняет интонацию между чанками при стриминге → клиент слышит "роботизированные скачки". *Решение:* Буферизация полных предложений + `phoneme-level alignment` перед отправкой в TTS.
4. **CRM Rate Limits:** Битрикс24/amoCRM имеют жесткие лимиты (2-5 req/sec). Composio не всегда их учитывает. *Решение:* Local token bucket перед вызовом Composio + exponential backoff с jitter.
5. **Team Skill Gap:** Проект требует 3+ senior-инженеров (WebRTC/Real-time, MLOps/LLM, Distributed Systems). *Реалистичный старт:* начать с managed-версий (LiveKit Cloud, Yandex Managed ClickHouse/Postgres, Yandex DataSphere для ML), а on-premise компоненты вводить поэтапно.
