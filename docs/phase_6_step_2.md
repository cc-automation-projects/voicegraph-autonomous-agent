В сложных распределенных AI-системах, особенно в real-time voice, наблюдаемость (Observability) — это не просто "хорошо иметь", а критическое требование для выживания. Мы реализуем сквозной трейсинг (Tracing) через OpenTelemetry и бизнес-метрики через Prometheus, чтобы любая деградация качества (рост задержек, падение конверсии) фиксировалась и алертилась в Telegram за секунды.

---

# 🚀 ЭТАП 6.2: Observability и дашборды

## Шаг 1: Зависимости и подготовка окружения

Добавляем библиотеки OpenTelemetry в `pyproject.toml` микросервисов (FastAPI Propensity Service, Orchestrator, Voice Worker).

```toml
# Добавить в pyproject.toml
dependencies = [
    "opentelemetry-api>=1.24.0",
    "opentelemetry-sdk>=1.24.0",
    "opentelemetry-instrumentation-fastapi>=0.45b0",
    "opentelemetry-exporter-prometheus>=1.12.0",
    "prometheus-client>=0.20.0", # Для кастомных метрик, если OTel exporter недостаточно гибок
]
```

---

## Шаг 2: Инструментация кода (OpenTelemetry + Custom Metrics)

Создаем единый модуль инициализации, который настраивает трейсинг и регистрирует требуемые бизнес-метрики.

**Файл: `src/observability/metrics.py`**

```python
import logging
from prometheus_client import Histogram, Gauge, Counter, start_http_server
import time

logger = logging.getLogger(__name__)

# 1. Гистограмма задержки голосового конвейера (End-to-End: ASR -> LLM -> TTS)
# Целевой p95 < 800 мс
VOICE_LATENCY_MS = Histogram(
    'voicegraph_voice_latency_ms',
    'End-to-end latency of the voice pipeline in milliseconds',
    buckets=[100, 300, 500, 800, 1000, 1500, 2000, 3000]
)

# 2. Счетчик исходов звонков (для расчета конверсии и отказов)
CALL_OUTCOMES = Counter(
    'voicegraph_call_outcomes_total',
    'Total number of call outcomes by campaign and script',
    ['campaign_id', 'script_id', 'outcome'] # outcome: SUCCESS, REFUSAL, HANGUP, ERROR
)

# 3. Gauge для весов алгоритма Томпсона (Bandit)
BANDIT_WEIGHTS = Gauge(
    'voicegraph_bandit_script_weights',
    'Current Thompson Sampling weights (alpha/beta) for script variants',
    ['campaign_id', 'script_id', 'parameter'] # parameter: 'alpha' or 'beta'
)

# 4. Gauge для Word Error Rate (WER) ASR
# Обновляется асинхронным джобом, который периодически сверяет эталонные аудио с транскриптом
ASR_WER = Gauge(
    'voicegraph_asr_wer',
    'Word Error Rate of the ASR pipeline (0.0 to 1.0)',
    ['model_version']
)

def start_metrics_server(port: int = 8001):
    """Запускает HTTP-сервер для скрейпинга метрик Prometheus."""
    start_http_server(port)
    logger.info(f"✅ Prometheus metrics server started on port {port}")
```

**Интеграция в Voice Worker (пример использования):**
```python
from src.observability.metrics import VOICE_LATENCY_MS, CALL_OUTCOMES, BANDIT_WEIGHTS

async def process_call_turn(session_id, campaign_id, script_id):
    start_time = time.perf_counter()
    
    try:
        # ... логика ASR -> LLM -> TTS ...
        pass
    finally:
        # 1. Запись latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        VOICE_LATENCY_MS.observe(latency_ms)
        
        # 2. Запись исхода (вызывается при завершении звонка)
        # CALL_OUTCOMES.labels(campaign_id=campaign_id, script_id=script_id, outcome="SUCCESS").inc()

def update_bandit_metrics(campaign_id: str, weights: dict):
    """Вызывается из optimizing_node для обновления метрик Bandit."""
    for script_id, params in weights.items():
        BANDIT_WEIGHTS.labels(
            campaign_id=campaign_id, 
            script_id=script_id, 
            parameter="alpha"
        ).set(params["alpha"])
        
        BANDIT_WEIGHTS.labels(
            campaign_id=campaign_id, 
            script_id=script_id, 
            parameter="beta"
        ).set(params["beta"])
```

**Интеграция в FastAPI (Propensity Service):**
```python
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

app = FastAPI(title="VoiceGraph Propensity ML API")
# Автоматическая инструментация всех HTTP-запросов (latency, status codes)
FastAPIInstrumentor.instrument_app(app)
```

---

## Шаг 3: Конфигурация Prometheus (`prometheus.yml`)

Настраиваем Prometheus для сбора метрик с наших подов.

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'voicegraph-orchestrator'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - voicegraph-prod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: langgraph-orchestrator
        action: keep
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - target_label: __address__
        replacement: $1:8001 # Порт нашего metrics server

  - job_name: 'voicegraph-propensity-service'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - voicegraph-prod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: propensity-service
        action: keep
```

---

## Шаг 4: Правила алертинга и интеграция с Telegram

Создаем файл правил для Prometheus Alertmanager.

**Файл: `infra/prometheus/alerting_rules.yml`**

```yaml
groups:
  - name: VoiceGraphRealTimeOps
    rules:
      # Алерт 1: Превышение задержки голосового конвейера (SLA breach)
      - alert: HighVoiceLatency
        expr: histogram_quantile(0.95, rate(voicegraph_voice_latency_ms_bucket[2m])) > 1000
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "🚨 Критическая задержка голосового конвейера"
          description: "P95 latency превысил 1000 мс в течение 2 минут. Текущее значение: {{ $value }} мс. Проверьте нагрузку на vLLM или сеть."

      # Алерт 2: Аномально высокая доля отказов по конкретному скрипту
      - alert: HighScriptRefusalRate
        expr: |
          sum(rate(voicegraph_call_outcomes_total{outcome="REFUSAL"}[10m])) by (script_id)
          /
          sum(rate(voicegraph_call_outcomes_total[10m])) by (script_id)
          > 0.80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "⚠️ Высокая доля отказов по скрипту {{ $labels.script_id }}"
          description: "Доля отказов по скрипту {{ $labels.script_id }} превысила 80% за последние 10 минут. Возможно, скрипт некорректен или раздражает клиентов."

      # Алерт 3: Деградация качества ASR (WER > 25%)
      - alert: HighASRWordErrorRate
        expr: voicegraph_asr_wer > 0.25
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "⚠️ Деградация качества распознавания речи (ASR)"
          description: "Word Error Rate (WER) превысил 25%. Текущее значение: {{ $value }}. Проверьте качество SIP-транка или работу VAD."
```

**Фрагмент `alertmanager.yml` для Telegram:**
```yaml
route:
  receiver: 'telegram-critical'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h

receivers:
  - name: 'telegram-critical'
    webhook_configs:
      - url: 'http://alertmanager-telegram-bot:8080/alert' # Кастомный мост в Telegram Bot API
        send_resolved: true
```

---

## Шаг 5: Grafana Dashboard "VoiceGraph Real-time Ops"

Ниже приведен JSON-шаблон для импорта в Grafana. Он содержит 4 ключевых панели, соответствующих требованиям.

*(Скопируйте этот JSON и импортируйте его в Grafana через "Import Dashboard")*

```json
{
  "dashboard": {
    "title": "VoiceGraph Real-time Ops",
    "tags": ["voicegraph", "ai", "production"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Voice Pipeline Latency (P95, ms)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(voicegraph_voice_latency_ms_bucket[5m]))",
            "legendFormat": "P95 Latency"
          }
        ],
        "thresholds": [
          {"value": 800, "colorMode": "warning"},
          {"value": 1000, "colorMode": "critical"}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "title": "Call Outcomes Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (outcome) (rate(voicegraph_call_outcomes_total[1h]))",
            "legendFormat": "{{outcome}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "title": "Bandit Algorithm Weights (Alpha/Beta)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "voicegraph_bandit_script_weights{parameter=\"alpha\"}",
            "legendFormat": "{{script_id}} (Alpha)"
          },
          {
            "expr": "voicegraph_bandit_script_weights{parameter=\"beta\"}",
            "legendFormat": "{{script_id}} (Beta)"
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      },
      {
        "title": "ASR Word Error Rate (WER)",
        "type": "stat",
        "targets": [
          {
            "expr": "voicegraph_asr_wer",
            "legendFormat": "Current WER"
          }
        ],
        "thresholds": [
          {"value": 0.15, "colorMode": "ok"},
          {"value": 0.25, "colorMode": "critical"}
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
      }
    ],
    "refresh": "10s"
  }
}
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 6.2)

Прежде чем переходить к финальной **Подзадаче 6.3 (Генерация финального отчета)**, убедитесь, что:

- [ ] Микросервисы успешно экспортируют метрики на порт `8001` (проверено через `curl localhost:8001/metrics`).
- [ ] Prometheus успешно скрейпит эти эндпоинты (статус `UP` в Targets).
- [ ] В Grafana импортирован дашборд "VoiceGraph Real-time Ops", и все 4 панели отображают актуальные данные (или нули, если нет нагрузки, но без ошибок "No data").
- [ ] Сгенерирован тестовый алерт (например, искусственное повышение WER или Latency), и сообщение успешно доставлено в указанный Telegram-канал с корректным форматированием.
- [ ] В трейсах (Jaeger/Tempo, если настроен OTel Tracing) виден сквозной `Trace ID`, связывающий HTTP-запрос к Propensity API с записью в PostgreSQL.
