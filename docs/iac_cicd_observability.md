# Инфраструктурные артефакты и DevOps (IaC, CI/CD, Observability)

## 3.1. Локальное окружение разработки (`docker-compose.dev.yml`)

**Цель:** Обеспечить разработчикам возможность поднять всю экосистему одной командой (`docker-compose up -d`) без необходимости иметь локальные GPU для LLM или сложные кластеры Kubernetes.

```yaml
version: '3.8'

services:
  # 1. Основная БД с поддержкой векторов и полнотекстового поиска
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: voicegraph_dev
      POSTGRES_PASSWORD: dev_password
      POSTGRES_DB: voicegraph
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql

  # 2. Redis для очередей (Streams) и кэширования состояний LangGraph (с AOF)
  redis:
    image: redis:7.2-alpine
    command: redis-server --appendonly yes --appendfsync everysec
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # 3. Векторная БД для Mem0 (Episodic Memory)
  qdrant:
    image: qdrant/qdrant:v1.8.4
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  # 4. S3-совместимое хранилище для аудио и логов (MinIO)
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  # 5. Mock vLLM Service (Эмулятор LLM для локальной разработки без GPU)
  # Возвращает предсказуемые ответы с искусственной задержкой для тестирования таймаутов
  mock-vllm:
    build: ./mocks/vllm_mock
    ports:
      - "8000:8000"
    environment:
      - MOCK_LATENCY_MS=300
      - MOCK_MODEL_NAME=qwen2.5-72b-mock

  # 6. Основной оркестратор LangGraph (с hot-reload для разработки)
  langgraph-orchestrator:
    build: 
      context: .
      dockerfile: Dockerfile.dev
    environment:
      - DATABASE_URL=postgresql://voicegraph_dev:dev_password@postgres:5432/voicegraph
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
      - LLM_API_URL=http://mock-vllm:8000/v1
      - ENV=development
    ports:
      - "8080:8080"
    volumes:
      - ./voicegraph:/app/voicegraph
    depends_on:
      - postgres
      - redis
      - qdrant
      - mock-vllm

volumes:
  pg_data:
  redis_data:
  qdrant_data:
  minio_data:
```

---

## 3.2. Kubernetes Манифесты (Production Deployment Snippets)

**Цель:** Предоставить шаблоны для Helm-чартов или Kustomize, обеспечивающие высокую доступность, масштабируемость и корректное использование GPU в продакшене.

### Сниппет A: Деплой vLLM (Инференс LLM)
*Критически важная конфигурация для распределения модели Qwen2.5-72B на 4x H100.*

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: voicegraph-prod
spec:
  replicas: 1 # Масштабируется вручную или через Karpenter из-за привязки к GPU
  selector:
    matchLabels:
      app: vllm-inference
  template:
    metadata:
      labels:
        app: vllm-inference
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:latest
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
          - "--model"
          - "Qwen/Qwen2.5-72B-Instruct"
          - "--tensor-parallel-size"
          - "4" # TP=4 для 4x GPU
          - "--enable-prefix-caching"
          - "true"
          - "--gpu-memory-utilization"
          - "0.90"
          - "--max-num-seqs"
          - "256"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 4 # Запрос 4 GPU
          requests:
            nvidia.com/gpu: 4
        env:
        - name: VLLM_WORKER_MULTIPROC_METHOD
          value: "spawn"
        # Монтирование секретов для HuggingFace/Model Hub (если нужно)
        volumeMounts:
        - name: model-cache
          mountPath: /.cache
      volumes:
      - name: model-cache
        emptyDir: {} # В продакшене заменить на PersistentVolume для быстрого старта
```

### Сниппет B: Horizontal Pod Autoscaler (HPA) для LiveKit Workers
*Автоматическое масштабирование голосовых воркеров на основе кастомной метрики.*

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: livekit-voice-worker-hpa
  namespace: voicegraph-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: livekit-voice-worker
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: active_webrtc_sessions # Кастомная метрика из Prometheus
      target:
        type: AverageValue
        averageValue: "15" # Масштабировать, если в среднем > 15 сессий на под
```

---

## 3.3. CI/CD Пайплайн (`.gitlab-ci.yml`)

**Цель:** Автоматизировать проверку кода, безопасность контейнеров и деплой, исключая человеческий фактор.

```yaml
stages:
  - lint
  - test
  - security_scan
  - build
  - deploy_staging

variables:
  DOCKER_DRIVER: overlay2
  IMAGE_TAG: $CI_COMMIT_SHORT_SHA

# 1. Линтинг и типизация
lint:
  stage: lint
  image: python:3.12-slim
  script:
    - pip install ruff mypy
    - ruff check voicegraph/
    - mypy voicegraph/ --strict

# 2. Юнит- и интеграционные тесты
test:
  stage: test
  image: python:3.12-slim
  services:
    - name: postgres:16
      alias: postgres
    - name: redis:7.2-alpine
      alias: redis
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_pass
    DATABASE_URL: postgresql://test_user:test_pass@postgres:5432/test_db
  script:
    - pip install -r requirements.txt pytest pytest-cov
    - pytest voicegraph/tests/ --cov=voicegraph --cov-report=xml
  coverage: '/TOTAL\s+\d+\s+\d+\s+(\d+%)/'

# 3. Сканирование безопасности контейнеров (Критично для 152-ФЗ)
security_scan:
  stage: security_scan
  image: aquasec/trivy:latest
  script:
    - trivy fs --severity HIGH,CRITICAL --exit-code 1 .
    - trivy image --severity HIGH,CRITICAL --exit-code 1 $CI_REGISTRY_IMAGE:latest

# 4. Сборка Docker-образа
build:
  stage: build
  image: docker:24-dind
  services:
    - docker:24-dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:$IMAGE_TAG -f Dockerfile.prod .
    - docker push $CI_REGISTRY_IMAGE:$IMAGE_TAG

# 5. Деплой на Staging (через kubectl или ArgoCD)
deploy_staging:
  stage: deploy_staging
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context staging-cluster
    - kubectl set image deployment/langgraph-orchestrator orchestrator=$CI_REGISTRY_IMAGE:$IMAGE_TAG -n voicegraph-staging
    - kubectl rollout status deployment/langgraph-orchestrator -n voicegraph-staging
  only:
    - main
```

---

## 3.4. Наблюдаемость и Мониторинг (Observability)

**Цель:** Обеспечить сквозную видимость (Trace ID) и мгновенное реагирование на аномалии (Prometheus + Grafana + OpenTelemetry).

### 3.4.1. Кастомные метрики Prometheus (`metrics.py`)
Реализация в коде через `prometheus_client`:

```python
from prometheus_client import Counter, Histogram, Gauge

# Гистограмма задержки голосового конвейера (цель: p95 < 800ms)
VOICE_LATENCY_MS = Histogram(
    'voicegraph_pipeline_latency_ms', 
    'End-to-end latency from ASR to TTS start',
    buckets=[100, 300, 500, 800, 1000, 1500, 2000]
)

# Счетчик успешных/неуспешных звонков по кампаниям
CALL_OUTCOME_COUNTER = Counter(
    'voicegraph_call_outcomes_total', 
    'Total calls by outcome and campaign',
    ['campaign_id', 'outcome'] # outcome: SUCCESS, REFUSAL, ERROR
)

# Gauge для весов алгоритма Томпсона (Bandit) в реальном времени
BANDIT_WEIGHT_GAUGE = Gauge(
    'voicegraph_bandit_script_weight', 
    'Current Thompson Sampling weight for script variants',
    ['campaign_id', 'script_id', 'parameter'] # parameter: alpha, beta
)

# Счетчик срабатываний PII-маскирования (для аудита безопасности)
PII_MASKING_EVENTS = Counter(
    'voicegraph_pii_masked_total',
    'Number of times PII was detected and masked',
    ['pii_type'] # pii_type: PASSPORT, CARD, PHONE
)
```

### 3.4.2. Правило алертинга Prometheus (`alerting_rules.yml`)
*Автоматическое оповещение SRE/DevOps в Telegram при критических сбоях.*

```yaml
groups:
  - name: VoiceGraphCriticalAlerts
    rules:
      # Алерт 1: Превышение задержки голосового конвейера
      - alert: HighVoiceLatency
        expr: histogram_quantile(0.95, rate(voicegraph_pipeline_latency_ms_bucket[5m])) > 800
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Высокая задержка голосового конвейера (>800ms)"
          description: "P95 latency превышает 800ms в течение 2 минут. Проверьте нагрузку на vLLM или сеть."

      # Алерт 2: Аномально высокий процент отказов (возможная проблема со скриптом или телефонией)
      - alert: HighRefusalRate
        expr: rate(voicegraph_call_outcomes_total{outcome="REFUSAL"}[10m]) / rate(voicegraph_call_outcomes_total[10m]) > 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Аномально высокий процент отказов (>70%)"
          description: "Проверьте логи Reflection Agent и качество SIP-транка."

      # Алерт 3: Обнаружение PII в логах (Критическое нарушение 152-ФЗ)
      # Примечание: Реализуется через Loki/Promtail regex alerting, если Presidio дал сбой
      - alert: PotentialPIILeakInLogs
        expr: sum by (namespace) (count_over_time({namespace="voicegraph-prod"} |~ "(?i)\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"[1m])) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Обнаружен потенциальный номер карты в логах!"
          description: "Немедленно проверьте работу middleware Presidio."
```