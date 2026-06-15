Ниже представлена исчерпывающая, production-ready реализация развертывания и оптимизации vLLM (с альтернативой для YandexGPT), полностью согласованная с архитектурой и требованиями проекта.

---

# 🚀 ЭТАП 1.1: Развертывание и оптимизация vLLM

## Шаг 1: Подготовка инфраструктуры Kubernetes (GPU Node)

Для запуска Qwen2.5-72B с `tensor_parallel_size=4` требуется узел с 4× GPU (NVIDIA H100 80GB или A100 80GB). Убедитесь, что в кластере установлен `nvidia-device-plugin`.

Создадим `NodeSelector` и `Tolerations`, чтобы гарантировать, что под с LLM будет развернут именно на GPU-ноде.

**Файл: `infra/k8s/vllm-node-affinity.yaml`** (Пример разметки ноды, если используется bare-metal или Yandex Managed Kubernetes):
```yaml
# Эта разметка должна быть применена к GPU-ноде в кластере
# kubectl label nodes <gpu-node-name> accelerator=nvidia-h100
# kubectl taint nodes <gpu-node-name> nvidia.com/gpu=present:NoSchedule
```

---

## Шаг 2: Production-манифест развертывания vLLM (On-Premise / Qwen2.5-72B)

Этот манифест содержит все критические флаги оптимизации, указанные в требованиях. Мы используем формат `bfloat16`, который нативно поддерживается H100/A100 и обеспечивает лучшую производительность и экономию памяти по сравнению с `float16`.

**Файл: `infra/k8s/vllm-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: voicegraph-prod
  labels:
    app: vllm-inference
spec:
  replicas: 1 # Масштабирование vLLM с TP=4 осуществляется вертикально (больше GPU на реплику), а не горизонтально
  selector:
    matchLabels:
      app: vllm-inference
  strategy:
    type: Recreate # Важно для GPU-подов, чтобы избежать конфликтов выделения памяти при обновлении
  template:
    metadata:
      labels:
        app: vllm-inference
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      nodeSelector:
        accelerator: nvidia-h100 # Соответствует метке GPU-ноды
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Equal"
          value: "present"
          effect: "NoSchedule"
      containers:
        - name: vllm-server
          image: vllm/vllm-openai:v0.6.1 # Фиксированная стабильная версия
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - "--model"
            - "Qwen/Qwen2.5-72B-Instruct"
            - "--tensor-parallel-size"
            - "4" # Распределение модели на 4 GPU
            - "--gpu-memory-utilization"
            - "0.90" # Резерв 10% для CUDA context и ОС
            - "--enable-prefix-caching"
            - "true" # Критично для Voice: кэширует системный промпт и историю диалога
            - "--max-num-seqs"
            - "128" # Баланс между throughput и latency. Для voice 128-256 оптимально, чтобы не создавать очередь
            - "--max-model-len"
            - "8192" # Ограничение контекста для экономии KV Cache
            - "--dtype"
            - "bfloat16" # Нативная поддержка на H100, быстрее и меньше места
            - "--disable-log-requests" # Снижает нагрузку на CPU при логировании каждого запроса
            - "--port"
            - "8000"
          ports:
            - containerPort: 8000
              name: http
          resources:
            limits:
              nvidia.com/gpu: 4
              memory: "256Gi" # Резерв системной RAM для загрузки модели
              cpu: "32"
            requests:
              nvidia.com/gpu: 4
              memory: "256Gi"
              cpu: "32"
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            - name: VLLM_WORKER_MULTIPROC_METHOD
              value: "spawn" # Стабильнее для multi-GPU
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120 # Время на скачивание и загрузку 72B модели
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 30
          volumeMounts:
            - name: hf-cache
              mountPath: /.cache/huggingface
      volumes:
        - name: hf-cache
          emptyDir: 
            medium: Memory # Кэширование весов модели в RAM для ускорения перезапуска пода
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: voicegraph-prod
spec:
  selector:
    app: vllm-inference
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
  type: ClusterIP
```

---

## Шаг 3: Альтернатива: API-шлюз для YandexGPT (Если on-premise H100 недоступен)

Если использование 4×H100 экономически или организационно невозможно, мы используем YandexGPT Pro (который показывает отличные результаты на русском языке и соответствует 152-ФЗ в контуре Yandex Cloud). 

Чтобы не переписывать код Voice Agent, мы создаем **FastAPI-шлюз**, который эмулирует OpenAI API, но под капотом вызывает YandexGPT с правильным IAM-аутентификацией и стримингом.

**Файл: `src/yandexgpt_gateway/main.py`**

```python
import os
import json
import httpx
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="VoiceGraph YandexGPT Gateway")

# Конфигурация Yandex Cloud
FOLDER_ID = os.getenv("YC_FOLDER_ID")
IAM_TOKEN = os.getenv("YC_IAM_TOKEN") # В продакшене должен обновляться через service account key
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt/latest"

async def get_iam_token():
    # В реальном продакшене здесь должен быть вызов к Yandex IAM для получения свежего токена
    # по Service Account Key. Для примера берем из env.
    return IAM_TOKEN

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    messages = body.get("messages", [])
    
    # Преобразование формата OpenAI в формат YandexGPT
    yandex_messages = []
    for msg in messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        yandex_messages.append({"role": role, "text": msg["content"]})

    headers = {
        "Authorization": f"Bearer {await get_iam_token()}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "modelUri": MODEL_URI,
        "completionOptions": {
            "stream": stream,
            "temperature": 0.7,
            "maxTokens": "500"
        },
        "messages": yandex_messages
    }

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    if stream:
        async def yandex_stream_generator():
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            # Преобразование ответа Yandex в формат OpenAI SSE
                            openai_chunk = {
                                "id": "chatcmpl-yandex",
                                "object": "chat.completion.chunk",
                                "created": 1710000000,
                                "model": "yandexgpt",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": data.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "")},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                            if data.get("result", {}).get("alternatives", [{}])[0].get("status") == "FINAL":
                                yield "data: [DONE]\n\n"
                                break
        return StreamingResponse(yandex_stream_generator(), media_type="text/event-stream")
    else:
        # Синхронный вызов (не рекомендуется для Voice, но нужен для fallback)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data["result"]["alternatives"][0]["message"]["text"]
            
            return {
                "id": "chatcmpl-yandex-sync",
                "object": "chat.completion",
                "created": 1710000000,
                "model": "yandexgpt",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
```

---

## Шаг 4: Бенчмаркинг и валидация Latency (< 300 мс TTFT)

Мы должны доказать, что конфигурация `enable_prefix_caching=True` и `tensor_parallel_size=4` дает требуемую производительность. Используем встроенный инструмент `vllm.benchmark_serving`.

### 4.1. Подготовка тестового датасета
Создадим файл `sharegpt_voice.json`, имитирующий реальные диалоги VoiceGraph (системный промпт + короткая реплика пользователя).

**Файл: `benchmarks/sharegpt_voice.json`**
```json
[
  {
    "conversations": [
      {"from": "system", "value": "Ты — автономный AI-менеджер кампаний VoiceGraph. Твоя цель: NPS_Optimization. Правила: 152-ФЗ, 38-ФЗ. Будь вежлив."},
      {"from": "human", "value": "Алло, да, слушаю вас."},
      {"from": "gpt", "value": "Здравствуйте! Это сервис контроля качества. Подскажите, как вы оцениваете нашу последнюю доставку по шкале от 1 до 10?"},
      {"from": "human", "value": "Ну, нормально, наверное, восьмерка."}
    ]
  },
  {
    "conversations": [
      {"from": "system", "value": "Ты — автономный AI-менеджер кампаний VoiceGraph. Твоя цель: NPS_Optimization. Правила: 152-ФЗ, 38-ФЗ. Будь вежлив."},
      {"from": "human", "value": "Да, я помню, в прошлый раз курьер опоздал."},
      {"from": "gpt", "value": "Понимаю ваше недовольство. Мы приняли меры. Скажите, сегодня доставка прошла вовремя?"},
      {"from": "human", "value": "Да, сегодня все хорошо."}
    ]
  }
]
```
*(Сгенерируйте 100-200 таких записей для статистической значимости).*

### 4.2. Запуск бенчмарка
Подключитесь к поду vLLM или запустите скрипт из кластера:

```bash
# Установка vllm для запуска бенчмарка
pip install vllm

# Запуск бенчмарка
python3 -m vllm.benchmark_serving \
    --model Qwen/Qwen2.5-72B-Instruct \
    --dataset-path benchmarks/sharegpt_voice.json \
    --dataset-name sharegpt \
    --request-rate 10 \
    --num-prompts 100 \
    --api-url http://vllm-service.voicegraph-prod.svc.cluster.local:8000/v1
```

### 4.3. Анализ результатов
В выводе бенчмарка нас интересуют три ключевые метрики:
1. **TTFT (Time To First Token) p95**: Должен быть **< 300 мс**. Благодаря `prefix_caching`, повторные запросы с одинаковым системным промптом будут иметь TTFT ~50-100 мс.
2. **TPOT (Time Per Output Token) p95**: Должен быть **< 50 мс** (обеспечивает скорость речи ~20 токенов в секунду, что соответствует естественной речи).
3. **Throughput**: Ожидаем > 40 запросов в секунду на данном кластере.

*Пример ожидаемого вывода:*
```text
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  15.2
Total input tokens:                      15000
Total generated tokens:                  5000
Request throughput (req/s):              6.58
Output token throughput (tok/s):         328.9
Time to First Token (TTFT) p95 (ms):     185.4  <-- ЦЕЛЬ ДОСТИГНУТА (< 300 мс)
Time Per Output Token (TPOT) p95 (ms):   35.2   <-- ЦЕЛЬ ДОСТИГНУТА (< 50 мс)
==================================================
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 1.1)

Прежде чем переходить к **Подзадаче 1.2 (Сборка LiveKit Voice Agent)**, убедитесь, что:

- [ ] Под `vllm-inference` находится в статусе `Running` и `Ready` в Kubernetes.
- [ ] В логах пода (`kubectl logs -l app=vllm-inference`) видно сообщение: `# GPU blocks: XXXX, # CPU blocks: YYYY` и `Prefix caching is enabled`.
- [ ] Бенчмарк `vllm.benchmark_serving` успешно выполнен, и метрика **TTFT p95 строго < 300 мс**.
- [ ] (Если используется YandexGPT) Шлюз успешно проксирует стриминг-запросы, и задержка сети до Yandex Cloud не превышает 50 мс (проверено через `ping`/`curl` изнутри кластера).
- [ ] Настроены алерты в Prometheus на метрику `vllm:gpu_cache_usage_perc` (если кэш заполнен на 95%+, нужно масштабировать или чистить).
