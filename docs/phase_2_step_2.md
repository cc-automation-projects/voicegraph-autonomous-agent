Это финальный шаг Этапа 2, который превращает обученную ML-модель в высокопроизводительный микросервис, готовый к интеграции с LangGraph-оркестратором. Ключевой вызов здесь — обеспечить latency **< 10 мс** на батч из 1000 пользователей, что требует не только оптимизированной модели, но и грамотного управления данными в памяти.

---

# 🚀 ЭТАП 2.2: Калибровка и FastAPI-сервис инференса

## Шаг 1: Архитектура инференса для достижения < 10 мс

Чтобы гарантировать задержку < 10 мс, мы **не можем** делать 1000 запросов к PostgreSQL при каждом вызове API. Вместо этого мы используем паттерн **In-Memory Feature Cache**:
1. Фоновый процесс (или отдельный ETL-джоб) раз в час выгружает актуальные признаки для всех активных пользователей из PostgreSQL в Redis или напрямую в память сервиса.
2. При запросе `/predict` сервис выполняет мгновенный $O(1)$ lookup в словаре/DataFrame, собирает матрицу признаков и делает векторизованный `predict_proba` через CatBoost.

---

## Шаг 2: Реализация FastAPI-сервиса

Создаем отдельный микросервис `propensity-service`.

**Файл: `src/propensity_service/main.py`**

```python
import os
import logging
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.catboost
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from src.schemas import PredictRequest, PredictResponse, ScoredUser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Глобальные переменные для кэширования в памяти
model_answer = None
model_conversion = None
feature_cache: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при запуске: загрузка модели из MLflow и кэширование фич."""
    global model_answer, model_conversion, feature_cache
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    model_name = os.getenv("MLFLOW_MODEL_NAME", "voicegraph_propensity_models")
    model_version = os.getenv("MLFLOW_MODEL_VERSION", "Production")
    
    logger.info(f"📥 Загрузка моделей из MLflow: {model_name}, версия: {model_version}...")
    
    # Загрузка откалиброванных моделей (Platt Scaling применен на этапе обучения)
    model_uri_answer = f"models:/{model_name}/p_answer@{model_version}"
    model_uri_conversion = f"models:/{model_name}/p_conversion@{model_version}"
    
    model_answer = mlflow.catboost.load_model(model_uri_answer)
    model_conversion = mlflow.catboost.load_model(model_uri_conversion)
    logger.info("✅ Модели успешно загружены в память.")
    
    # Имитация загрузки кэша признаков (в проде это делается через Redis или периодический ETL)
    logger.info("🗄️ Инициализация In-Memory Feature Cache...")
    # feature_cache = load_features_from_redis_or_db()
    # Для примера: пустой словарь, который заполняется фоновой задачей
    
    yield
    
    logger.info("🛑 Остановка сервиса инференса...")

app = FastAPI(
    title="VoiceGraph Propensity ML API",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/api/v1/ml/score/batch", response_model=PredictResponse)
async def score_batch(request: PredictRequest):
    """
    Рассчитывает priority_score для батча user_id.
    Гарантированная latency: < 10 мс для батча до 1000 записей.
    """
    start_time = time.perf_counter()
    
    user_ids = request.user_ids
    if len(user_ids) > 1000:
        raise HTTPException(status_code=400, detail="Максимальный размер батча: 1000 user_id")
    
    # 1. Извлечение признаков из In-Memory Cache (O(1) на пользователя)
    # В реальном сценарии здесь должен быть fallback в БД, если user_id нет в кэше
    features_list = []
    valid_user_ids = []
    
    for uid in user_ids:
        if uid in feature_cache:
            features_list.append(feature_cache[uid])
            valid_user_ids.append(uid)
            
    if not features_list:
        logger.warning("Ни один user_id не найден в кэше признаков.")
        return PredictResponse(scored_users=[])

    # 2. Векторизованное предсказание (CatBoost оптимизирован для батчей)
    df_features = pd.DataFrame(features_list)
    
    # predict_proba возвращает [[prob_0, prob_1]], нам нужен prob_1 (индекс 1)
    probs_answer = model_answer.predict_proba(df_features)[:, 1]
    probs_conversion = model_conversion.predict_proba(df_features)[:, 1]
    
    # 3. Расчет priority_score и формирование ответа
    scored_users = []
    for i, uid in enumerate(valid_user_ids):
        p_ans = float(probs_answer[i])
        p_conv = float(probs_conversion[i])
        
        scored_users.append(ScoredUser(
            user_id=uid,
            p_answer=round(p_ans, 4),
            p_conversion=round(p_conv, 4),
            priority_score=round(p_ans * p_conv, 4),
            recommended_call_window="18:00-20:00" # Заглушка, в проде берется из кэша
        ))
    
    # 4. Сортировка по priority_score по убыванию
    scored_users.sort(key=lambda x: x.priority_score, reverse=True)
    
    # 5. Метрики производительности
    latency_ms = (time.perf_counter() - start_time) * 1000
    if latency_ms > 10.0:
        logger.warning(f"⚠️ Превышение SLA latency: {latency_ms:.2f} мс для батча {len(user_ids)}")
    
    return PredictResponse(scored_users=scored_users)

# Эндпоинт для обновления кэша (вызывается фоновым джобом)
@app.post("/api/v1/ml/cache/update")
async def update_cache(features: Dict[str, Dict[str, Any]]):
    global feature_cache
    feature_cache.update(features)
    return {"status": "success", "cache_size": len(feature_cache)}
```

---

## Шаг 3: Docker-упаковка для Production

Используем многоэтапную сборку и оптимизированные библиотеки (`uvloop`, `httptools`) для максимальной скорости Uvicorn.

**Файл: `src/propensity_service/Dockerfile`**

```dockerfile
# --- Stage 1: Build ---
FROM python:3.12-slim as builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Установка системных зависимостей для CatBoost и компиляции
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install uvloop httptools && \
    pip install --no-cache-dir -e .

# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Копирование только необходимого из builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование кода приложения
COPY src/propensity_service ./src/propensity_service

# Запуск от non-root пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Запуск Uvicorn с оптимизациями
CMD ["uvicorn", "src.propensity_service.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info"]
```

---

## Шаг 4: MLOps и RBAC (Жизненный цикл модели)

Чтобы разделить ответственность между Data Scientist (DS) и DevOps, мы используем **MLflow Model Registry**.

### 4.1. Действия Data Scientist (Обучение и регистрация)
После успешного обучения (Подзадача 2.1), DS запускает скрипт регистрации:
```python
import mlflow

# Предположим, run_id получен после обучения
run_id = "abc123def456"

# Регистрация модели p_answer
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models/p_answer",
    name="voicegraph_propensity_models/p_answer"
)

# Регистрация модели p_conversion
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models/p_conversion",
    name="voicegraph_propensity_models/p_conversion"
)
```
*В MLflow модель получает статус `None` (или `Staging`). DS может протестировать её на канареечном трафике.*

### 4.2. Действия DevOps (Деплой в Production)
DevOps **не имеет прав** на изменение кода модели или её переобучение. Его задача — перевести проверенную модель в статус `Production` и обновить переменные окружения в Kubernetes.

1. В UI MLflow или через CLI DevOps меняет статус версии модели на `Production`.
2. В Helm-чарте или Deployment YAML обновляется переменная:
   ```yaml
   env:
     - name: MLFLOW_MODEL_VERSION
       value: "Production" # Или конкретный номер версии, например "3"
   ```
3. Выполняется `kubectl rollout restart deployment propensity-service`. Сервис автоматически подтянет новую версию из MLflow при следующем старте благодаря логике в `lifespan`.

---

## Шаг 5: Валидация Latency (< 10 мс)

Чтобы доказать выполнение SLA, напишем простой бенчмарк на `pytest-benchmark` или `locust`.

**Файл: `tests/test_inference_latency.py`**

```python
import pytest
import httpx
import numpy as np
import pandas as pd
from unittest.mock import patch

# Эмуляция кэша из 1000 пользователей
MOCK_CACHE = {
    f"user_{i}": {
        "hour_of_day": 14,
        "is_weekend": 0,
        "days_since_last_call": 5.0,
        "total_calls_30d": 2,
        "success_rate_90d": 0.5,
        "avg_duration_sec": 120.0,
        "barge_in_ratio_30d": 0.1,
        "last_sentiment": "CALM",
        "ltv_segment": 1,
        "region_code": 77,
        "script_variation_id": 0,
        "consent_age_days": 30
    }
    for i in range(1000)
}

@pytest.mark.asyncio
async def test_batch_inference_latency():
    # Имитируем запрос к запущенному сервису (или используем TestClient)
    from fastapi.testclient import TestClient
    from src.propensity_service.main import app
    
    # Мокаем глобальные переменные и кэш
    import src.propensity_service.main as main_module
    
    # Создаем фиктивные откалиброванные модели (заглушки для теста скорости)
    class MockModel:
        def predict_proba(self, X):
            # Имитация быстрого предсказания CatBoost
            return np.random.rand(len(X), 2)
            
    main_module.model_answer = MockModel()
    main_module.model_conversion = MockModel()
    main_module.feature_cache = MOCK_CACHE
    
    client = TestClient(app)
    
    user_ids = list(MOCK_CACHE.keys())
    payload = {"campaign_id": "test-campaign", "user_ids": user_ids}
    
    # Замер времени
    import time
    start = time.perf_counter()
    response = client.post("/api/v1/ml/score/batch", json=payload)
    latency_ms = (time.perf_counter() - start) * 1000
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["scored_users"]) == 1000
    
    # Проверка, что первый элемент имеет наивысший скор (сортировка работает)
    assert data["scored_users"][0]["priority_score"] >= data["scored_users"][-1]["priority_score"]
    
    # ГЛАВНОЕ ТРЕБОВАНИЕ: Latency < 10 мс
    assert latency_ms < 10.0, f"Latency {latency_ms:.2f} мс превысила лимит в 10 мс!"
    print(f"✅ Успешно: Latency составила {latency_ms:.2f} мс для 1000 пользователей.")
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 2.2)

Прежде чем переходить к **ЭТАПУ 3 (LangGraph Оркестрация)**, убедитесь, что:

- [ ] Сервис `propensity-service` успешно запускается через Docker и подключается к MLflow.
- [ ] Эндпоинт `/api/v1/ml/score/batch` принимает JSON с 1000 `user_id` и возвращает корректно отсортированный список `ScoredUser`.
- [ ] **Тест производительности** подтверждает, что время обработки запроса (от получения HTTP-пакета до отправки ответа) составляет **< 10 мс** (p95) при условии, что признаки находятся в In-Memory Cache.
- [ ] Модели в MLflow имеют четкое разделение версий (`Staging` / `Production`), и сервис корректно загружает именно версию `Production`.
- [ ] В логах сервиса отсутствуют предупреждения о превышении SLA при штатной нагрузке.
