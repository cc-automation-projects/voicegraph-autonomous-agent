Это фундамент гиперперсонализации. Чтобы агент мог говорить: *"Иван Иванович, в прошлый раз вы упоминали, что у вас была задержка доставки, сегодня все пришло вовремя?"*, ему нужна долговременная эпизодическая память. Критически важно, чтобы эта память извлекалась и обрабатывалась **строго внутри контура РФ** (on-premise), чтобы не нарушать 152-ФЗ при анализе диалогов.

---

# 🚀 ЭТАП 4.1: Настройка Mem0 и Qdrant

## Шаг 1: Зависимости и подготовка окружения

Добавляем необходимые библиотеки в `pyproject.toml` микросервиса памяти (или общего `orchestrator`/`voice-worker`, в зависимости от вашей модульной структуры).

```toml
# Добавить в pyproject.toml
dependencies = [
    "mem0ai>=0.1.20",       # Фреймворк для управления памятью агентов
    "qdrant-client>=1.9.0", # Официальный клиент Qdrant для Python
    "httpx>=0.27.0"         # Для асинхронных запросов к локальному vLLM
]
```

---

## Шаг 2: Инфраструктура Qdrant (On-Premise Deployment)

Развертываем Qdrant в Kubernetes как StatefulSet с постоянным хранилищем и включенным индексированием payload для молниеносной фильтрации по `user_id`.

**Файл: `infra/k8s/qdrant-statefulset.yaml`**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant-memory
  namespace: voicegraph-prod
spec:
  serviceName: "qdrant"
  replicas: 1 # Для production можно увеличить до 3 с включенным distributed mode
  selector:
    matchLabels:
      app: qdrant-memory
  template:
    metadata:
      labels:
        app: qdrant-memory
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.9.0
        ports:
        - containerPort: 6333 # REST API
        - containerPort: 6334 # gRPC API (рекомендуется для высокой производительности)
        env:
        - name: QDRANT__SERVICE__GRPC_PORT
          value: "6334"
        - name: QDRANT__STORAGE__HNSW_INDEX__MAX_INDEXING_THREADS
          value: "2"
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 50Gi # Объем зависит от ожидаемого количества фактов
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: voicegraph-prod
spec:
  selector:
    app: qdrant-memory
  ports:
    - name: rest
      port: 6333
      targetPort: 6333
    - name: grpc
      port: 6334
      targetPort: 6334
```

---

## Шаг 3: Инициализация коллекции и схемы Payload в Qdrant

Прежде чем Mem0 начнет писать данные, мы должны явно создать коллекцию с правильной схемой, чтобы обеспечить быструю фильтрацию по `user_id` и сортировку по `timestamp`/`decay_factor`.

**Файл: `src/memory/qdrant_setup.py`**

```python
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

logger = logging.getLogger(__name__)

COLLECTION_NAME = "voicegraph_episodic_memory"
VECTOR_SIZE = 1024 # Должен совпадать с размером эмбеддингов модели (например, intfloat/multilingual-e5-large или аналогичной, используемой Mem0)

def initialize_qdrant_collection():
    """Создает коллекцию Qdrant с оптимизированной схемой payload для Mem0."""
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-service:6333")
    
    # Используем gRPC для максимальной производительности
    client = QdrantClient(url=qdrant_url, prefer_grpc=True)

    if client.collection_exists(COLLECTION_NAME):
        logger.info(f"✅ Коллекция '{COLLECTION_NAME}' уже существует.")
        return client

    logger.info(f"🛠️ Создание коллекции '{COLLECTION_NAME}'...")
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100
            )
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2,
            memmap_threshold=10000
        ),
        # Явное определение схемы payload для ускорения фильтрации
        on_disk_payload=True,
    )

    # Создаем индексы для payload полей, по которым будем фильтровать
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="timestamp",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="decay_factor",
        field_schema=models.PayloadSchemaType.FLOAT,
    )

    logger.info("✅ Коллекция и индексы Qdrant успешно созданы.")
    return client

if __name__ == "__main__":
    initialize_qdrant_collection()
```

---

## Шаг 4: Конфигурация и обертка Mem0 (Memory Manager)

Настраиваем Mem0 на использование нашего локального Qdrant и локального vLLM (Qwen2.5-7B) для извлечения фактов. Это гарантирует, что сырые транскрипты **никогда не покидают периметр**.

**Файл: `src/memory/manager.py`**

```python
import os
import time
import logging
from mem0 import Memory
from typing import List, Dict, Any
from src.schemas import MemoryFact # Из data_api_contracts.md

logger = logging.getLogger(__name__)

class VoiceGraphMemoryManager:
    """
    Управляющий класс для эпизодической памяти на базе Mem0.
    Обеспечивает строгий контроль над тем, какие данные сохраняются и извлекаются.
    """
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-service:6333")
        vllm_url = os.getenv("VLLM_API_URL", "http://vllm-service:8000/v1")
        
        # Конфигурация Mem0 для on-premise стека
        self.config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": qdrant_url.replace("http://", "").replace("https://", "").split(":")[0],
                    "port": int(qdrant_url.split(":")[-1]),
                    "collection_name": "voicegraph_episodic_memory",
                    "path": None, # Не использовать локальный файл, работать с сервером
                }
            },
            "llm": {
                # Используем провайдер 'vllm' или 'openai', указывая на наш локальный эндпоинт
                "provider": "openai", 
                "config": {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "base_url": vllm_url,
                    "api_key": "dummy-key", # vLLM часто не требует ключа, но параметр обязателен
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "intfloat/multilingual-e5-large", # Отличная модель для русского языка
                }
            },
            "version": "v1.1"
        }
        
        logger.info("🧠 Инициализация VoiceGraph Memory Manager (Mem0 + Local Qdrant + Local vLLM)...")
        self.memory = Memory.from_config(self.config)

    def add_fact(self, user_id: str, fact: str, category: str, confidence: float) -> None:
        """
        Добавляет новый факт в память пользователя.
        """
        try:
            # Вычисляем decay_factor (начинается с 1.0, будет уменьшаться со временем при чтении)
            current_timestamp = int(time.time())
            
            # Mem0 добавляет метаданные автоматически, но мы можем передать свои через metadata
            metadata = {
                "user_id": user_id,
                "category": category,
                "timestamp": current_timestamp,
                "decay_factor": 1.0,
                "confidence": confidence
            }
            
            # Вызов Mem0 для извлечения сущностей и сохранения вектора
            # text - это сырой фрагмент диалога или сформулированный факт
            self.memory.add(
                messages=[{"role": "user", "content": fact}],
                user_id=user_id,
                metadata=metadata
            )
            logger.info(f"✅ Факт добавлен в память для user_id={user_id}: {fact[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении факта в память: {e}")

    def get_facts(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Извлекает топ-N наиболее релевантных фактов для пользователя с учетом Memory Decay.
        """
        try:
            # Запрос к Mem0 с фильтром по user_id
            results = self.memory.search(
                query="предпочтения, жалобы, важные детали из прошлых разговоров",
                user_id=user_id,
                limit=limit * 2 # Берем с запасом для применения decay-сортировки
            )
            
            processed_facts = []
            current_time = int(time.time())
            
            for res in results:
                metadata = res.get("metadata", {})
                
                # Реализация Memory Decay: штраф за старость факта
                # lambda = 0.0001 (примерно 50% веса теряется за 100 дней)
                age_seconds = current_time - metadata.get("timestamp", current_time)
                decay_lambda = 0.0001
                decay_factor = max(0.1, 1.0 - (decay_lambda * age_seconds)) # Минимальный вес 0.1
                
                # Итоговый скор = релевантность вектора * decay_factor * уверенность
                base_score = res.get("score", 0.5)
                confidence = metadata.get("confidence", 1.0)
                final_score = base_score * decay_factor * confidence
                
                processed_facts.append({
                    "fact": res.get("memory", ""), # Mem0 возвращает сформулированный факт в поле 'memory'
                    "category": metadata.get("category", "UNKNOWN"),
                    "final_score": round(final_score, 4),
                    "decay_factor": round(decay_factor, 4)
                })
            
            # Сортируем по итоговому скорубыванию и берем топ limit
            processed_facts.sort(key=lambda x: x["final_score"], reverse=True)
            return processed_facts[:limit]
            
        except Exception as e:
            logger.error(f"❌ Ошибка при извлечении фактов из памяти: {e}")
            return []

# Глобальный синглтон для использования в VoiceWorker и Reflection Agent
memory_manager = VoiceGraphMemoryManager()
```

---

## Шаг 5: Модульное тестирование (Shift-Left Testing)

Проверяем, что память корректно сохраняет данные и применяет механизм `decay_factor`.

**Файл: `tests/memory/test_memory_manager.py`**

```python
import pytest
import time
from unittest.mock import patch, MagicMock
from src.memory.manager import VoiceGraphMemoryManager

@pytest.fixture
def mock_mem0():
    with patch("src.memory.manager.Memory.from_config") as mock_from_config:
        mock_memory_instance = MagicMock()
        mock_from_config.return_value = mock_memory_instance
        yield mock_memory_instance

@pytest.fixture
def manager(mock_mem0):
    return VoiceGraphMemoryManager()

def test_add_fact_structure(manager, mock_mem0):
    user_id = "test-user-123"
    fact = "Клиент просит звонить только после 18:00"
    
    manager.add_fact(user_id, fact, category="PREFERENCE", confidence=0.9)
    
    # Проверяем, что Mem0.add был вызван с правильными метаданными
    mock_mem0.add.assert_called_once()
    call_kwargs = mock_mem0.add.call_args[1]
    assert call_kwargs["user_id"] == user_id
    assert call_kwargs["metadata"]["category"] == "PREFERENCE"
    assert call_kwargs["metadata"]["decay_factor"] == 1.0
    assert "timestamp" in call_kwargs["metadata"]

def test_get_facts_with_decay(manager, mock_mem0):
    user_id = "test-user-123"
    
    # Мокаем ответ от Mem0.search
    old_timestamp = int(time.time()) - (86400 * 100) # 100 дней назад
    mock_mem0.search.return_value = [
        {
            "memory": "Клиент не любит, когда ему звонят утром",
            "score": 0.9,
            "metadata": {
                "user_id": user_id,
                "category": "PREFERENCE",
                "timestamp": old_timestamp,
                "confidence": 0.95
            }
        }
    ]
    
    facts = manager.get_facts(user_id, limit=1)
    
    assert len(facts) == 1
    assert "Клиент не любит" in facts[0]["fact"]
    # Проверяем, что decay_factor применился (должен быть < 1.0 из-за возраста 100 дней)
    assert facts[0]["decay_factor"] < 1.0
    assert facts[0]["decay_factor"] > 0.1 # Не должен упасть ниже минимального порога
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 4.1)

Прежде чем переходить к **Подзадаче 4.2 (Auto-Recall и инъекция в контекст)**, убедитесь, что:

- [ ] Qdrant развернут в Kubernetes (или Docker) и доступен по gRPC/REST, коллекция `voicegraph_episodic_memory` создана с индексами по `user_id`, `timestamp` и `decay_factor`.
- [ ] Конфигурация Mem0 успешно подключается к **локальному** vLLM (`http://vllm-service:8000/v1`) и локальному Qdrant. Никакие внешние API (OpenAI, Anthropic) не используются для обработки памяти.
- [ ] Метод `add_fact` корректно сохраняет факт с начальным `decay_factor = 1.0` и текущим `timestamp`.
- [ ] Метод `get_facts` успешно извлекает факты, применяет формулу экспоненциального затухания (`decay_factor`) и возвращает отсортированный список.
- [ ] Юнит-тесты проходят, подтверждая корректность расчета `decay_factor` для старых и новых записей.
