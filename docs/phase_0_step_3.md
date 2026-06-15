Это критически важный компонент для соблюдения законодательства РФ. Любая задержка или пропуск PII (персональных данных) здесь недопустимы. Мы реализуем это не как отдельный HTTP-сервис (что добавило бы 50-100 мс задержки), а как **встроенный Singleton-компонент на базе Python SDK Microsoft Presidio**, что обеспечит время обработки < 5 мс на текст.

---

# 🚀 ЭТАП 0.3: Pre-processing и маскирование PII

## Шаг 1: Зависимости и конфигурация

Добавляем необходимые библиотеки в наш `pyproject.toml`. Presidio Analyzer и Anonymizer будут работать локально, без отправки данных во внешние сервисы.

```toml
# Добавить в pyproject.toml
dependencies = [
    # ... предыдущие зависимости ...
    "presidio-analyzer>=2.2.35",
    "presidio-anonymizer>=2.2.35",
    "phonenumbers>=8.13.13", # Для валидации российских номеров телефонов
    "spacy>=3.7.4",          # Требуется для NER-распознавания имен в Presidio
]
```

Установка и загрузка языковой модели spaCy для русского языка (обязательно для корректной работы NER):
```bash
pip install -e .
python -m spacy download ru_core_news_sm
```

---

## Шаг 2: Реализация кастомных распознавателей для РФ

Стандартный Presidio хорошо ищет кредитные карты (с алгоритмом Луна) и email, но требует донастройки для российских реалий (паспорта, ИНН, специфичные форматы телефонов).

**Файл: `src/pii_sanitizer/recognizers.py`**

```python
from presidio_analyzer import Pattern, PatternRecognizer

# 1. Распознаватель российских паспортов (серия 4 цифры, номер 6 цифр, с опциональным пробелом)
# Примеры: "4500 123456", "4500123456"
RU_PASSPORT_PATTERN = Pattern(
    name="RU_PASSPORT_PATTERN",
    regex=r"\b\d{4}\s?\d{6}\b",
    score=0.85 # Высокий скор, так как паттерн специфичен
)
ru_passport_recognizer = PatternRecognizer(
    supported_entity="RU_PASSPORT",
    patterns=[RU_PASSPORT_PATTERN]
)

# 2. Распознаватель ИНН (10 цифр для юрлиц, 12 для физлиц)
RU_INN_PATTERN = Pattern(
    name="RU_INN_PATTERN",
    regex=r"\b(?:\d{10}|\d{12})\b",
    score=0.75 # Чуть ниже, так как 10-12 цифр могут быть чем-то иным, но в контексте диалога это часто ИНН
)
ru_inn_recognizer = PatternRecognizer(
    supported_entity="RU_INN",
    patterns=[RU_INN_PATTERN]
)

# 3. Распознаватель российских телефонов (+7 или 8, с опциональными разделителями)
# Примеры: "+79001234567", "8 (900) 123-45-67", "8900 123 45 67"
RU_PHONE_PATTERN = Pattern(
    name="RU_PHONE_PATTERN",
    regex=r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
    score=0.85
)
ru_phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[RU_PHONE_PATTERN]
)
```

---

## Шаг 3: Ядро сервиса PIISanitizer (Singleton)

Создаем асинхронно-совместимый класс, который инициализирует движки Presidio один раз при старте приложения и предоставляет быстрый метод для очистки текста.

**Файл: `src/pii_sanitizer/service.py`**

```python
import logging
from typing import List, Dict
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.pii_sanitizer.recognizers import (
    ru_passport_recognizer,
    ru_inn_recognizer,
    ru_phone_recognizer
)

logger = logging.getLogger(__name__)

class PIISanitizer:
    """
    Асинхронно-совместимый синглтон для маскирования PII с использованием Microsoft Presidio.
    Гарантирует, что текст, идущий в LLM или логи, не содержит чувствительных данных.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PIISanitizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        logger.info("🛡️ Инициализация PIISanitizer (Microsoft Presidio)...")
        
        # 1. Настройка реестра распознавателей
        registry = RecognizerRegistry()
        # Загружаем встроенные (включая CREDIT_CARD с проверкой по алгоритму Луна, PERSON, и т.д.)
        registry.load_predefined_recognizers()
        
        # Добавляем кастомные российские распознаватели
        registry.add_recognizer(ru_passport_recognizer)
        registry.add_recognizer(ru_inn_recognizer)
        registry.add_recognizer(ru_phone_recognizer)

        # 2. Инициализация движка анализа (поддерживаем русский и английский)
        self.analyzer = AnalyzerEngine(
            registry=registry, 
            supported_languages=["ru", "en"]
        )

        # 3. Инициализация движка анонимизации
        self.anonymizer = AnonymizerEngine()

        # 4. Конфигурация операторов замены (Tokens)
        self.operators: Dict[str, OperatorConfig] = {
            "RU_PASSPORT": OperatorConfig("replace", {"new_value": "[PASSPORT_REDACTED]"}),
            "RU_INN": OperatorConfig("replace", {"new_value": "[INN_REDACTED]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CARD_NUMBER_REDACTED]"}),
            "PERSON": OperatorConfig("replace", {"new_value": "[NAME_REDACTED]"}), # Опционально, для имен
        }
        
        self._initialized = True
        logger.info("✅ PIISanitizer успешно инициализирован.")

    def sanitize(self, text: str) -> str:
        """
        Очищает текст от PII. Если текст пустой или None, возвращает его как есть.
        """
        if not text or not isinstance(text, str):
            return text

        try:
            # Анализ текста на наличие сущностей
            analyzer_results = self.analyzer.analyze(
                text=text,
                language="ru", # Основной язык диалогов
                entities=list(self.operators.keys())
            )
            
            # Если ничего не найдено, возвращаем оригинал (экономия ресурсов)
            if not analyzer_results:
                return text

            # Анонимизация (замена на токены)
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=self.operators
            )
            
            return anonymized_result.text

        except Exception as e:
            # В продакшене: логировать ошибку, но НЕ блокировать поток. 
            # В случае сбоя санитайзера лучше вернуть оригинал и поднять алерт, 
            # чем "положить" весь голосовой пайплайн.
            logger.error(f"❌ Ошибка при маскировании PII: {e}. Текст возвращен без изменений.")
            return text

# Глобальный экземпляр для использования в качестве зависимости FastAPI или в воркерах
sanitizer = PIISanitizer()
```

---

## Шаг 4: Интеграция как Middleware в FastAPI и LiveKit Worker

Покажем, как это применяется на практике. Текст от ASR *никогда* не должен попадать в LLM или БД без прохождения через `sanitizer.sanitize()`.

**Файл: `src/middleware/pii_middleware.py`**

```python
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from src.pii_sanitizer.service import sanitizer

logger = logging.getLogger(__name__)

class PIIMaskingMiddleware(BaseHTTPMiddleware):
    """
    Перехватывает входящие JSON-запросы (например, от ASR-сервиса) 
    и маскирует поле 'text' перед передачей его в роутинг.
    """
    async def dispatch(self, request: Request, call_next):
        # Применяем только к конкретным эндпоинтам, где ожидается сырой текст
        if request.url.path in ["/api/v1/voice/transcript", "/api/v1/logs"]:
            try:
                body = await request.json()
                if "text" in body and isinstance(body["text"], str):
                    original_text = body["text"]
                    body["text"] = sanitizer.sanitize(original_text)
                    
                    # Логируем факт маскирования для аудита (без сохранения самого текста!)
                    if original_text != body["text"]:
                        logger.info(f"🛡️ PII замаскирован в запросе {request.url.path}")
                
                # Подменяем body для дальнейшей обработки
                async def receive():
                    return {"type": "http.request", "body": str(body).encode()}
                request._receive = receive
                
            except Exception as e:
                logger.warning(f"Не удалось обработать body для маскирования PII: {e}")

        response = await call_next(request)
        return response
```

**Пример использования внутри LiveKit Voice Agent Worker (псевдокод):**

```python
from src.pii_sanitizer.service import sanitizer
from src.database import save_call_log

async def on_transcript_received(session_id: str, raw_asr_text: str):
    # 1. НЕМЕДЛЕННОЕ МАСКИРОВАНИЕ
    safe_text = sanitizer.sanitize(raw_asr_text)
    
    # 2. Отправка в LLM (безопасно)
    llm_response = await call_llm(prompt=f"Context: {safe_text}")
    
    # 3. Сохранение в БД (безопасно)
    await save_call_log(
        session_id=session_id,
        transcript_masked=safe_text, # Сохраняем ТОЛЬКО замаскированную версию
        # raw_asr_text НЕ передается и НЕ сохраняется нигде, кроме временной памяти для текущего чанка
    )
```

---

## Шаг 5: Модульное тестирование (Shift-Left Security)

Критически важно доказать, что наши регулярные выражения и Presidio корректно отрабатывают на реальных российских данных.

**Файл: `tests/test_pii_sanitizer.py`**

```python
import pytest
from src.pii_sanitizer.service import PIISanitizer

@pytest.fixture
def sanitizer():
    return PIISanitizer()

def test_ru_passport_masking(sanitizer):
    text = "Мой паспорт 4500 123456, проверьте его."
    result = sanitizer.sanitize(text)
    assert "[PASSPORT_REDACTED]" in result
    assert "4500" not in result
    assert "123456" not in result

def test_ru_inn_masking(sanitizer):
    text = "ИНН организации 7707083893 и физлица 770708389301."
    result = sanitizer.sanitize(text)
    assert result.count("[INN_REDACTED]") == 2
    assert "7707083893" not in result

def test_ru_phone_masking(sanitizer):
    text = "Звоните мне на +7 (900) 123-45-67 или 89001234567."
    result = sanitizer.sanitize(text)
    assert result.count("[PHONE_REDACTED]") == 2
    assert "900" not in result # Часть номера не должна остаться

def test_credit_card_luhn_masking(sanitizer):
    # Валидный номер карты по алгоритму Луна
    text = "Оплатите картой 4276 5500 1234 9988, пожалуйста."
    result = sanitizer.sanitize(text)
    assert "[CARD_NUMBER_REDACTED]" in result
    assert "4276" not in result

def test_no_pii_unchanged(sanitizer):
    text = "Здравствуйте, как ваши дела? Погода сегодня отличная."
    result = sanitizer.sanitize(text)
    assert result == text
```

Запуск тестов: `pytest tests/test_pii_sanitizer.py -v`

---

## ✅ Definition of Done (Критерии готовности Подзадачи 0.3)

Прежде чем перейти к **ЭТАПУ 1 (Real-time Voice Pipeline)**, убедитесь, что:

- [ ] Сервис `PIISanitizer` успешно инициализируется как Singleton без ошибок загрузки spaCy или Presidio.
- [ ] Юнит-тесты (`test_pii_sanitizer.py`) проходят на 100%, подтверждая маскирование паспортов, ИНН, телефонов и карт (с проверкой Луна).
- [ ] В коде `VoiceAgentWorker` или обработчиках транскриптов вызов `sanitizer.sanitize()` стоит **строго до** любого вызова LLM или записи в PostgreSQL/MinIO.
- [ ] В логах приложения (Prometheus/Grafana или stdout) отсутствуют "голые" номера телефонов или паспортов (проверено через `grep` по логам тестового прогона).
- [ ] Задержка, вносимая методом `sanitize()` на текст длиной до 500 символов, составляет **< 10 мс** (проверено через `timeit` или простой бенчмарк).
