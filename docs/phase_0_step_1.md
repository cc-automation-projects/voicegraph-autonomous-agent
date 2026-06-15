# 🚀 ЭТАП 0.1: Аудит и нормализация данных CRM

## Шаг 1: Профессиональная структура проекта

Создадим структуру, которая разделяет конфигурацию, бизнес-логику, тесты и артефакты данных.

```bash
mkdir -p voicegraph/{src,data/{raw,processed},tests,great_expectations/{expectations,validations,uncommitted}}
cd voicegraph

# Инициализация Git и DVC
git init
dvc init

# Создание виртуального окружения Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

Создаем файл зависимостей `pyproject.toml` (современный стандарт вместо `requirements.txt`):

```toml
[project]
name = "voicegraph-data-pipeline"
version = "0.1.0"
description = "ETL pipeline for VoiceGraph CRM data validation and normalization"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2.2",
    "pydantic>=2.7.1",
    "pydantic-settings>=2.2.1",
    "great-expectations>=1.1.0",
    "dvc[s3]>=3.50.0",
    "pyarrow>=16.1.0", # Для быстрого чтения/записи Parquet
    "pytest>=8.2.0",
    "ruff>=0.4.0",     # Линтер и форматтер
    "mypy>=1.10.0"     # Статическая типизация
]
```

Установим зависимости: `pip install -e .`

---

## Шаг 2: Управление конфигурацией (Best Practice)

Никогда не хардкодим пути или настройки. Используем `pydantic-settings`.

**Файл: `src/config.py`**
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # Пути к данным
    raw_data_path: Path = Path("data/raw/crm_history_12m.csv")
    processed_data_path: Path = Path("data/processed/crm_history_12m_clean.parquet")
    
    # Настройки Great Expectations
    gx_expectation_suite_name: str = "crm_raw_data_suite"
    
    # Настройки DVC / S3 (для будущего push)
    dvc_remote_url: str = "s3://voicegraph-data"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
```

Создайте файл `.env.example` в корне проекта:
```env
RAW_DATA_PATH=data/raw/crm_history_12m.csv
PROCESSED_DATA_PATH=data/processed/crm_history_12m_clean.parquet
```

---

## Шаг 3: Строгая типизация данных (Pydantic V2)

**Файл: `src/schemas.py`**
Здесь мы реализуем схему, строго соответствующую `data_api_contracts.md`.

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from typing import Optional
from datetime import datetime
import re
import hashlib

class LTVSegment(str):
    PREMIUM = "PREMIUM"
    STANDARD = "STANDARD"
    LOW = "LOW"

class CampaignDataSchema(BaseModel):
    """Строгая схема валидации записи CRM перед передачей в ML-пайплайн."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    user_id: str = Field(description="Уникальный идентификатор пользователя (UUID v4)")
    phone_hash: str = Field(description="SHA-256 хэш номера телефона (64 hex символа)")
    consent_to_call: bool = Field(description="Флаг согласия на обработку ПДн и звонки (38-ФЗ)")
    last_contact_date: datetime = Field(description="Дата последнего контакта")
    ltv_segment: LTVSegment = Field(description="Сегмент ценности клиента")

    @field_validator('user_id')
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)
        if not uuid_pattern.match(v):
            raise ValueError(f'Невалидный формат UUID: {v}')
        return v.lower()

    @field_validator('phone_hash')
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        if not re.match(r'^[a-f0-9]{64}$', v.lower()):
            raise ValueError(f'phone_hash должен быть валидной строкой SHA-256: {v}')
        return v.lower()

    @classmethod
    def hash_phone(cls, phone: str) -> str:
        """Утилита для хэширования сырого номера телефона, если он попал в raw-данные."""
        clean_phone = re.sub(r'\D', '', str(phone))
        return hashlib.sha256(clean_phone.encode('utf-8')).hexdigest()
```

---

## Шаг 4: Программная инициализация Great Expectations

Вместо ручного создания JSON-файлов, мы генерируем ожидания программно, чтобы они были версионируемы в Git.

**Файл: `src/gx_setup.py`**
```python
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration
import logging
from src.config import settings

logging.basicConfig(level=logging.INFO)

def initialize_gx_suite():
    context = gx.get_context()
    suite_name = settings.gx_expectation_suite_name
    
    # Создаем или получаем suite
    suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
    
    expectations = [
        ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "user_id"}),
        ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "consent_to_call"}),
        ExpectationConfiguration(expectation_type="expect_column_values_to_be_of_type", kwargs={"column": "consent_to_call", "type_": "bool"}),
        ExpectationConfiguration(expectation_type="expect_column_values_to_match_regex", kwargs={"column": "phone_hash", "regex": "^[a-f0-9]{64}$"}),
        ExpectationConfiguration(expectation_type="expect_column_values_to_be_in_set", kwargs={"column": "ltv_segment", "value_set": ["PREMIUM", "STANDARD", "LOW"]}),
    ]
    
    for exp in expectations:
        suite.add_expectation(exp)
        
    context.save_expectation_suite(suite)
    # Генерация Data Docs для визуальной проверки
    context.build_data_docs()
    logging.info(f"✅ Expectation Suite '{suite_name}' успешно создан и сохранен.")

if __name__ == "__main__":
    initialize_gx_suite()
```

---

## Шаг 5: Production-Ready ETL Пайплайн

Этот скрипт оптимизирован для работы с большими объемами данных (chunking), включает строгую фильтрацию по 38-ФЗ и логирование.

**Файл: `src/data_pipeline.py`**
```python
import pandas as pd
import great_expectations as gx
import logging
from pathlib import Path
from typing import List
from pydantic import ValidationError

from src.config import settings
from src.schemas import CampaignDataSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_data_pipeline():
    logging.info(f"🚀 Запуск пайплайна. Источник: {settings.raw_data_path}")
    
    if not settings.raw_data_path.exists():
        logging.error("❌ Файл сырых данных не найден. Создайте фиктивные данные для теста или укажите правильный путь.")
        return

    # 1. Чтение данных (используем chunksize для экономии RAM на больших файлах)
    # Для примера читаем целиком, но в продакшене лучше итерироваться по chunks
    logging.info("📥 Чтение сырых данных...")
    if settings.raw_data_path.suffix == '.parquet':
        df_raw = pd.read_parquet(settings.raw_data_path)
    else:
        df_raw = pd.read_csv(settings.raw_data_path, low_memory=False)
    
    initial_count = len(df_raw)
    logging.info(f"Загружено {initial_count} записей.")

    # 2. Валидация Great Expectations
    logging.info("🔍 Запуск Great Expectations...")
    context = gx.get_context()
    validator = context.get_validator(
        dataframe=df_raw,
        expectation_suite_name=settings.gx_expectation_suite_name,
        data_asset_name="crm_raw_data"
    )
    results = validator.validate()
    
    if not results["success"]:
        failed_checks = [res["expectation_config"]["kwargs"]["column"] for res in results["results"] if not res["success"]]
        logging.error(f"❌ Валидация GX не пройдена! Ошибки в колонках: {set(failed_checks)}")
        raise ValueError("Data Quality Check Failed. См. great_expectations/uncommitted/data_docs/")
    logging.info("✅ Great Expectations: Все проверки пройдены.")

    # 3. Строгая фильтрация по 38-ФЗ (Compliance Gate)
    logging.info("🛡️ Применение фильтра 38-ФЗ (consent_to_call == True)...")
    df_consent = df_raw[df_raw['consent_to_call'] == True].copy()
    filtered_count = initial_count - len(df_consent)
    logging.info(f"⚠️ Отфильтровано {filtered_count} записей ({(filtered_count/initial_count)*100:.2f}%) без согласия.")

    # 4. Построчная строгая валидация и нормализация через Pydantic
    logging.info("🧹 Нормализация и строгая типизация через Pydantic...")
    valid_records: List[dict] = []
    rejected_count = 0
    
    for _, row in df_consent.iterrows():
        try:
            # Приведение типов, если pandas прочитал их некорректно
            row_dict = row.to_dict()
            if isinstance(row_dict.get('last_contact_date'), str):
                row_dict['last_contact_date'] = pd.to_datetime(row_dict['last_contact_date'])
            
            # Валидация и сериализация
            validated_record = CampaignDataSchema(**row_dict).model_dump(mode='json') # mode='json' для совместимости с Parquet
            valid_records.append(validated_record)
        except ValidationError as e:
            rejected_count += 1
            # В продакшене эти ошибки пишутся в отдельный 'dead_letter_queue' лог
            if rejected_count <= 5: # Логируем только первые 5 для краткости
                logging.warning(f"Отброена запись user_id={row.get('user_id')}: {e.errors()[0]['msg']}")

    if rejected_count > 0:
        logging.warning(f"Всего отброено невалидных записей на этапе Pydantic: {rejected_count}")

    df_final = pd.DataFrame(valid_records)
    
    # 5. Сохранение в эффективном формате
    logging.info(f"💾 Сохранение {len(df_final)} чистых записей в {settings.processed_data_path}")
    settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(settings.processed_data_path, index=False, engine='pyarrow')
    
    logging.info("🎉 Пайплайн успешно завершен!")

if __name__ == "__main__":
    run_data_pipeline()
```

---

## Шаг 6: Модульное тестирование (Shift-Left Testing)

Качественный код обязан иметь тесты. Проверим, что пайплайн действительно отбрасывает пользователей без согласия.

**Файл: `tests/test_data_pipeline.py`**
```python
import pandas as pd
import pytest
from pathlib import Path
from src.schemas import CampaignDataSchema, LTVSegment
from datetime import datetime

def test_schema_validation_success():
    valid_data = {
        "user_id": "123e4567-e89b-12d3-a456-426614174000",
        "phone_hash": "a" * 64,
        "consent_to_call": True,
        "last_contact_date": datetime.now(),
        "ltv_segment": LTVSegment.PREMIUM
    }
    # Должно пройти без ошибок
    schema = CampaignDataSchema(**valid_data)
    assert schema.consent_to_call is True

def test_schema_validation_fails_on_bad_uuid():
    invalid_data = {
        "user_id": "not-a-uuid",
        "phone_hash": "a" * 64,
        "consent_to_call": True,
        "last_contact_date": datetime.now(),
        "ltv_segment": "PREMIUM"
    }
    with pytest.raises(ValueError, match="Невалидный формат UUID"):
        CampaignDataSchema(**invalid_data)

def test_consent_filtering_logic():
    # Имитация сырых данных
    df_raw = pd.DataFrame([
        {"user_id": "123e4567-e89b-12d3-a456-426614174000", "phone_hash": "a"*64, "consent_to_call": True, "last_contact_date": "2023-01-01", "ltv_segment": "LOW"},
        {"user_id": "123e4567-e89b-12d3-a456-426614174001", "phone_hash": "b"*64, "consent_to_call": False, "last_contact_date": "2023-01-02", "ltv_segment": "STANDARD"},
    ])
    
    # Логика фильтрации из пайплайна
    df_consent = df_raw[df_raw['consent_to_call'] == True].copy()
    
    assert len(df_consent) == 1
    assert df_consent.iloc[0]['user_id'] == "123e4567-e89b-12d3-a456-426614174000"
```

Запуск тестов: `pytest tests/ -v`

---

## Шаг 7: Версионирование данных через DVC

Теперь, когда код готов, мы фиксируем состояние данных.

```bash
# 1. Создаем тестовый сырой файл (если его нет)
mkdir -p data/raw
echo 'user_id,phone_hash,consent_to_call,last_contact_date,ltv_segment
123e4567-e89b-12d3-a456-426614174000,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,True,2023-10-01,PREMIUM
123e4567-e89b-12d3-a456-426614174001,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,False,2023-10-02,STANDARD' > data/raw/crm_history_12m.csv

# 2. Запускаем инициализацию GX и сам пайплайн
python -m src.gx_setup
python -m src.data_pipeline

# 3. Добавляем данные в DVC
dvc add data/raw/crm_history_12m.csv
dvc add data/processed/crm_history_12m_clean.parquet

# 4. Коммитим метаданные в Git
git add .
git commit -m "feat(data): implement Phase 0.1 ETL pipeline with GX validation and 38-FZ compliance filtering"
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 0.1)

Прежде чем мы перейдем к **Подзадаче 0.2 (Инфраструктура телефонии)**, убедитесь, что:
- [ ] Скрипт `src/data_pipeline.py` выполняется без ошибок (`python -m src.data_pipeline`).
- [ ] В выходном файле `data/processed/crm_history_12m_clean.parquet` **100%** строк имеют `consent_to_call == True`.
- [ ] Тесты `pytest` проходят успешно (100% pass rate).
- [ ] Файлы `.dvc` закоммичены в Git, а команда `dvc status` показывает, что изменения отсутствуют (все зафиксировано).
- [ ] Код отформатирован и проверен: `ruff check src/ tests/` и `mypy src/` не выдают ошибок.
