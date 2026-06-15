import logging
from typing import List

import great_expectations as gx
import pandas as pd
from pydantic import ValidationError

from src.voicegraph.config import settings
from src.voicegraph.schemas import CampaignDataSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_data_pipeline():
    logger.info(f"Запуск пайплайна. Источник: {settings.raw_data_path}")

    if not settings.raw_data_path.exists():
        logger.error("Файл сырых данных не найден.")
        return

    logger.info("Чтение сырых данных...")
    if settings.raw_data_path.suffix == ".parquet":
        df_raw = pd.read_parquet(settings.raw_data_path)
    else:
        df_raw = pd.read_csv(settings.raw_data_path, low_memory=False)

    initial_count = len(df_raw)
    logger.info(f"Загружено {initial_count} записей.")

    logger.info("Запуск Great Expectations...")
    context = gx.get_context()
    validator = context.get_validator(
        dataframe=df_raw,
        expectation_suite_name=settings.gx_expectation_suite_name,
        data_asset_name="crm_raw_data",
    )
    results = validator.validate()

    if not results.success:
        failed_checks = [
            res.expectation_config.kwargs.get("column", "unknown")
            for res in results.results
            if not res.success
        ]
        logger.error(f"Валидация GX не пройдена! Ошибки в колонках: {set(failed_checks)}")
        raise ValueError("Data Quality Check Failed.")

    logger.info("Great Expectations: Все проверки пройдены.")

    logger.info("Применение фильтра 38-ФЗ (consent_to_call == True)...")
    df_consent = df_raw[df_raw["consent_to_call"]].copy()
    filtered_count = initial_count - len(df_consent)
    logger.info(f"Отфильтровано {filtered_count} записей ({(filtered_count / initial_count) * 100:.2f}%) без согласия.")

    logger.info("Нормализация и строгая типизация через Pydantic...")
    valid_records: List[dict] = []
    rejected_count = 0

    for _, row in df_consent.iterrows():
        try:
            row_dict = row.to_dict()
            if isinstance(row_dict.get("last_contact_date"), str):
                row_dict["last_contact_date"] = pd.to_datetime(row_dict["last_contact_date"])

            validated_record = CampaignDataSchema(**row_dict).model_dump(mode="json")
            valid_records.append(validated_record)
        except ValidationError as e:
            rejected_count += 1
            if rejected_count <= 5:
                logger.warning(f"Отброшена запись user_id={row.get('user_id')}: {e.errors()[0]['msg']}")

    if rejected_count > 0:
        logger.warning(f"Всего отброшено невалидных записей на этапе Pydantic: {rejected_count}")

    df_final = pd.DataFrame(valid_records)

    logger.info(f"Сохранение {len(df_final)} чистых записей в {settings.processed_data_path}")
    settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(settings.processed_data_path, index=False, engine="pyarrow")

    logger.info("Пайплайн успешно завершен!")


if __name__ == "__main__":
    run_data_pipeline()
