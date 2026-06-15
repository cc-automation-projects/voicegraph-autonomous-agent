from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VoiceGraph"
    debug: bool = False
    log_level: str = "INFO"

    raw_data_path: Path = Path("data/raw/crm_history_12m.csv")
    processed_data_path: Path = Path("data/processed/crm_history_12m_clean.parquet")
    users_data_path: Path = Path("data/processed/users_latest.parquet")
    logs_data_path: Path = Path("data/processed/call_logs_latest.parquet")

    database_url: str = "postgresql://voicegraph_dev:dev_password@postgres:5432/voicegraph"
    redis_url: str = "redis://redis-checkpointer:6379/0"
    qdrant_url: str = "http://qdrant-service:6333"
    llm_api_url: str = "http://vllm-service:8000/v1"
    vllm_api_url: str = "http://vllm-service:8000/v1"
    mlflow_tracking_uri: str = "http://mlflow-server:5000"
    mlflow_model_name: str = "voicegraph_propensity_models"
    mlflow_model_version: str = "Production"

    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_ws_url: str = ""

    composio_api_key: str = ""
    telegram_bot_token: str = ""
    supervisor_chat_id: str = ""

    yc_folder_id: str = ""
    yc_iam_token: str = ""
    yandexgpt_api_key: str = ""

    gx_expectation_suite_name: str = "crm_raw_data_suite"
    dvc_remote_url: str = "s3://voicegraph-data"
    mock_latency_ms: int = 300

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
