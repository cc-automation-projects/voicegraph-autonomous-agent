from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class PropensitySettings(BaseSettings):
    model_path: str = "models/propensity_model.pkl"
    feature_store_path: str = "data/feature_store/features.parquet"
    golden_dataset_path: str = "fixtures/golden_dataset.csv"
    catboost_iterations: int = 2000
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6
    catboost_l2_leaf_reg: float = 3.0
    catboost_random_seed: int = 42
    catboost_early_stopping_rounds: int = 200
    prediction_threshold: float = 0.3
    inference_batch_size: int = 256

    model_config = SettingsConfigDict(env_prefix="PROPENSITY_", extra="ignore")
