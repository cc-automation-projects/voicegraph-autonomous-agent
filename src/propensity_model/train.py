import logging
from pathlib import Path

import joblib
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

from src.propensity_model.config import PropensitySettings
from src.propensity_model.features import build_training_dataset

logger = logging.getLogger(__name__)

class PropensityTrainer:
    def __init__(self, config: PropensitySettings | None = None):
        self.config = config or PropensitySettings()

    def train(self, logs: list[dict]) -> CatBoostClassifier:
        logger.info("Построение тренировочного датасета через Featuretools DFS...")
        x = build_training_dataset(logs)
        y = pd.Series([log.get("is_converted", 0) for log in logs])

        logger.info(f"Тренировочный датасет: {x.shape[0]} строк, {x.shape[1]} признаков")

        smote = SMOTE(random_state=self.config.catboost_random_seed)
        x_resampled, y_resampled = smote.fit_resample(x, y)
        logger.info(f"После SMOTE: {x_resampled.shape[0]} строк")

        train_pool = Pool(
            x_resampled,
            y_resampled,
            cat_features=[],
        )

        model = CatBoostClassifier(
            iterations=self.config.catboost_iterations,
            learning_rate=self.config.catboost_learning_rate,
            depth=self.config.catboost_depth,
            l2_leaf_reg=self.config.catboost_l2_leaf_reg,
            random_seed=self.config.catboost_random_seed,
            early_stopping_rounds=self.config.catboost_early_stopping_rounds,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=100,
        )

        logger.info("Запуск обучения CatBoost...")
        model.fit(train_pool)

        logger.info("Калибровка вероятностей через Platt Scaling (sigmoid)...")
        calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        calibrated.fit(x_resampled, y_resampled)

        logger.info("Вычисление SHAP значений для интерпретации модели...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Pool(x_resampled))
        logger.info(f"SHAP computed: shape={shap_values.shape}")

        output_path = Path(self.config.model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, output_path)
        logger.info(f"Модель успешно сохранена: {output_path}")

        return model
