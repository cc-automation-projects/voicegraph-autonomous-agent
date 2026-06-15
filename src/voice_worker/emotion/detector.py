import asyncio
import logging
from enum import Enum
from pathlib import Path

import joblib
import librosa
import numpy as np
import opensmile

logger = logging.getLogger(__name__)


class EmotionState(str, Enum):
    CALM = "CALM"
    ANNOYED = "ANNOYED"
    CONFUSED = "CONFUSED"
    ANGRY = "ANGRY"
    UNKNOWN = "UNKNOWN"


class EmotionDetector:
    def __init__(self, model_path: str | None = None):
        logger.info("Инициализация Emotion Detector (OpenSMILE eGeMAPSv02)...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.model_path = Path(model_path) if model_path else Path("models/emotion_classifier.joblib")
        self.classifier = None

        if self.model_path.exists():
            self.classifier = joblib.load(self.model_path)
            logger.info(f"Модель классификации эмоций загружена из {self.model_path}")
        else:
            logger.warning("Модель классификации не найдена. Используется эвристическая заглушка.")

    def extract_and_classify(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> EmotionState:
        if len(audio_buffer) == 0:
            return EmotionState.UNKNOWN

        try:
            if sample_rate != 16000:
                audio_buffer = librosa.resample(audio_buffer, orig_sr=sample_rate, target_sr=16000)

            audio_buffer = librosa.util.normalize(audio_buffer)

            features = self.smile.process_signal(audio_buffer, 16000)
            feature_vector = features.values.reshape(1, -1)

            if self.classifier is not None:
                prediction = str(self.classifier.predict(feature_vector)[0])
                confidence = float(np.max(self.classifier.predict_proba(feature_vector)))

                if confidence < 0.6:
                    return EmotionState.CALM

                return EmotionState(prediction.upper())
            else:
                energy = np.mean(np.abs(audio_buffer))
                if energy > 0.15:
                    return EmotionState.ANGRY
                return EmotionState.CALM

        except Exception as e:
            logger.error(f"Ошибка при анализе эмоций: {e}")
            return EmotionState.UNKNOWN

    async def extract_and_classify_async(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> EmotionState:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.extract_and_classify,
            audio_buffer,
            sample_rate,
        )
