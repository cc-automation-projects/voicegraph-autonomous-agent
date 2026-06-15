import logging
from typing import Dict

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.pii_sanitizer.recognizers import (
    ru_card_recognizer,
    ru_inn_recognizer,
    ru_passport_recognizer,
    ru_phone_recognizer,
)

logger = logging.getLogger(__name__)

class PIISanitizer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PIISanitizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Инициализация PIISanitizer (Microsoft Presidio)...")

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        registry.add_recognizer(ru_passport_recognizer)
        registry.add_recognizer(ru_inn_recognizer)
        registry.add_recognizer(ru_phone_recognizer)
        registry.add_recognizer(ru_card_recognizer)

        if "ru" not in registry.supported_languages:
            registry.supported_languages.append("ru")

        nlp_engine = SpacyNlpEngine(models=[
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "ru", "model_name": "ru_core_news_sm"},
        ])

        self.analyzer = AnalyzerEngine(
            registry=registry, supported_languages=["ru", "en"],
            nlp_engine=nlp_engine,
        )
        self.anonymizer = AnonymizerEngine()

        self.operators: Dict[str, OperatorConfig] = {
            "RU_PASSPORT": OperatorConfig("replace", {"new_value": "[PASSPORT_REDACTED]"}),
            "RU_INN": OperatorConfig("replace", {"new_value": "[INN_REDACTED]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CARD_NUMBER_REDACTED]"}),
            "PERSON": OperatorConfig("replace", {"new_value": "[NAME_REDACTED]"}),
        }

        self._initialized = True
        logger.info("PIISanitizer успешно инициализирован с поддержкой Luhn-проверки карт.")

    def sanitize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text

        try:
            analyzer_results = self.analyzer.analyze(
                text=text,
                language="ru",
            )

            if not analyzer_results:
                return text

            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=self.operators,
            )

            return anonymized_result.text

        except Exception as e:
            logger.error(f"Ошибка при маскировании PII: {e}. Текст возвращен без изменений.")
            return text


sanitizer = PIISanitizer()
