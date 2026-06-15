import phonenumbers
from presidio_analyzer import Pattern, PatternRecognizer

RU_PASSPORT_PATTERN = Pattern(
    name="RU_PASSPORT_PATTERN",
    regex=r"\b\d{4}\s?\d{6}\b",
    score=0.85,
)

ru_passport_recognizer = PatternRecognizer(
    supported_entity="RU_PASSPORT",
    supported_language="ru",
    patterns=[RU_PASSPORT_PATTERN],
)

RU_INN_PATTERN = Pattern(
    name="RU_INN_PATTERN",
    regex=r"\b(?:\d{10}|\d{12})\b",
    score=0.75,
)

ru_inn_recognizer = PatternRecognizer(
    supported_entity="RU_INN",
    supported_language="ru",
    patterns=[RU_INN_PATTERN],
)

RU_PHONE_PATTERN = Pattern(
    name="RU_PHONE_PATTERN",
    regex=r"(?:\+7|8)[\s\-]?(?:\(?\d{3}\)?)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
    score=0.85,
)

ru_phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    supported_language="ru",
    patterns=[RU_PHONE_PATTERN],
)

def luhn_checksum(card_number: str) -> bool:
    digits = [int(d) for d in card_number if d.isdigit()]
    if not digits:
        return False
    checksum = sum(digits[-1::-2]) + sum(sum(divmod(d * 2, 10)) for d in digits[-2::-2])
    return checksum % 10 == 0

class RuCreditCardRecognizer(PatternRecognizer):
    def __init__(self):
        pattern = Pattern(
            name="RU_CREDIT_CARD_PATTERN",
            regex=r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
            score=0.6,
        )
        super().__init__(supported_entity="CREDIT_CARD", supported_language="ru", patterns=[pattern])

    def analyze(self, text: str, entities: list, nlp_artifacts=None):
        results = super().analyze(text, entities, nlp_artifacts)
        valid_results = []
        for result in results:
            clean_num = "".join(filter(str.isdigit, text[result.start:result.end]))
            if luhn_checksum(clean_num):
                result.score = 0.95
                valid_results.append(result)
        return valid_results

ru_card_recognizer = RuCreditCardRecognizer()

def validate_phone(phone_str: str) -> bool:
    try:
        parsed = phonenumbers.parse(phone_str, "RU")
        return phonenumbers.is_possible_number(parsed)
    except phonenumbers.NumberParseException:
        return False
