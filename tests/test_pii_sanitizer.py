from __future__ import annotations

import re

from src.pii_sanitizer.recognizers import (
    RU_INN_PATTERN,
    RU_PASSPORT_PATTERN,
    RU_PHONE_PATTERN,
)
from src.pii_sanitizer.service import PIISanitizer


class TestRecognizers:
    def test_passport_pattern(self):
        assert re.search(RU_PASSPORT_PATTERN.regex, "паспорт 4510 123456")

    def test_inn_pattern(self):
        assert re.search(RU_INN_PATTERN.regex, "ИНН 7712345678")

    def test_phone_pattern(self):
        assert re.search(RU_PHONE_PATTERN.regex, "+7 912 345-67-89")


class TestPIISanitizer:
    def setup_method(self):
        self.sanitizer = PIISanitizer()

    def test_mask_passport(self):
        result = self.sanitizer.sanitize("паспорт 4510 123456")
        assert "4510" not in result
        assert "123456" not in result

    def test_mask_phone(self):
        result = self.sanitizer.sanitize("номер +7 912 345-67-89")
        assert "+7 912 345-67-89" not in result

    def test_mask_inn(self):
        result = self.sanitizer.sanitize("ИНН 7712345678")
        assert "7712345678" not in result

    def test_clean_text_unchanged(self):
        text = "Здравствуйте, как ваши дела?"
        result = self.sanitizer.sanitize(text)
        assert result == text

    def test_multiple_pii(self):
        text = "паспорт 4510 123456, телефон +7 912 345-67-89"
        result = self.sanitizer.sanitize(text)
        assert "4510" not in result
        assert "123456" not in result
        assert "+7 912 345-67-89" not in result

    def test_singleton(self):
        s1 = PIISanitizer()
        s2 = PIISanitizer()
        assert s1 is s2
