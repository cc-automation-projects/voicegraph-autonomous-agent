from __future__ import annotations

from src.voicegraph.config import Settings


class TestSettings:
    def test_default_values(self):
        settings = Settings()
        assert settings.app_name == "VoiceGraph"
        assert settings.debug is False
        assert settings.log_level == "INFO"

    def test_env_prefix(self):
        settings = Settings()
        assert hasattr(settings, "database_url")
        assert hasattr(settings, "yandexgpt_api_key")
