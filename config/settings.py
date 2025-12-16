import os
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


def load_criteria_from_yaml() -> dict[str, str]:
    yaml_path = BASE_DIR / "config" / "criteria.yaml"
    if not yaml_path.exists():
        # Fallback default if file missing
        return {
            "history_taking": "Сбор анамнеза (полнота, уточнение деталей)",
            "clinical_reasoning": "Клиническое мышление (аргументация, гипотезы)",
            "communication": "Коммуникация (ясность, эмпатия, ответы на вопросы)",
            "safety": "Безопасность (проверка аллергий, совместимости)",
        }

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "medbro"

    # Models
    OPENAI_API_KEY: str = Field(default="")
    DEEPGRAM_API_KEY: str = Field(default="")

    DEEPGRAM_MODEL: str = "nova-3"

    LLM_MODEL: str = "gpt-5.2"
    STT_MODEL: str = "gpt-4o-transcribe"
    STT_DIARIZATION_MODEL: str = "gpt-4o-transcribe-diarize"

    TTS_MODEL: str = "tts-1-hd"
    DEFAULT_TTS_VOICE: str = "sage"

    # Mocking
    USE_MOCK_SERVICES: bool = False

    # Application Paths
    TEMP_DIR: Path = BASE_DIR / "_temp"
    DATA_DIR: Path = BASE_DIR / "_data"

    # Criteria (Doctor Evaluation)
    EVALUATION_CRITERIA: dict[str, str] = Field(default_factory=load_criteria_from_yaml)


config = AppConfig()

# Ensure temp dir exists
os.makedirs(config.TEMP_DIR, exist_ok=True)
os.makedirs(config.DATA_DIR, exist_ok=True)
