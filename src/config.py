import os
import yaml
from pathlib import Path
from typing import Any, Dict
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=4, env="API_WORKERS")

    NLP_MODEL_NAME: str = Field(default="bert-base-multilingual-cased", env="NLP_MODEL_NAME")
    VISION_MODEL_NAME: str = Field(default="efficientnet_b0", env="VISION_MODEL_NAME")
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    MAX_LENGTH: int = Field(default=512, env="MAX_LENGTH")

    ELASTICSEARCH_HOST: str = Field(default="localhost", env="ELASTICSEARCH_HOST")
    ELASTICSEARCH_PORT: int = Field(default=9200, env="ELASTICSEARCH_PORT")
    ELASTICSEARCH_INDEX: str = Field(default="ai_sentinel", env="ELASTICSEARCH_INDEX")

    # GDELT Configuration (Free, Open Access - No API Key Required)
    GDELT_BASE_URL: str = Field(default="https://api.gdeltproject.org/api/v2/doc/doc", env="GDELT_BASE_URL")
    GDELT_CACHE_DAYS: int = Field(default=7, env="GDELT_CACHE_DAYS")

    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/ai_sentinel.log", env="LOG_FILE")

    DASHBOARD_PORT: int = Field(default=8501, env="DASHBOARD_PORT")
    DASHBOARD_TITLE: str = Field(default="AI Sentinel Dashboard", env="DASHBOARD_TITLE")

    DATA_DIR: str = Field(default="data", env="DATA_DIR")
    MODELS_DIR: str = Field(default="models", env="MODELS_DIR")
    CACHE_DIR: str = Field(default="cache", env="CACHE_DIR")

    HATE_SPEECH_THRESHOLD: float = Field(default=0.75, env="HATE_SPEECH_THRESHOLD")
    DEEPFAKE_THRESHOLD: float = Field(default=0.80, env="DEEPFAKE_THRESHOLD")
    DISINFORMATION_THRESHOLD: float = Field(default=0.70, env="DISINFORMATION_THRESHOLD")

    class Config:
        env_file = ".env"
        case_sensitive = True


def load_yaml_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    root_dir = Path(__file__).parent.parent
    config_file = root_dir / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


settings = Settings()
yaml_config = load_yaml_config()


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def ensure_directories():
    root = get_project_root()
    directories = [
        root / settings.DATA_DIR / "raw",
        root / settings.DATA_DIR / "processed",
        root / settings.CACHE_DIR,
        root / settings.MODELS_DIR,
        root / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()