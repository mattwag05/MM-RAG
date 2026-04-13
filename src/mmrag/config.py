from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    return Path.home() / ".local" / "share" / "mmrag"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MMRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default_factory=_default_data_dir)
    ollama_url: str = "http://localhost:11434"
    model_primary: str = "gemma4:e4b"
    model_fallback: str = "gemma4:e2b"
    sbt_url: str | None = None
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    log_level: str = "INFO"
    worker_concurrency: int = 2

    @property
    def db_path(self) -> Path:
        return self.data_dir / "mmrag.db"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_tests(settings: Settings) -> None:
    global _settings
    _settings = settings
