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
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8766
    mcp_path: str = "/mcp"
    mcp_token: str | None = None
    mcp_public_url: str | None = None
    log_level: str = "INFO"
    ingest_inline: bool = True
    worker_concurrency: int = 2
    graph_enabled: bool = True
    # Ingest-time VLM captioning of silent scenes. Off-switch for edge
    # deployments that cannot spare the 469 MB Florence-2 download.
    caption_enabled: bool = True
    # ASR model id resolved by onnx-asr. Parakeet TDT v3 is ~670 MB at int8;
    # edge deployments that cannot spare it can point this at a smaller
    # onnx-asr model such as "whisper-base". The quantization knob travels
    # with it because not every model ships an int8 variant; set it to the
    # empty string (MMRAG_TRANSCRIBE_QUANTIZATION="") to select fp32.
    transcribe_model: str = "nemo-parakeet-tdt-0.6b-v3"
    transcribe_quantization: str | None = "int8"
    # Fetch manual platform caption tracks and use them instead of ASR when
    # present (MM-RAG-8vj). Auto-captions are never fetched.
    subtitles_enabled: bool = True
    # Height ceiling for downloaded video (MM-RAG-7rm). On-screen text is
    # unreadable below ~720p — at 360p the pixels simply do not carry it — so
    # this is the lever that decides what OCR and captioning can ever see.
    # Lower it for bandwidth- or CPU-constrained deployments; raise it only if
    # you are prepared to pay the transcode and per-frame cost.
    max_video_height: int = 1080
    # Backend for the opt-in synthesize=true path (MM-RAG-thx). "ollama" talks
    # to MMRAG_OLLAMA_URL; "minicpm" loads MiniCPM-V-4.6 locally and can reason
    # over retrieved frame JPEGs as well as the evidence text. Neither runs
    # unless a caller passes synthesize=true.
    # WARNING: "minicpm" needs ~5.9 GB resident — do not enable it on an edge
    # box. See src/mmrag/providers/minicpm.py.
    synthesize_provider: str = "ollama"
    # Model repo for synthesize_provider="minicpm". Configurable so a smaller
    # or quantized variant is a config change, not a code change. Note that on
    # Apple silicon the only practical 4-bit route is MLX (mlx-community/*),
    # which needs the mlx-vlm runtime rather than transformers — see the
    # provider docstring before pointing this at one.
    synthesize_model: str = "openbmb/MiniCPM-V-4_6"
    # Frames handed to a vision-capable synthesize backend. Peak memory scales
    # with image count, so this is a memory guard, not a quality knob.
    synthesize_max_frames: int = 4
    query_vector_enabled: bool = True
    vector_backend: str = "sqlite"
    qdrant_url: str | None = None

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
