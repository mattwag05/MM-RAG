from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Asset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    content_hash: str
    source_url: str | None = None
    source_kind: str  # 'url' | 'file'
    title: str | None = None
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    mezzanine_path: str | None = None
    audio_path: str | None = None
    ingested_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)
