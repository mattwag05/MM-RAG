from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class IngestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="URL or local file path")
    mode: Literal["standard", "shortform"] = "standard"
    wait_ms: int = Field(30000, ge=0, le=600000)
    push_to_sbt: bool = False


class IngestOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["done", "in_progress", "error"]
    asset_id: str | None = None
    job_id: str | None = None
    summary: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


class AskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    asset_id: str | None = None
    time_range: tuple[float, float] | None = None
    top_k: int = Field(5, ge=1, le=50)
    model: Literal["gemma4:e4b", "gemma4:e2b"] = "gemma4:e4b"
    synthesize: bool = False


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    scene_id: str | None = None
    frame_id: str | None = None
    start_s: float
    end_s: float
    source_stream: str = "hybrid"
    snippet: str | None = None
    score: float | None = None
    summary: str | None = None
    ocr_snippet: str | None = None
    transcript_snippet: str | None = None


class AskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    asset_id: str | None = None
    top_k: int = Field(10, ge=1, le=100)
    mode: Literal["hybrid", "vector", "fts"] = "hybrid"


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    scene_id: str | None = None
    frame_id: str | None = None
    start_s: float
    end_s: float
    score: float
    snippet: str | None = None
    source_stream: str = "hybrid"


class SearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[SearchHit] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str


class StatusOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["queued", "running", "done", "error"]
    stage: str
    progress: float = Field(0.0, ge=0.0, le=1.0)
    asset_id: str | None = None
    error: str | None = None
