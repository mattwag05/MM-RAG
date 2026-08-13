from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class IngestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="URL or local file path")
    wait_ms: int = Field(30000, ge=0, le=600000)
    # "transcript_only" skips frame sampling, OCR, and captioning — the three
    # stages that dominate ingest cost — for bulk runs where speech is the
    # point. Scene detection and transcript embeddings still run.
    profile: Literal["full", "transcript_only"] = "full"


class IngestOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["done", "in_progress", "error"]
    asset_id: str | None = None
    job_id: str | None = None
    summary: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# densify
# ---------------------------------------------------------------------------


class DensifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    time_range: tuple[float, float]
    # Ingest samples a scene midpoint plus a 2s stride on long scenes, so
    # anything under 2s is a genuine densification. The 0.1s floor is the
    # point where fast-seek frame extraction stops resolving distinct frames.
    interval_s: float = Field(0.5, ge=0.1, le=5.0)
    wait_ms: int = Field(60000, ge=0, le=600000)

    @model_validator(mode="after")
    def _time_range_ordered(self) -> DensifyInput:
        if self.time_range[0] >= self.time_range[1]:
            raise ValueError("time_range start must be < end")
        return self


class DensifyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["done", "in_progress", "error"]
    asset_id: str | None = None
    job_id: str | None = None
    frames_added: int = 0
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
    # Free-form and ignored unless synthesize=true. It used to be a Literal of
    # two Ollama Gemma tags, which advertised those as the only legal models
    # while the actual backend is chosen by MMRAG_SYNTHESIZE_PROVIDER — a tag
    # the caller can neither see nor influence. None means "whatever the
    # configured provider defaults to" (docs/pmf-rethink.md).
    model: str | None = None
    synthesize: bool = False
    include_frames: bool = False

    @model_validator(mode="after")
    def _time_range_ordered(self) -> AskInput:
        if self.time_range is not None and self.time_range[0] > self.time_range[1]:
            raise ValueError("time_range start must be <= end")
        return self


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    content_item_id: str | None = None
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
    # VLM caption written at ingest for scenes with no speech and no
    # on-screen text.
    caption: str | None = None
    # Local path of the hit's frame JPEG (or the scene's representative
    # frame). Populated only when the caller opts in with include_frames —
    # stdio MCP runs on the same host, so the consuming agent can open it.
    frame_path: str | None = None
    # Set only when this hit's scene is thinly sampled for its duration, and
    # it names the remedy (`densify`). Absent means coverage is normal — the
    # agent should not have to reason about frame counts to notice a gap.
    coverage_note: str | None = None


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
    mode: Literal["hybrid", "vector", "fts", "hybrid_graph"] = "hybrid"
    time_range: tuple[float, float] | None = None
    include_frames: bool = False

    @model_validator(mode="after")
    def _time_range_ordered(self) -> SearchInput:
        if self.time_range is not None and self.time_range[0] > self.time_range[1]:
            raise ValueError("time_range start must be <= end")
        return self


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    content_item_id: str | None = None
    scene_id: str | None = None
    frame_id: str | None = None
    start_s: float
    end_s: float
    score: float
    snippet: str | None = None
    source_stream: str = "hybrid"
    # See Evidence.frame_path — opt-in via SearchInput.include_frames.
    frame_path: str | None = None
    # See Evidence.coverage_note. Always evaluated; usually None.
    coverage_note: str | None = None


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
