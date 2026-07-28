from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Stage(StrEnum):
    QUEUED = "queued"
    FETCH = "fetch"
    NORMALIZE = "normalize"
    SCENE_DETECT = "scene_detect"
    TRANSCRIBE = "transcribe"
    FRAME_SAMPLE = "frame_sample"
    OCR = "ocr"
    CAPTION = "caption"
    EMBED = "embed"
    SUMMARIZE = "summarize"
    DONE = "done"


# Stage progression for milestone 1: stages 3-8 are no-ops that just advance
# the recorded stage so progress reporting works end-to-end.
M1_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.FETCH,
    Stage.NORMALIZE,
    Stage.SCENE_DETECT,
    Stage.TRANSCRIBE,
    Stage.FRAME_SAMPLE,
    Stage.OCR,
    # After OCR (needs ocr_text to know which scenes have no evidence) and
    # before SUMMARIZE (which uses the caption instead of its empty-scene
    # constant). Resume keys off the stage *name*, so inserting here is safe
    # for in-flight jobs: one that had completed OCR simply runs CAPTION next.
    Stage.CAPTION,
    Stage.EMBED,
    Stage.SUMMARIZE,
)


class Job(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    asset_id: str | None = None
    source: str
    push_to_sbt: bool = False
    status: JobStatus = JobStatus.QUEUED
    stage: Stage = Stage.QUEUED
    progress: float = 0.0
    retries: int = 0
    error_kind: str | None = None
    error_message: str | None = None
    wait_ms: int = 30000
    runner_id: str | None = None
    runner_heartbeat_at: datetime | None = None
    pipeline_state: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
