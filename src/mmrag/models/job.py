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

# Densify: re-sample an already-ingested time range at higher frame density
# (MM-RAG-nwk). Everything before FRAME_SAMPLE is already persisted, so the
# job resumes straight into sampling with a pre-seeded pipeline state.
#
# CAPTION is deliberately absent, and stays absent (decided in MM-RAG-ot7).
# Ingest captions silent scenes because no agent is in the loop there. Densify
# is always agent-initiated and hands back frame JPEG paths, so a 0.23B local
# caption would duplicate — worse — what the calling agent gets by opening the
# frame itself. The eligible population is also small and shrinking: raising
# the source resolution (MM-RAG-7rm) cut frames with no OCR text at all from
# 39/103 to 15/103 on the reference asset.
# Mechanical blocker if this is ever revisited: _frames_needing_caption
# targets frame_idx 0, densified frames start at max(frame_idx)+1, and
# frame_idx 0 is the scene MIDPOINT — so "lowest t_s" is not an equivalent
# rule and would change ingest-time behaviour.
# SUMMARIZE is absent because it would rewrite scene summaries from a
# partial state; the OCR persist branch already refreshes fts_scenes and
# content_items from the DB.
DENSIFY_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.FRAME_SAMPLE,
    Stage.OCR,
    Stage.EMBED,
)

# transcript_only: speech and scene structure, no visual pipeline (MM-RAG-3c6).
# For bulk ingestion where the transcript is the whole point, this drops the
# three stages that dominate wall-clock and memory — frame extraction, OCR,
# and VLM captioning.
#
# EMBED stays: with no frames it encodes only transcript segments, which is
# what vector-mode search over speech needs. With the m3-visual extra present
# it loads SigLIP for the text tower, so the profile saves time and
# frames-on-disk rather than the model download. Without the extra, EMBED
# degrades to writing no vectors instead of failing the job (MM-RAG-bdi), which
# is what makes this profile usable on a core-only install: FTS retrieval over
# transcript and scene text is unaffected.
# SUMMARIZE stays and simply summarises from segments alone.
TRANSCRIPT_ONLY_STAGE_ORDER: tuple[Stage, ...] = (
    Stage.FETCH,
    Stage.NORMALIZE,
    Stage.SCENE_DETECT,
    Stage.TRANSCRIBE,
    Stage.EMBED,
    Stage.SUMMARIZE,
)


class Job(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    asset_id: str | None = None
    source: str
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
