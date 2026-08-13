"""Ingest profiles: transcript_only skips the visual pipeline (MM-RAG-3c6)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mmrag.db.connection import connect
from mmrag.handlers.ingest import handle_ingest
from mmrag.handlers.search import handle_search
from mmrag.models.job import (
    DENSIFY_STAGE_ORDER,
    M1_STAGE_ORDER,
    TRANSCRIPT_ONLY_STAGE_ORDER,
    Stage,
)
from mmrag.models.mcp_io import IngestInput, SearchInput
from mmrag.pipeline.runner import _stage_order
from tests.conftest import SAMPLE_MP4, SPEECH_PHRASE


def test_stage_order_selection() -> None:
    assert _stage_order({}) is M1_STAGE_ORDER
    assert _stage_order({"profile": "full"}) is M1_STAGE_ORDER
    assert _stage_order({"profile": "transcript_only"}) is TRANSCRIPT_ONLY_STAGE_ORDER
    assert _stage_order({"densify": True}) is DENSIFY_STAGE_ORDER
    # densify wins: a densify job has no profile of its own to honour.
    assert _stage_order({"densify": True, "profile": "transcript_only"}) is DENSIFY_STAGE_ORDER


def test_transcript_only_drops_exactly_the_visual_stages() -> None:
    dropped = set(M1_STAGE_ORDER) - set(TRANSCRIPT_ONLY_STAGE_ORDER)
    assert dropped == {Stage.FRAME_SAMPLE, Stage.OCR, Stage.CAPTION}
    # Order is otherwise preserved, so resume-by-stage-name keeps working.
    assert list(TRANSCRIPT_ONLY_STAGE_ORDER) == [s for s in M1_STAGE_ORDER if s not in dropped]


@pytest.mark.asyncio
async def test_transcript_only_ingest_writes_no_frames(isolated_data_dir: Path) -> None:
    """The visual stages are the expensive half; transcript_only must leave no
    frames behind while still producing the scene structure."""
    result = await handle_ingest(
        IngestInput(source=str(SAMPLE_MP4), wait_ms=120000, profile="transcript_only")
    )
    assert result.status == "done", f"expected done, got {result.status}: {result.error}"
    asset_id = result.asset_id
    assert asset_id is not None

    with connect() as conn:
        n_frames = conn.execute(
            "SELECT COUNT(*) AS n FROM frames WHERE asset_id = ?", (asset_id,)
        ).fetchone()["n"]
        n_scenes = conn.execute(
            "SELECT COUNT(*) AS n FROM scenes WHERE asset_id = ?", (asset_id,)
        ).fetchone()["n"]
        n_frame_vecs = conn.execute(
            "SELECT COUNT(*) AS n FROM vec_frames WHERE asset_id = ?", (asset_id,)
        ).fetchone()["n"]

    assert n_frames == 0
    assert n_frame_vecs == 0
    assert n_scenes >= 1

    # No frames dir was written either — the stage never ran.
    frames_dirs = list((isolated_data_dir / "assets").glob("*/frames/*.jpg"))
    assert frames_dirs == []


@pytest.mark.asyncio
async def test_transcript_only_still_supports_transcript_search(
    isolated_data_dir: Path, speech_mp4: Path
) -> None:
    """Dropping the visual stages must not cost transcript retrieval."""
    result = await handle_ingest(
        IngestInput(source=str(speech_mp4), wait_ms=600000, profile="transcript_only")
    )
    assert result.status == "done", f"expected done, got {result.status}: {result.error}"

    with connect() as conn:
        n_segments = conn.execute(
            "SELECT COUNT(*) AS n FROM transcript_segments WHERE asset_id = ?",
            (result.asset_id,),
        ).fetchone()["n"]
        n_seg_vecs = conn.execute(
            "SELECT COUNT(*) AS n FROM vec_transcript WHERE asset_id = ?",
            (result.asset_id,),
        ).fetchone()["n"]
    assert n_segments >= 1
    if importlib.util.find_spec("open_clip") is not None:
        # EMBED is kept in the profile precisely so vector-mode transcript
        # search keeps working; this is the assertion that would catch
        # dropping it.
        assert n_seg_vecs >= 1
    else:
        # Core-only install: EMBED degrades to no vectors rather than failing
        # the job (MM-RAG-bdi). The FTS assertion below is what still has to
        # hold, and it is the whole point of the profile on that install.
        assert n_seg_vecs == 0

    out = await handle_search(
        SearchInput(query=SPEECH_PHRASE.split()[0], asset_id=result.asset_id, top_k=5)
    )
    assert out.hits, "transcript search returned nothing after a transcript_only ingest"
