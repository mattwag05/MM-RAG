"""What a core-only install (no m3-visual extra) does at each stage (MM-RAG-bdi).

Two different behaviours are correct here, and the split is the point:

* EMBED degrades — it writes no vectors and lets the job finish. That is what
  makes the ``transcript_only`` profile usable without torch: FTS over
  transcript and scene text is untouched, and search already falls back to
  FTS-only when the query cannot be encoded.
* FRAME_SAMPLE fails — a profile that samples frames cannot produce anything
  useful without Pillow, so it raises the typed error carrying the install
  hint rather than dying on a bare ModuleNotFoundError deep in a worker thread.

These tests run on both install shapes: they simulate the missing extra rather
than requiring its absence, so they are deliberately NOT marked ``m3_visual``.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import pytest

from mmrag.db.connection import connect, transaction
from mmrag.pipeline.m3_errors import M3ExtraMissingError
from mmrag.pipeline.stages import embed as embed_mod
from mmrag.pipeline.stages import frame_sample as frame_sample_mod


@pytest.mark.asyncio
async def test_embed_writes_no_vectors_when_the_extra_is_missing(monkeypatch) -> None:
    """A transcript_only ingest on a core install must not fail in EMBED."""

    def _no_extra(_texts: list[str]) -> list[list[float]]:
        raise M3ExtraMissingError(stage="embed")

    monkeypatch.setattr(embed_mod, "_encode_texts_sync", _no_extra)

    out = await embed_mod.embed(
        frames=[],
        scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 2.0}],
        segments=[{"seg_idx": 0, "text": "a spoken sentence"}],
    )

    assert out["vectors_written"] == 0
    assert out["segment_vectors"] == []
    assert out["frame_vectors"] == []
    assert out["scene_vectors"] == []


@pytest.mark.asyncio
async def test_frame_sample_raises_the_typed_error_without_pillow(
    monkeypatch, tmp_path: Path
) -> None:
    real_find_spec = importlib.util.find_spec

    def _pillow_absent(name: str, package: str | None = None):
        return None if name == "PIL" else real_find_spec(name, package)

    monkeypatch.setattr(frame_sample_mod.importlib.util, "find_spec", _pillow_absent)

    with pytest.raises(M3ExtraMissingError) as excinfo:
        await frame_sample_mod.frame_sample(
            mezzanine_path=str(tmp_path / "mezzanine.mp4"),
            scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 2.0}],
            assets_dir=tmp_path,
            content_hash="deadbeef",
        )

    assert excinfo.value.stage == "frame_sample"
    assert "m3-visual" in str(excinfo.value)


@pytest.mark.asyncio
async def test_runner_records_a_missing_extra_as_its_own_error_kind(
    monkeypatch, isolated_data_dir: Path
) -> None:
    """Without the dedicated handler this lands as kind 'unknown', which tells
    the caller nothing about the one thing that would fix it."""
    from mmrag.pipeline import runner as runner_mod

    job_id = str(uuid.uuid4())
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO jobs (id, source, status, stage, progress, wait_ms, pipeline_state_json)
            VALUES (?, ?, 'queued', 'queued', 0.0, 1000, ?)
            """,
            (job_id, "/nonexistent/clip.mp4", json.dumps({})),
        )

    async def _no_extra(stage, state):
        raise M3ExtraMissingError(stage="frame_sample")

    monkeypatch.setattr(runner_mod, "_run_stage", _no_extra)
    await runner_mod.run_pipeline(job_id)

    with connect() as conn:
        row = conn.execute(
            "SELECT status, error_kind, error_message FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()

    assert row["status"] == "error"
    assert row["error_kind"] == "m3_extra_missing"
    assert "m3-visual" in row["error_message"]
