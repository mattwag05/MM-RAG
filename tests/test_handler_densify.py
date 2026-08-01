"""Densify: re-sample an ingested time range at higher frame density (MM-RAG-nwk)."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from mmrag.db.connection import connect, transaction
from mmrag.handlers.densify import DensifyError, _plan, handle_densify
from mmrag.models.mcp_io import DensifyInput

pytestmark = pytest.mark.m3_visual


def _make_test_video(path: Path, duration: int = 6) -> None:
    # testsrc animates, so the near-duplicate dedup in frame_sample keeps the
    # densified frames instead of collapsing them into one.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=160x120:rate=24",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _seed_asset(mezzanine_path: Path | None = None) -> str:
    """One asset, one 0-6s scene, one already-sampled frame at frame_idx 0."""
    asset_id = str(uuid.uuid4())
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO assets (id, content_hash, source_kind, source_url,
                                mezzanine_path, metadata_json)
            VALUES (?, ?, 'file', 'file:///tmp/fake.mp4', ?, '{}')
            """,
            (asset_id, f"hash-{asset_id}", str(mezzanine_path) if mezzanine_path else None),
        )
        conn.execute(
            "INSERT INTO scenes (asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 6.0)",
            (asset_id,),
        )
        scene_id = conn.execute("SELECT id FROM scenes WHERE asset_id = ?", (asset_id,)).fetchone()[
            "id"
        ]
        conn.execute(
            """
            INSERT INTO frames (asset_id, scene_id, frame_idx, t_s, path, width, height)
            VALUES (?, ?, 0, 3.0, '/tmp/existing.jpg', 160, 120)
            """,
            (asset_id, scene_id),
        )
    return asset_id


def test_plan_starts_after_existing_frame_indices(isolated_data_dir):
    """A colliding frame_idx would silently UPDATE the ingest-time row instead
    of adding a frame — the whole point of densify."""
    asset_id = _seed_asset()
    plan = _plan(asset_id, 0.0, 2.0, 0.5)

    assert len(plan) == 1
    entry = plan[0]
    assert entry["scene_idx"] == 0
    assert entry["frame_idx_start"] == 1  # existing frame is frame_idx 0
    # Bucket midpoints inside the window, never on the boundary.
    assert entry["times"] == [0.25, 0.75, 1.25, 1.75]


def test_plan_skips_timestamps_an_existing_frame_already_covers(isolated_data_dir):
    """Without this, calling densify twice on a range doubles its frames."""
    asset_id = _seed_asset()  # existing frame at t_s = 3.0
    plan = _plan(asset_id, 2.5, 4.5, 1.0)

    # Buckets land on 3.0 and 4.0; the 3.0 one is exactly the existing frame,
    # which is what a repeated densify call with the same arguments produces.
    assert plan[0]["times"] == [4.0]


def test_plan_rejects_a_range_with_no_scenes(isolated_data_dir):
    asset_id = _seed_asset()
    with pytest.raises(DensifyError):
        _plan(asset_id, 60.0, 70.0, 0.5)


def test_plan_rejects_more_frames_than_the_cap(isolated_data_dir):
    asset_id = _seed_asset()
    with pytest.raises(DensifyError, match="limit is"):
        _plan(asset_id, 0.0, 6.0, 0.01)


async def test_densify_adds_frames_and_frame_vectors(isolated_data_dir, tmp_path):
    """End-to-end: the job runs FRAME_SAMPLE/OCR/EMBED in a child process and
    lands new frames plus their SigLIP vectors in the same index."""
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not on PATH; the densify pipeline runs the OCR stage")

    video = tmp_path / "testsrc.mp4"
    _make_test_video(video, duration=6)
    asset_id = _seed_asset(mezzanine_path=video)

    out = await handle_densify(
        DensifyInput(asset_id=asset_id, time_range=(1.0, 4.0), interval_s=1.0, wait_ms=600000)
    )

    assert out.status == "done", out.error
    assert out.frames_added > 0

    with connect() as conn:
        rows = conn.execute(
            "SELECT frame_idx, t_s, path FROM frames WHERE asset_id = ? AND frame_idx > 0",
            (asset_id,),
        ).fetchall()
        n_vecs = conn.execute(
            "SELECT COUNT(*) AS n FROM vec_frames WHERE asset_id = ?", (asset_id,)
        ).fetchone()["n"]

    assert len(rows) == out.frames_added
    assert all(1.0 <= float(r["t_s"]) <= 4.0 for r in rows)
    assert all(Path(r["path"]).exists() for r in rows)
    # The ingest-time frame at frame_idx 0 survived and was not overwritten.
    with connect() as conn:
        original = conn.execute(
            "SELECT path FROM frames WHERE asset_id = ? AND frame_idx = 0", (asset_id,)
        ).fetchone()
    assert original["path"] == "/tmp/existing.jpg"
    # Frame vectors cover the new frames (the original's path does not exist,
    # so it was never embedded).
    assert n_vecs >= len(rows)
