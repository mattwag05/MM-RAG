"""Sparse-coverage hints on search hits (MM-RAG-4k5)."""

from __future__ import annotations

import uuid

from mmrag.db.connection import connect, transaction
from mmrag.handlers.search import _attach_coverage_notes
from mmrag.models.mcp_io import SearchHit


def _scene_with_frames(*, duration_s: float, n_frames: int) -> str:
    """Insert one scene of the given duration with n_frames frames. Returns its id."""
    asset_id = str(uuid.uuid4())
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO assets (id, content_hash, source_kind, metadata_json)
            VALUES (?, ?, 'file', '{}')
            """,
            (asset_id, f"hash-{asset_id}"),
        )
        conn.execute(
            "INSERT INTO scenes (asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, ?)",
            (asset_id, duration_s),
        )
        scene_id = conn.execute("SELECT id FROM scenes WHERE asset_id = ?", (asset_id,)).fetchone()[
            "id"
        ]
        for i in range(n_frames):
            conn.execute(
                """
                INSERT INTO frames (asset_id, scene_id, frame_idx, t_s, path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (asset_id, scene_id, i, float(i), f"/tmp/f{i}.jpg"),
            )
    return str(scene_id)


def _hit(scene_id: str) -> SearchHit:
    return SearchHit(asset_id="a", scene_id=scene_id, start_s=0.0, end_s=1.0, score=1.0)


def test_long_scene_with_one_frame_is_flagged(isolated_data_dir):
    """A 40s static shot deduped down to one frame is the case that used to be
    indistinguishable from 'there is nothing here'."""
    scene_id = _scene_with_frames(duration_s=40.0, n_frames=1)
    (hit,) = _attach_coverage_notes([_hit(scene_id)])

    assert hit.coverage_note is not None
    assert "1 frame sampled across 40.0s" in hit.coverage_note
    # The note has to name the remedy, or the agent has nothing to act on.
    assert "densify" in hit.coverage_note


def test_scene_with_no_frames_is_flagged(isolated_data_dir):
    """A transcript_only ingest leaves scenes with zero frames."""
    scene_id = _scene_with_frames(duration_s=30.0, n_frames=0)
    (hit,) = _attach_coverage_notes([_hit(scene_id)])

    assert hit.coverage_note is not None
    assert "0 frames sampled" in hit.coverage_note


def test_well_sampled_scene_is_not_flagged(isolated_data_dir):
    """20 frames over 30s is normal ingest density — no note, or the agent
    learns to ignore the field."""
    scene_id = _scene_with_frames(duration_s=30.0, n_frames=20)
    (hit,) = _attach_coverage_notes([_hit(scene_id)])

    assert hit.coverage_note is None


def test_short_scene_with_its_single_midpoint_frame_is_not_flagged(isolated_data_dir):
    """Ingest samples exactly one frame for a scene under 10s by design;
    flagging those would fire the note on most hits."""
    scene_id = _scene_with_frames(duration_s=3.0, n_frames=1)
    (hit,) = _attach_coverage_notes([_hit(scene_id)])

    assert hit.coverage_note is None


def test_hits_without_a_scene_are_left_alone(isolated_data_dir):
    """Document content_items hits carry no scene and cannot be densified."""
    hit = SearchHit(asset_id="a", content_item_id="c1", start_s=0.0, end_s=1.0, score=1.0)
    (out,) = _attach_coverage_notes([hit])

    assert out.coverage_note is None
