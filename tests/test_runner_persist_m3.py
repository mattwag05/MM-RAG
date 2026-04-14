"""Runner persistence for M3: _persist_frames, _persist_vectors, _rewrite_fts_scenes."""

from __future__ import annotations

import uuid

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect
from mmrag.db.migrations import apply_migrations
from mmrag.pipeline.runner import (
    _persist_frames,
    _persist_scenes,
    _persist_vectors,
    _rewrite_fts_scenes,
)

pytestmark = pytest.mark.m3_visual


def _bootstrap_asset(tmp_path) -> str:
    reset_settings_for_tests(Settings(data_dir=tmp_path))
    apply_migrations()

    asset_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO assets (id, content_hash, source_kind, source_url, metadata_json)
            VALUES (?, ?, 'file', 'file:///tmp/fake.mp4', '{}')
            """,
            (asset_id, f"hash-{asset_id}"),
        )
    return asset_id


def test_persist_frames_and_rewrite_fts_scenes(tmp_path):
    try:
        asset_id = _bootstrap_asset(tmp_path)
        _persist_scenes(
            asset_id=asset_id,
            scenes=[
                {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
                {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
            ],
        )
        with connect() as conn:
            id_by_idx = {
                int(r["scene_idx"]): int(r["id"])
                for r in conn.execute(
                    "SELECT id, scene_idx FROM scenes WHERE asset_id = ?", (asset_id,)
                ).fetchall()
            }

        frames = [
            {
                "scene_idx": 0, "frame_idx": 0, "t_s": 1.0,
                "path": "/tmp/a.jpg", "width": 100, "height": 80,
                "ocr_text": "red color bars",
            },
            {
                "scene_idx": 1, "frame_idx": 0, "t_s": 3.0,
                "path": "/tmp/b.jpg", "width": 100, "height": 80,
                "ocr_text": "weather map",
            },
        ]
        frame_id_map = _persist_frames(
            asset_id=asset_id, scene_id_by_idx=id_by_idx, frames=frames
        )
        assert len(frame_id_map) == 2

        _rewrite_fts_scenes(asset_id=asset_id)

        with connect() as conn:
            rows = conn.execute(
                "SELECT rowid FROM fts_scenes WHERE fts_scenes MATCH 'red'"
            ).fetchall()
            assert len(rows) == 1
            assert int(rows[0]["rowid"]) == id_by_idx[0]
    finally:
        reset_settings_for_tests(Settings())


def test_persist_vectors_writes_all_three_vec_tables(tmp_path):
    try:
        asset_id = _bootstrap_asset(tmp_path)
        _persist_scenes(
            asset_id=asset_id,
            scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 1.0}],
        )
        with connect() as conn:
            scene_id_by_idx = {
                int(r["scene_idx"]): int(r["id"])
                for r in conn.execute(
                    "SELECT id, scene_idx FROM scenes WHERE asset_id=?", (asset_id,)
                ).fetchall()
            }

        frame_id_map = _persist_frames(
            asset_id=asset_id,
            scene_id_by_idx=scene_id_by_idx,
            frames=[{
                "scene_idx": 0, "frame_idx": 0, "t_s": 0.5,
                "path": "/tmp/x.jpg", "width": 100, "height": 80,
                "ocr_text": "",
            }],
        )

        with connect() as conn:
            conn.execute(
                """
                INSERT INTO transcript_segments
                    (asset_id, scene_id, seg_idx, start_s, end_s, text)
                VALUES (?, ?, 0, 0.0, 1.0, 'hello')
                """,
                (asset_id, scene_id_by_idx[0]),
            )
            seg_row = conn.execute(
                "SELECT id FROM transcript_segments WHERE asset_id=?", (asset_id,)
            ).fetchone()
            seg_id_by_idx = {0: int(seg_row["id"])}

        v_frame = [0.0] * 768
        v_frame[0] = 1.0
        v_scene = [0.0] * 768
        v_scene[1] = 1.0
        v_seg = [0.0] * 768
        v_seg[2] = 1.0

        _persist_vectors(
            frame_id_map=frame_id_map,
            scene_id_by_idx=scene_id_by_idx,
            segment_id_by_idx=seg_id_by_idx,
            frame_vectors=[{"scene_idx": 0, "frame_idx": 0, "vector": v_frame}],
            scene_vectors=[{"scene_idx": 0, "vector": v_scene}],
            segment_vectors=[{"seg_idx": 0, "vector": v_seg}],
        )

        with connect() as conn:
            assert conn.execute("SELECT COUNT(*) AS n FROM vec_frames").fetchone()["n"] == 1
            assert conn.execute("SELECT COUNT(*) AS n FROM vec_scenes").fetchone()["n"] == 1
            assert conn.execute("SELECT COUNT(*) AS n FROM vec_transcript").fetchone()["n"] == 1
    finally:
        reset_settings_for_tests(Settings())
