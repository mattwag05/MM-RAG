"""Runner persistence for M2: scenes + transcript_segments after each stage."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.db.connection import connect
from mmrag.pipeline.runner import _persist_segments, _persist_scenes


def _seed_asset(asset_id: str, content_hash: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind) VALUES (?, ?, 'file')",
            (asset_id, content_hash),
        )


def test_persist_scenes_writes_rows(isolated_data_dir: Path) -> None:
    _seed_asset("a1", "h1")
    _persist_scenes(
        asset_id="a1",
        scenes=[
            {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
            {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
        ],
    )
    with connect() as conn:
        rows = conn.execute(
            "SELECT scene_idx, start_s, end_s FROM scenes "
            "WHERE asset_id = 'a1' ORDER BY scene_idx"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["scene_idx"] == 0
    assert rows[0]["end_s"] == pytest.approx(2.0)
    assert rows[1]["scene_idx"] == 1


def test_persist_scenes_is_idempotent(isolated_data_dir: Path) -> None:
    """Re-running the stage on the same asset must not produce duplicates."""
    _seed_asset("a2", "h2")
    scenes = [
        {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
        {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
    ]
    _persist_scenes(asset_id="a2", scenes=scenes)
    _persist_scenes(asset_id="a2", scenes=scenes)
    with connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM scenes WHERE asset_id = 'a2'"
        ).fetchone()["c"]
    assert count == 2


def test_persist_segments_writes_rows_and_fts(isolated_data_dir: Path) -> None:
    _seed_asset("a3", "h3")
    _persist_scenes(
        asset_id="a3",
        scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 3.0}],
    )
    _persist_segments(
        asset_id="a3",
        segments=[
            {
                "seg_idx": 0,
                "start_s": 0.0,
                "end_s": 1.5,
                "text": "hello mmrag world",
                "scene_idx": 0,
            },
            {
                "seg_idx": 1,
                "start_s": 1.5,
                "end_s": 3.0,
                "text": "testing one two three",
                "scene_idx": 0,
            },
        ],
    )
    with connect() as conn:
        rows = conn.execute(
            "SELECT seg_idx, text, scene_id FROM transcript_segments "
            "WHERE asset_id = 'a3' ORDER BY seg_idx"
        ).fetchall()
    assert [r["text"] for r in rows] == ["hello mmrag world", "testing one two three"]
    # scene_id should be the INTEGER row id of the scenes row, not the scene_idx.
    assert rows[0]["scene_id"] is not None
    assert rows[1]["scene_id"] is not None

    # And the text should be BM25-searchable via the FTS index.
    with connect() as conn:
        hit = conn.execute(
            "SELECT ts.text FROM transcript_segments ts "
            "JOIN fts_transcript ON fts_transcript.rowid = ts.id "
            "WHERE fts_transcript MATCH 'mmrag'"
        ).fetchone()
    assert hit is not None
    assert "mmrag" in hit["text"]


def test_persist_segments_is_idempotent(isolated_data_dir: Path) -> None:
    _seed_asset("a4", "h4")
    _persist_scenes(
        asset_id="a4",
        scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 2.0}],
    )
    segments = [
        {
            "seg_idx": 0,
            "start_s": 0.0,
            "end_s": 1.0,
            "text": "unique_token_abcd",
            "scene_idx": 0,
        },
    ]
    _persist_segments(asset_id="a4", segments=segments)
    _persist_segments(asset_id="a4", segments=segments)
    with connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM transcript_segments WHERE asset_id = 'a4'"
        ).fetchone()["c"]
    assert count == 1
    # FTS should also contain only one copy.
    with connect() as conn:
        fts_count = conn.execute(
            "SELECT COUNT(*) AS c FROM fts_transcript "
            "WHERE fts_transcript MATCH 'unique_token_abcd'"
        ).fetchone()["c"]
    assert fts_count == 1
