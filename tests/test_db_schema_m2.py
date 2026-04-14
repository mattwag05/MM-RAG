"""M2 schema: scenes + transcript_segments + fts_transcript."""

from __future__ import annotations

from pathlib import Path

from mmrag.db.connection import connect


def _column_names(conn, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_scenes_table_has_expected_columns(isolated_data_dir: Path) -> None:
    with connect() as conn:
        cols = _column_names(conn, "scenes")
    assert {"id", "asset_id", "scene_idx", "start_s", "end_s"} <= cols


def test_transcript_segments_table_has_expected_columns(isolated_data_dir: Path) -> None:
    with connect() as conn:
        cols = _column_names(conn, "transcript_segments")
    assert {
        "id",
        "asset_id",
        "scene_id",
        "seg_idx",
        "start_s",
        "end_s",
        "text",
    } <= cols


def test_fts_transcript_is_fts5(isolated_data_dir: Path) -> None:
    with connect() as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'fts_transcript'"
        ).fetchone()
    assert row is not None
    assert "fts5" in row["sql"].lower()


def test_fts_trigger_mirrors_segment_text(isolated_data_dir: Path) -> None:
    """Inserting a transcript_segments row must make the text BM25-searchable."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind) VALUES "
            "('a1', 'h1', 'file')"
        )
        conn.execute(
            """
            INSERT INTO transcript_segments
                (asset_id, scene_id, seg_idx, start_s, end_s, text)
            VALUES ('a1', NULL, 0, 0.0, 1.0, 'hello mmrag world')
            """
        )
        row = conn.execute(
            """
            SELECT ts.text
              FROM transcript_segments ts
              JOIN fts_transcript ON fts_transcript.rowid = ts.id
             WHERE fts_transcript MATCH 'mmrag'
            """
        ).fetchone()
    assert row is not None
    assert row["text"] == "hello mmrag world"


def test_fts_trigger_updates_on_delete(isolated_data_dir: Path) -> None:
    """Deleting a transcript_segments row must purge it from the FTS index."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind) VALUES "
            "('a1', 'h1', 'file')"
        )
        conn.execute(
            """
            INSERT INTO transcript_segments
                (asset_id, scene_id, seg_idx, start_s, end_s, text)
            VALUES ('a1', NULL, 0, 0.0, 1.0, 'unique_token_xyz')
            """
        )
        conn.execute("DELETE FROM transcript_segments WHERE asset_id = 'a1'")
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM fts_transcript "
            "WHERE fts_transcript MATCH 'unique_token_xyz'"
        ).fetchone()["c"]
    assert count == 0
