"""Migration 0003 renames shots -> scenes, adds frames, vec_*, fts_scenes."""

from __future__ import annotations

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect
from mmrag.db.migrations import apply_migrations

pytestmark = pytest.mark.m3_visual  # vec_* requires the extra


def _table_names(conn) -> set[str]:
    """Return the names of all regular and virtual tables (virtual tables
    surface as type='table' in sqlite_master — there is no 'virtual' type)."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r["name"] for r in rows}


def test_migration_0003_applies_cleanly(tmp_path):
    reset_settings_for_tests(Settings(data_dir=tmp_path))
    try:
        apply_migrations()
        with connect() as conn:
            names = _table_names(conn)

            # Rename: scenes exists, shots does not.
            assert "scenes" in names
            assert "shots" not in names

            # New table.
            assert "frames" in names

            # sqlite-vec virtual tables.
            assert "vec_frames" in names
            assert "vec_scenes" in names
            assert "vec_transcript" in names

            # Plain FTS5 scenes index.
            assert "fts_scenes" in names

            # scenes has a new summary column.
            cols = {
                r["name"]: r["type"] for r in conn.execute("PRAGMA table_info(scenes)").fetchall()
            }
            assert "summary" in cols
            assert "scene_idx" in cols  # renamed from shot_idx
            assert "shot_idx" not in cols

            # transcript_segments column rename.
            seg_cols = {
                r["name"] for r in conn.execute("PRAGMA table_info(transcript_segments)").fetchall()
            }
            assert "scene_id" in seg_cols
            assert "shot_id" not in seg_cols
    finally:
        reset_settings_for_tests(Settings())


def test_migration_0003_vec_tables_are_writable(tmp_path):
    reset_settings_for_tests(Settings(data_dir=tmp_path))
    try:
        apply_migrations()
        import struct

        blob = struct.pack("768f", *([0.0] * 768))
        with connect() as conn:
            conn.execute(
                "INSERT INTO vec_frames(rowid, embedding) VALUES (1, ?)",
                (blob,),
            )
            row = conn.execute("SELECT COUNT(*) AS n FROM vec_frames").fetchone()
            assert row["n"] == 1
    finally:
        reset_settings_for_tests(Settings())
