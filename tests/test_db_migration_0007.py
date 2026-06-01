from __future__ import annotations

from pathlib import Path

from mmrag.db.connection import connect


def test_migration_0007_adds_content_fts_and_graph_tables(isolated_data_dir: Path) -> None:
    with connect() as conn:
        names = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }

    assert "fts_content_items" in names
    assert "nodes" in names
    assert "edges" in names
