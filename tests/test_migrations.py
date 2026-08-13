from __future__ import annotations

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db import migrations
from mmrag.db.connection import connect


def test_failed_migration_rolls_back_sql_and_tracking_row(tmp_path, monkeypatch) -> None:
    reset_settings_for_tests(Settings(data_dir=tmp_path))
    monkeypatch.setattr(
        migrations,
        "_migration_files",
        lambda: [
            (
                "9999_broken.sql",
                "CREATE TABLE should_rollback (id INTEGER PRIMARY KEY);SELECT missing_function();",
            )
        ],
    )

    try:
        with pytest.raises(Exception, match="missing_function"):
            migrations.apply_migrations()

        with connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'should_rollback'"
            ).fetchone()
            recorded = conn.execute(
                "SELECT name FROM schema_migrations WHERE name = '9999_broken.sql'"
            ).fetchone()

        assert table is None
        assert recorded is None
    finally:
        reset_settings_for_tests(Settings())
