"""The db.connection module must load sqlite-vec on every new connection
when the extra is installed, and fail loudly only at query time (not
import time) when it isn't."""

from __future__ import annotations

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect

pytestmark = pytest.mark.m3_visual


def test_sqlite_vec_extension_loads_and_vec0_is_available(tmp_path):
    # Inject a fresh Settings pointing at tmp_path so we don't need the real data dir.
    reset_settings_for_tests(Settings(data_dir=tmp_path))
    try:
        with connect() as conn:
            # vec0 is a virtual table module registered by sqlite-vec.
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS t_vec USING vec0(embedding float[4])")
            rows = conn.execute("SELECT name FROM sqlite_master WHERE name='t_vec'").fetchall()
            assert len(rows) == 1
    finally:
        reset_settings_for_tests(Settings())  # reset to env defaults
