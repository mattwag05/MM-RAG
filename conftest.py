"""Repo-root conftest.

Auto-skips tests marked ``m3_visual`` when the optional ``m3-visual`` extra
is not installed, so core-only installs can still run ``pytest`` cleanly.
"""

from __future__ import annotations

import importlib.util

import pytest


def _m3_visual_available() -> bool:
    for mod in ("open_clip", "pytesseract", "PIL", "sqlite_vec", "numpy"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


_HAS_M3 = _m3_visual_available()


def pytest_collection_modifyitems(config, items):
    if _HAS_M3:
        return
    skip_m3 = pytest.mark.skip(reason="m3-visual extra not installed")
    for item in items:
        if "m3_visual" in item.keywords:
            item.add_marker(skip_m3)
