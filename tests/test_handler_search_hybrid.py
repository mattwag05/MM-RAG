"""Hybrid RRF retrieval — all four streams; vector mode returns cosine."""

from __future__ import annotations

import struct
import uuid

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect
from mmrag.db.migrations import apply_migrations
from mmrag.handlers.search import handle_search
from mmrag.models.mcp_io import SearchInput

pytestmark = pytest.mark.m3_visual


def _pack(v):
    return struct.pack(f"<{len(v)}f", *v)


def _bootstrap(tmp_path):
    reset_settings_for_tests(Settings(data_dir=tmp_path))
    apply_migrations()


async def test_fts_mode_matches_transcript_text(tmp_path):
    try:
        _bootstrap(tmp_path)
        asset_id = str(uuid.uuid4())
        with connect() as conn:
            conn.execute(
                "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
                "VALUES (?, ?, 'file', '{}')",
                (asset_id, "h1"),
            )
            conn.execute(
                "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 2.0)",
                (asset_id,),
            )
            scene_id = conn.execute(
                "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
            ).fetchone()["id"]
            conn.execute(
                "INSERT INTO transcript_segments(asset_id, scene_id, seg_idx, start_s, end_s, text) "
                "VALUES (?, ?, 0, 0.0, 2.0, ?)",
                (asset_id, scene_id, "the weather today is sunny"),
            )
        out = await handle_search(
            SearchInput(query="weather", mode="fts", asset_id=asset_id)
        )
        assert out.hits and out.hits[0].asset_id == asset_id
    finally:
        reset_settings_for_tests(Settings())


async def test_vector_mode_returns_raw_cosine(tmp_path, monkeypatch):
    try:
        _bootstrap(tmp_path)
        asset_id = str(uuid.uuid4())
        with connect() as conn:
            conn.execute(
                "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
                "VALUES (?, ?, 'file', '{}')",
                (asset_id, "h2"),
            )
            conn.execute(
                "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 2.0)",
                (asset_id,),
            )
            scene_id = conn.execute(
                "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
            ).fetchone()["id"]
            conn.execute(
                "INSERT INTO frames(asset_id, scene_id, frame_idx, t_s, path) "
                "VALUES (?, ?, 0, 1.0, '/tmp/x.jpg')",
                (asset_id, scene_id),
            )
            frame_id = conn.execute(
                "SELECT id FROM frames WHERE asset_id=?", (asset_id,)
            ).fetchone()["id"]

        target = [0.0] * 768
        target[0] = 1.0
        with connect() as conn:
            conn.execute(
                "INSERT INTO vec_frames(rowid, embedding) VALUES (?, ?)",
                (frame_id, _pack(target)),
            )

        from mmrag.handlers import search as search_mod

        async def fake_encode(_q: str) -> list[float]:
            return target

        monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

        out = await handle_search(
            SearchInput(query="anything", mode="vector", asset_id=asset_id, top_k=3)
        )
        assert out.hits
        assert out.hits[0].score > 0.99
    finally:
        reset_settings_for_tests(Settings())


async def test_hybrid_mode_fuses_streams(tmp_path, monkeypatch):
    try:
        _bootstrap(tmp_path)
        asset_id = str(uuid.uuid4())
        with connect() as conn:
            conn.execute(
                "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
                "VALUES (?, ?, 'file', '{}')",
                (asset_id, "h3"),
            )
            conn.execute(
                "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 2.0)",
                (asset_id,),
            )
            scene_id = conn.execute(
                "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
            ).fetchone()["id"]
            conn.execute(
                "INSERT INTO transcript_segments(asset_id, scene_id, seg_idx, start_s, end_s, text) "
                "VALUES (?, ?, 0, 0.0, 2.0, ?)",
                (asset_id, scene_id, "red color bars pattern"),
            )

        from mmrag.handlers import search as search_mod

        async def fake_encode(_q: str) -> list[float]:
            return [0.0] * 768

        monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

        out = await handle_search(
            SearchInput(query="red color bars", mode="hybrid", asset_id=asset_id, top_k=5)
        )
        assert out.hits
        assert out.hits[0].asset_id == asset_id
        assert out.hits[0].scene_id is not None
    finally:
        reset_settings_for_tests(Settings())
