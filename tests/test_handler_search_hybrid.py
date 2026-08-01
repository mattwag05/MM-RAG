"""Hybrid RRF retrieval — all four streams; vector mode returns cosine."""

from __future__ import annotations

import struct
import uuid
from pathlib import Path

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
        out = await handle_search(SearchInput(query="weather", mode="fts", asset_id=asset_id))
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
                "INSERT INTO vec_frames(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (frame_id, _pack(target), asset_id),
            )

        from mmrag.handlers import search as search_mod

        async def fake_encode(_q: str) -> list[float]:
            return target

        monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

        out = await handle_search(
            SearchInput(query="anything", mode="vector", asset_id=asset_id, top_k=3)
        )
        assert out.hits
        # Vector mode returns raw SigLIP cosine similarity, NOT an RRF score.
        # cosine(target, target) = 1.0. An RRF score would be 1/61 ≈ 0.016 for
        # a top-1 match, which is far below any of our cosine thresholds.
        assert out.hits[0].score > 0.99
        assert out.hits[0].score > 0.5  # what the M3 acceptance test will threshold
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
            # FTS source: transcript segment.
            conn.execute(
                "INSERT INTO transcript_segments(asset_id, scene_id, seg_idx, start_s, end_s, text) "
                "VALUES (?, ?, 0, 0.0, 2.0, ?)",
                (asset_id, scene_id, "red color bars pattern"),
            )
            # Vec source: frame with a known vector.
            conn.execute(
                "INSERT INTO frames(asset_id, scene_id, frame_idx, t_s, path) "
                "VALUES (?, ?, 0, 1.0, '/tmp/x.jpg')",
                (asset_id, scene_id),
            )
            frame_id = conn.execute(
                "SELECT id FROM frames WHERE asset_id=? AND frame_idx=0",
                (asset_id,),
            ).fetchone()["id"]

        target = [0.0] * 768
        target[0] = 1.0
        with connect() as conn:
            conn.execute(
                "INSERT INTO vec_frames(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (frame_id, _pack(target), asset_id),
            )

        from mmrag.handlers import search as search_mod

        async def fake_encode(_q: str) -> list[float]:
            return target  # unit vector → cosine 1.0 vs the stored frame

        monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

        out = await handle_search(
            SearchInput(query="red color bars", mode="hybrid", asset_id=asset_id, top_k=5)
        )
        assert out.hits
        assert out.hits[0].asset_id == asset_id
        assert out.hits[0].scene_id is not None
        # The one scene should appear exactly once — fused across both streams,
        # not duplicated. With BOTH FTS transcript AND vec_frames matching the
        # same scene, RRF should accumulate 1/(60+1) + 1/(60+1) ≈ 0.033 for it.
        assert len([h for h in out.hits if h.scene_id == str(scene_id)]) == 1
        top = out.hits[0]
        # RRF with both streams matching rank-1 gives 2/61 ≈ 0.033.
        # A single-stream match would give 1/61 ≈ 0.016.
        assert top.score > 0.025, (
            f"hybrid score {top.score} too low — fusion probably not happening"
        )
    finally:
        reset_settings_for_tests(Settings())


async def test_vector_mode_queries_vec_scenes_stream(tmp_path, monkeypatch):
    """A scene whose ONLY match is its vec_scenes embedding is retrievable in
    vector mode.

    Regression for MM-RAG-7l1: vec_scenes is populated at ingest but no
    retrieval stream queried it. It joins vector mode only — as a hybrid RRF
    stream it double-counts vec_frames (scene vectors are mean-pools of frame
    vectors) and measurably regressed eval MRR, so hybrid must NOT see it.
    """
    try:
        _bootstrap(tmp_path)
        asset_id = str(uuid.uuid4())
        with connect() as conn:
            conn.execute(
                "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
                "VALUES (?, ?, 'file', '{}')",
                (asset_id, "h4"),
            )
            conn.execute(
                "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s, summary) "
                "VALUES (?, 0, 0.0, 2.0, 'Scene shows: a red square')",
                (asset_id,),
            )
            scene_id = conn.execute(
                "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
            ).fetchone()["id"]

        target = [0.0] * 768
        target[0] = 1.0
        with connect() as conn:
            conn.execute(
                "INSERT INTO vec_scenes(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (scene_id, _pack(target), asset_id),
            )

        from mmrag.handlers import search as search_mod

        async def fake_encode(_q: str) -> list[float]:
            return target

        monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

        out_v = await handle_search(
            SearchInput(query="red square", mode="vector", asset_id=asset_id, top_k=5)
        )
        assert out_v.hits, "scene with only a vec_scenes embedding was not retrieved"
        assert out_v.hits[0].scene_id == str(scene_id)
        assert out_v.hits[0].source_stream == "vec_scenes"
        # Vector mode returns raw cosine (== 1.0 for identical vectors).
        assert out_v.hits[0].score > 0.99

        # Hybrid must NOT fuse vec_scenes (it double-counts vec_frames — see
        # module docstring). This scene has no other index rows, so hybrid
        # returns nothing for it.
        out_h = await handle_search(
            SearchInput(query="red square", mode="hybrid", asset_id=asset_id, top_k=5)
        )
        assert not [h for h in out_h.hits if h.source_stream == "vec_scenes"]
    finally:
        reset_settings_for_tests(Settings())


async def test_include_frames_returns_frame_paths(tmp_path, monkeypatch):
    """include_frames=True attaches the frame JPEG path so a local agent can
    look at the retrieved moment (MM-RAG-0t2). Off by default."""
    try:
        _bootstrap(tmp_path)
        asset_id = str(uuid.uuid4())
        with connect() as conn:
            conn.execute(
                "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
                "VALUES (?, ?, 'file', '{}')",
                (asset_id, "h5"),
            )
            conn.execute(
                "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) VALUES (?, 0, 0.0, 2.0)",
                (asset_id,),
            )
            scene_id = conn.execute(
                "SELECT id FROM scenes WHERE asset_id=?", (asset_id,)
            ).fetchone()["id"]
            # Transcript match only — the hit itself carries no frame_id, so the
            # scene's first frame must be attached as the representative.
            conn.execute(
                "INSERT INTO transcript_segments(asset_id, scene_id, seg_idx, start_s, end_s, text) "
                "VALUES (?, ?, 0, 0.0, 2.0, 'green mountain landscape')",
                (asset_id, scene_id),
            )
            conn.execute(
                "INSERT INTO frames(asset_id, scene_id, frame_idx, t_s, path) "
                "VALUES (?, ?, 0, 1.0, '/data/frames/rep.jpg')",
                (asset_id, scene_id),
            )

        out = await handle_search(
            SearchInput(
                query="green mountain", mode="hybrid", asset_id=asset_id, include_frames=True
            )
        )
        assert out.hits
        assert out.hits[0].frame_path == "/data/frames/rep.jpg"

        # Off by default — no filesystem paths leak unless asked for.
        out_default = await handle_search(
            SearchInput(query="green mountain", mode="hybrid", asset_id=asset_id)
        )
        assert out_default.hits and out_default.hits[0].frame_path is None
    finally:
        reset_settings_for_tests(Settings())


async def test_asset_scoped_vector_search_filters_inside_knn(tmp_path, monkeypatch):
    """Regression: sqlite-vec k= is global unless asset_id is an aux filter."""
    try:
        _bootstrap(tmp_path)
        asset_a = str(uuid.uuid4())
        asset_b = str(uuid.uuid4())
        with connect() as conn:
            for asset_id, content_hash in ((asset_a, "asset-a"), (asset_b, "asset-b")):
                conn.execute(
                    "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
                    "VALUES (?, ?, 'file', '{}')",
                    (asset_id, content_hash),
                )
                conn.execute(
                    "INSERT INTO scenes(asset_id, scene_idx, start_s, end_s) "
                    "VALUES (?, 0, 0.0, 2.0)",
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

            frame_a = conn.execute("SELECT id FROM frames WHERE asset_id=?", (asset_a,)).fetchone()[
                "id"
            ]
            frame_b = conn.execute("SELECT id FROM frames WHERE asset_id=?", (asset_b,)).fetchone()[
                "id"
            ]

        query = [0.0] * 768
        query[0] = 1.0
        near_other_asset = [0.0] * 768
        near_other_asset[0] = 1.0
        scoped_asset_match = [0.0] * 768
        scoped_asset_match[1] = 1.0

        with connect() as conn:
            conn.execute(
                "INSERT INTO vec_frames(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (frame_a, _pack(scoped_asset_match), asset_a),
            )
            conn.execute(
                "INSERT INTO vec_frames(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (frame_b, _pack(near_other_asset), asset_b),
            )

        from mmrag.handlers import search as search_mod

        async def fake_encode(_q: str) -> list[float]:
            return query

        monkeypatch.setattr(search_mod, "_encode_query_text", fake_encode)

        out = await handle_search(
            SearchInput(query="anything", mode="vector", asset_id=asset_a, top_k=1)
        )
        assert len(out.hits) == 1
        assert out.hits[0].asset_id == asset_a
    finally:
        reset_settings_for_tests(Settings())


@pytest.mark.asyncio
async def test_graph_expansion_reserves_slots_so_it_can_actually_contribute(
    isolated_data_dir: Path, monkeypatch
) -> None:
    """hybrid_graph appended expanded hits AFTER a list already truncated to
    top_k, so a graph hit could never survive when direct retrieval filled the
    result set — i.e. essentially always. Measured before the fix: 10 graph
    candidates per query, 0 in the output, at top_k 10/20/40 (MM-RAG-gje).
    """
    from mmrag.handlers import search as search_mod
    from mmrag.models.mcp_io import SearchHit

    base = [
        SearchHit(asset_id="a", scene_id=str(i), start_s=0.0, end_s=1.0, score=1.0 - i / 100)
        for i in range(10)
    ]
    graph = [
        SearchHit(
            asset_id="a",
            content_item_id=f"g{i}",
            start_s=0.0,
            end_s=1.0,
            score=0.5,
            source_stream="graph",
        )
        for i in range(5)
    ]
    monkeypatch.setattr(search_mod, "expand_search_hits", lambda *a, **k: graph)

    out = search_mod._with_graph_expansion(base, SearchInput(query="q", top_k=10))

    assert len(out) == 10
    n_graph = sum(1 for h in out if h.source_stream == "graph")
    assert n_graph == search_mod._graph_quota(10) == 2


@pytest.mark.asyncio
async def test_graph_expansion_with_nothing_fresh_leaves_the_result_untouched(
    isolated_data_dir: Path, monkeypatch
) -> None:
    """A quota must not cost real hits when the graph has nothing new to add."""
    from mmrag.handlers import search as search_mod
    from mmrag.models.mcp_io import SearchHit

    base = [
        SearchHit(asset_id="a", scene_id=str(i), start_s=0.0, end_s=1.0, score=1.0)
        for i in range(10)
    ]
    # Everything the graph returns is already in the result set.
    monkeypatch.setattr(search_mod, "expand_search_hits", lambda *a, **k: list(base))

    out = search_mod._with_graph_expansion(base, SearchInput(query="q", top_k=10))

    assert [h.scene_id for h in out] == [h.scene_id for h in base]
