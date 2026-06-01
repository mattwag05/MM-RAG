"""handle_search: FTS5 BM25 over transcript_segments."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.db.connection import connect
from mmrag.handlers.search import handle_search
from mmrag.models.mcp_io import SearchInput
from mmrag.pipeline.runner import _persist_scenes, _persist_segments


def _seed_asset_with_segments(asset_id: str, content_hash: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind) VALUES (?, ?, 'file')",
            (asset_id, content_hash),
        )
    _persist_scenes(
        asset_id=asset_id,
        scenes=[
            {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
            {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
        ],
    )
    _persist_segments(
        asset_id=asset_id,
        segments=[
            {
                "seg_idx": 0,
                "start_s": 0.0,
                "end_s": 1.5,
                "text": "the quick brown fox jumps over the lazy dog",
                "scene_idx": 0,
            },
            {
                "seg_idx": 1,
                "start_s": 1.6,
                "end_s": 3.0,
                "text": "multimodal retrieval is fun",
                "scene_idx": 0,
            },
            {
                "seg_idx": 2,
                "start_s": 3.1,
                "end_s": 4.0,
                "text": "gemma four answers questions",
                "scene_idx": 1,
            },
        ],
    )


@pytest.mark.asyncio
async def test_fts_search_returns_matching_segment(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(SearchInput(query="multimodal", mode="fts"))
    assert len(out.hits) == 1
    hit = out.hits[0]
    assert hit.asset_id == "a1"
    assert hit.start_s == pytest.approx(1.6)
    assert hit.end_s == pytest.approx(3.0)
    assert hit.snippet is not None
    assert "multimodal" in hit.snippet.lower()


@pytest.mark.asyncio
async def test_fts_search_scopes_to_asset_id(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    _seed_asset_with_segments("a2", "h2")
    out = await handle_search(
        SearchInput(query="multimodal", asset_id="a2", mode="fts")
    )
    assert len(out.hits) == 1
    assert out.hits[0].asset_id == "a2"


@pytest.mark.asyncio
async def test_fts_search_respects_top_k(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    # Every segment contains "the" or similar common word — use "is" which
    # only shows up in the middle segment after tokenization.
    out = await handle_search(
        SearchInput(query="multimodal OR fox OR gemma", mode="fts", top_k=2)
    )
    assert len(out.hits) == 2


@pytest.mark.asyncio
async def test_fts_search_no_match_returns_empty(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(SearchInput(query="nonexistent_token_zzz", mode="fts"))
    assert out.hits == []


@pytest.mark.asyncio
async def test_fts_search_ranked_by_bm25_higher_is_better(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(
        SearchInput(query="multimodal OR gemma", mode="fts")
    )
    assert len(out.hits) == 2
    # Scores should be finite, and we return higher-is-better so the sort
    # is descending.
    for h in out.hits:
        assert h.score >= 0
    assert out.hits[0].score >= out.hits[1].score
