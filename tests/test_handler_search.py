"""handle_search: FTS5 BM25 over transcript_segments."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect
from mmrag.db.content_items import replace_content_items_for_asset
from mmrag.handlers.search import handle_search
from mmrag.models.content_item import ContentItem
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


def _seed_asset_with_content_items(asset_id: str, content_hash: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind) VALUES (?, ?, 'file')",
            (asset_id, content_hash),
        )
    replace_content_items_for_asset(
        asset_id,
        [
            ContentItem(
                id=f"{asset_id}:early",
                type="text",
                source_id=asset_id,
                chunk_idx=0,
                asset_id=asset_id,
                start_s=0.0,
                end_s=1.0,
                text="needle early",
            ),
            ContentItem(
                id=f"{asset_id}:late",
                type="text",
                source_id=asset_id,
                chunk_idx=1,
                asset_id=asset_id,
                start_s=10.0,
                end_s=11.0,
                text="needle late",
            ),
        ],
    )


def _seed_content_items_among_filler(asset_id: str, content_hash: str, n: int = 30) -> None:
    """Seed one matching content item among ``n`` non-matching ones.

    BM25 only produces a meaningful magnitude when the matched term is
    actually rare in the index, so the filler is what makes -bm25 land in its
    real ~1-20 range instead of ~1e-6.
    """
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind) VALUES (?, ?, 'file')",
            (asset_id, content_hash),
        )
    items = [
        ContentItem(
            id=f"{asset_id}:filler{i}",
            type="text",
            source_id=asset_id,
            chunk_idx=i,
            asset_id=asset_id,
            start_s=float(i),
            end_s=float(i) + 1.0,
            text=f"filler passage number {i} about unrelated topics",
        )
        for i in range(n)
    ]
    items.append(
        ContentItem(
            id=f"{asset_id}:needle",
            type="text",
            source_id=asset_id,
            chunk_idx=n,
            asset_id=asset_id,
            start_s=0.0,
            end_s=1.0,
            text="needle early",
        )
    )
    replace_content_items_for_asset(asset_id, items)


@pytest.mark.asyncio
async def test_fts_search_returns_matching_segment(isolated_data_dir: Path) -> None:
    """The matching segment runs 1.6-3.0s across the 2.0s cut, so it belongs to
    BOTH scenes. It used to surface only scene 0 — the scene its start time
    landed in — which is the defect in MM-RAG-s0l."""
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(SearchInput(query="multimodal", mode="fts"))
    assert len(out.hits) == 2
    assert {h.scene_id for h in out.hits} == {"1", "2"}
    for hit in out.hits:
        assert hit.asset_id == "a1"
        # Both carry the segment's own timing, not the scene boundary.
        assert hit.start_s == pytest.approx(1.6)
        assert hit.end_s == pytest.approx(3.0)
        assert hit.snippet is not None
        assert "multimodal" in hit.snippet.lower()


@pytest.mark.asyncio
async def test_fts_search_scopes_to_asset_id(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    _seed_asset_with_segments("a2", "h2")
    out = await handle_search(SearchInput(query="multimodal", asset_id="a2", mode="fts"))
    # Two hits because the segment spans both scenes (see the test above); the
    # point here is that neither of them comes from a1.
    assert len(out.hits) == 2
    assert {h.asset_id for h in out.hits} == {"a2"}


@pytest.mark.asyncio
async def test_fts_search_respects_top_k(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    # Every segment contains "the" or similar common word — use "is" which
    # only shows up in the middle segment after tokenization.
    out = await handle_search(SearchInput(query="multimodal OR fox OR gemma", mode="fts", top_k=2))
    assert len(out.hits) == 2


@pytest.mark.asyncio
async def test_fts_search_no_match_returns_empty(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(SearchInput(query="nonexistent_token_zzz", mode="fts"))
    assert out.hits == []


@pytest.mark.asyncio
async def test_fts_search_handles_natural_language_punctuation(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(SearchInput(query="What is multimodal retrieval?", mode="fts"))
    assert len(out.hits) >= 1
    assert out.hits[0].asset_id == "a1"


@pytest.mark.asyncio
async def test_fts_search_applies_time_range_before_top_k(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(
        SearchInput(query="multimodal OR gemma", mode="fts", top_k=1, time_range=(3.05, 4.0))
    )
    assert len(out.hits) == 1
    assert out.hits[0].start_s == pytest.approx(3.1)
    assert out.hits[0].end_s == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_fts_search_filters_content_items_by_time_range(isolated_data_dir: Path) -> None:
    _seed_asset_with_content_items("content-time", "content-time-hash")

    out = await handle_search(
        SearchInput(query="needle", mode="fts", top_k=5, time_range=(9.0, 12.0))
    )

    assert [hit.content_item_id for hit in out.hits] == ["content-time:late"]


@pytest.mark.asyncio
async def test_hybrid_filters_content_items_by_time_range(
    isolated_data_dir: Path,
) -> None:
    """Kept from the hybrid_graph era (MM-RAG-88j removed that mode): the
    subject here is the time_range filter on the content_items stream, which
    the graph path only ever rode on top of."""
    _seed_asset_with_content_items("hybrid-time", "hybrid-time-hash")

    out = await handle_search(
        SearchInput(query="early", mode="hybrid", top_k=2, time_range=(0.0, 1.0))
    )

    assert [hit.content_item_id for hit in out.hits] == ["hybrid-time:early"]


@pytest.mark.asyncio
async def test_hybrid_search_can_skip_query_vector_encoding(
    isolated_data_dir: Path, monkeypatch
) -> None:
    _seed_asset_with_segments("a1", "h1")
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir, query_vector_enabled=False))
    called = False

    async def fake_encode_query_text(query: str) -> list[float]:
        nonlocal called
        called = True
        return [0.0]

    monkeypatch.setattr("mmrag.handlers.search._encode_query_text", fake_encode_query_text)

    out = await handle_search(SearchInput(query="multimodal", mode="hybrid"))

    assert not called
    assert len(out.hits) == 2  # spanning segment, see the fts test above
    assert {h.asset_id for h in out.hits} == {"a1"}


@pytest.mark.asyncio
async def test_fts_search_ranked_by_bm25_higher_is_better(isolated_data_dir: Path) -> None:
    _seed_asset_with_segments("a1", "h1")
    out = await handle_search(SearchInput(query="multimodal OR gemma", mode="fts"))
    assert len(out.hits) == 2
    # Scores should be finite, and we return higher-is-better so the sort
    # is descending.
    for h in out.hits:
        assert h.score >= 0
    assert out.hits[0].score >= out.hits[1].score


# --- MM-RAG-aux: one ranking space -----------------------------------------
# _rrf_fuse is a pure function, so these lock the two invariants directly
# without standing up a store.


@pytest.mark.asyncio
async def test_hybrid_scores_share_one_scale_across_all_streams(
    isolated_data_dir: Path, monkeypatch
) -> None:
    """Every hybrid hit must be scored on the RRF scale.

    Regression for the defect: content_items hits carried raw -bm25 (~1-20)
    and were concatenated onto fused hits (~0.016-0.05) and re-sorted, so a
    content_items hit outranked every fused hit by scale alone. Asserted at
    the handler level because that concatenation is where the leak lived.

    The filler rows are load-bearing: on a single-document FTS index BM25's
    IDF term collapses and ``-bm25()`` returns ~1e-6, which is *below* the RRF
    scale, so a two-row fixture cannot reproduce the bug at all. With 30
    non-matching rows the one match scores ~4.2 and the gap is 259x.
    """
    _seed_asset_with_segments("scale-seg", "scale-seg-hash")
    _seed_content_items_among_filler("scale-ci", "scale-ci-hash")
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir, query_vector_enabled=False))

    out = await handle_search(
        SearchInput(query="multimodal retrieval needle", mode="hybrid", top_k=10)
    )

    # Both an fts stream and the content_items stream must have contributed,
    # or the assertion below proves nothing.
    assert any(hit.content_item_id for hit in out.hits)
    assert any(hit.scene_id for hit in out.hits)
    for hit in out.hits:
        assert hit.score < 0.1, f"{hit.source_stream} scored {hit.score} — off the RRF scale"


def test_rrf_fuse_labels_a_snippet_with_the_stream_it_came_from() -> None:
    """source must describe the snippet it ships.

    handlers/ask.py files a snippet under ocr_snippet or transcript_snippet
    purely by source_stream, so a hit labelled vec_frames that carries an
    FTS-transcript snippet reports transcript text as on-screen text.
    """
    from mmrag.handlers.search import _rrf_fuse, _StreamHit

    # vec_frames ranks scene 7 first (larger RRF contribution) but has no
    # snippet; fts_transcript ranks it second and carries the actual text.
    vec = [_StreamHit(scene_id=7, score=0.9, snippet=None, frame_id=3, source="vec_frames")]
    transcript = [
        _StreamHit(scene_id=9, score=5.0, snippet="unrelated", source="fts_transcript"),
        _StreamHit(scene_id=7, score=4.0, snippet="spoken words", source="fts_transcript"),
    ]

    fused = _rrf_fuse([vec, transcript], top_k=5)
    _, hit, frame_id = next(entry for entry in fused if entry[1].scene_id == 7)

    assert hit.snippet == "spoken words"
    assert hit.source == "fts_transcript"
    # The frame still comes through — it is identity, not a snippet label.
    assert frame_id == 3


def test_rrf_fuse_scores_a_key_once_per_stream() -> None:
    """A key matching several rows of ONE stream must score once, at its best rank.

    content_items is a projection of scenes/segments/frames, so a single
    scene routinely matches two or three of its rows. Counting each one made
    fusion reward assets with more indexed rows rather than better matches.
    """
    from mmrag.handlers.search import _rrf_fuse, _StreamHit

    # Same scene twice in one stream, at ranks 0 and 1.
    duplicated = [
        _StreamHit(scene_id=7, score=10.0, snippet="first row", source="content_items"),
        _StreamHit(scene_id=7, score=10.0, snippet="second row", source="content_items"),
    ]
    single = [_StreamHit(scene_id=9, score=10.0, snippet="only row", source="fts_transcript")]

    fused = dict(
        (hit.scene_id, score) for score, hit, _ in _rrf_fuse([duplicated, single], top_k=5)
    )

    # Both are rank 0 of one stream at full score, so they must tie. Counting
    # the duplicate would push scene 7 strictly ahead.
    assert fused[7] == pytest.approx(fused[9])


def test_rrf_fuse_weights_contributions_by_match_strength() -> None:
    """Rank alone must not erase how well a hit matched.

    BM25 scored a verbatim match at 10.31 against 5.39 for a hit sharing only
    "the" and "does"; plain RRF flattened both to ~1/61 and the verbatim match
    lost on breadth. Contributions are normalised against the stream's own top
    score so that discrimination survives fusion.
    """
    from mmrag.handlers.search import _rrf_fuse, _StreamHit

    stream = [
        _StreamHit(scene_id=1, score=10.0, snippet="verbatim match", source="fts_transcript"),
        _StreamHit(scene_id=2, score=1.0, snippet="stopword coincidence", source="fts_transcript"),
    ]

    fused = dict((hit.scene_id, score) for score, hit, _ in _rrf_fuse([stream], top_k=5))

    # Adjacent ranks differ by only ~1.6% on rank alone; the 10x score gap
    # must dominate that.
    assert fused[1] > 5 * fused[2]


@pytest.mark.asyncio
async def test_segment_spanning_a_cut_reaches_every_scene_it_covers(
    isolated_data_dir: Path,
) -> None:
    """transcript_segments.scene_id is a single FK assigned from the segment's
    START time, so a segment crossing a cut used to be filed under the first
    scene only and the rest were unreachable by transcript search (MM-RAG-s0l).

    Measured on the reference asset: 82% of segments span >1 scene (median
    segment 4.4s vs median scene 2.5s), and the stored FK reached 51 of 89
    scenes while speech overlapped all 89.
    """
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind) VALUES ('sp', 'hsp', 'file')"
        )
    _persist_scenes(
        asset_id="sp",
        scenes=[
            {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
            {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
            {"scene_idx": 2, "start_s": 4.0, "end_s": 6.0},
            {"scene_idx": 3, "start_s": 6.0, "end_s": 8.0},
        ],
    )
    # One segment of continuous speech straddling three of the four scenes.
    _persist_segments(
        asset_id="sp",
        segments=[
            {
                "seg_idx": 0,
                "start_s": 1.0,
                "end_s": 5.0,
                "text": "kangaroo telemetry across the boundary",
                "scene_idx": 0,
            }
        ],
    )

    out = await handle_search(SearchInput(query="kangaroo", asset_id="sp", mode="fts"))

    with connect() as conn:
        by_idx = {
            int(r["scene_idx"]): str(r["id"])
            for r in conn.execute(
                "SELECT id, scene_idx FROM scenes WHERE asset_id = 'sp'"
            ).fetchall()
        }
    # Scenes 0, 1 and 2 overlap 1.0-5.0s; scene 3 (6.0-8.0s) must NOT match.
    assert {h.scene_id for h in out.hits} == {by_idx[0], by_idx[1], by_idx[2]}


@pytest.mark.asyncio
async def test_fanning_out_does_not_reorder_the_stream(isolated_data_dir: Path) -> None:
    """All scenes of one segment share that segment's rank, so a segment that
    spans many scenes cannot demote the segments ranked after it."""
    from mmrag.handlers.search import _rrf_fuse, _StreamHit

    spanning = [
        _StreamHit(scene_id=10, score=9.0, snippet="best", source="fts_transcript", rank_hint=0),
        _StreamHit(scene_id=11, score=9.0, snippet="best", source="fts_transcript", rank_hint=0),
        _StreamHit(scene_id=12, score=9.0, snippet="best", source="fts_transcript", rank_hint=0),
        # Positionally 4th, but genuinely the 2nd-best segment.
        _StreamHit(scene_id=20, score=3.0, snippet="next", source="fts_transcript", rank_hint=1),
    ]
    fused = dict((hit.scene_id, score) for score, hit, _ in _rrf_fuse([spanning], top_k=10))

    # Runner-up scores as rank 1 (1/62 * 3/9), not as rank 3 (1/64 * 3/9).
    assert fused[20] == pytest.approx((1.0 / 62) * (3.0 / 9.0))
    # Every scene of the top segment scores identically at rank 0.
    assert fused[10] == fused[11] == fused[12] == pytest.approx(1.0 / 61)
