"""End-to-end M2 integration: ingest a speech clip → DB has scenes + segments."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.db.connection import connect
from mmrag.handlers.ingest import handle_ingest
from mmrag.handlers.search import handle_search
from mmrag.models.mcp_io import IngestInput, SearchInput
from tests.conftest import SAMPLE_MP4


@pytest.mark.asyncio
async def test_ingest_sample_mp4_persists_scenes(isolated_data_dir: Path) -> None:
    """The sine-tone testsrc clip has no visual cuts → one fallback scene
    should land in the scenes table. This exercises the runner hook without
    depending on a TTS tool being available."""
    result = await handle_ingest(
        IngestInput(source=str(SAMPLE_MP4), wait_ms=120000)
    )
    assert result.status == "done", f"expected done, got {result.status}: {result.error}"
    asset_id = result.asset_id
    assert asset_id is not None

    with connect() as conn:
        rows = conn.execute(
            "SELECT scene_idx, start_s, end_s FROM scenes "
            "WHERE asset_id = ? ORDER BY scene_idx",
            (asset_id,),
        ).fetchall()
    assert len(rows) >= 1
    assert rows[0]["scene_idx"] == 0


@pytest.mark.asyncio
async def test_ingest_speech_mp4_persists_transcript_segments(
    isolated_data_dir: Path, speech_mp4: Path
) -> None:
    """Full pipeline on a clip containing real speech should populate
    transcript_segments, and the text should be BM25-searchable via FTS."""
    result = await handle_ingest(
        IngestInput(source=str(speech_mp4), wait_ms=180000)
    )
    assert result.status == "done", f"expected done, got {result.status}: {result.error}"
    asset_id = result.asset_id
    assert asset_id is not None

    with connect() as conn:
        segs = conn.execute(
            "SELECT seg_idx, text, start_s, end_s FROM transcript_segments "
            "WHERE asset_id = ? ORDER BY seg_idx",
            (asset_id,),
        ).fetchall()
    assert len(segs) >= 1

    joined = " ".join(r["text"].lower() for r in segs)
    assert ("test" in joined) or ("fixture" in joined) or ("generation" in joined), (
        f"expected a TTS phrase token in transcript, got: {joined!r}"
    )

    # And FTS should find it.
    with connect() as conn:
        # Pick a token that actually appeared in the transcript to search for.
        token = next(
            (t for t in ("test", "fixture", "generation") if t in joined),
            None,
        )
        assert token is not None
        hit = conn.execute(
            "SELECT ts.text FROM transcript_segments ts "
            "JOIN fts_transcript ON fts_transcript.rowid = ts.id "
            "WHERE ts.asset_id = ? AND fts_transcript MATCH ?",
            (asset_id, token),
        ).fetchone()
    assert hit is not None


@pytest.mark.asyncio
async def test_mcp_ingest_then_search_round_trip(
    isolated_data_dir: Path, speech_mp4: Path
) -> None:
    """Full MCP surface: handle_ingest the speech clip, then handle_search
    through the FTS path and confirm a hit points at the ingested asset."""
    ingest = await handle_ingest(
        IngestInput(source=str(speech_mp4), wait_ms=180000)
    )
    assert ingest.status == "done"
    asset_id = ingest.asset_id
    assert asset_id is not None

    # Read back a token we expect the transcript to contain, to drive the
    # query deterministically regardless of which TTS voice generated the fixture.
    with connect() as conn:
        row = conn.execute(
            "SELECT group_concat(text, ' ') AS joined "
            "FROM transcript_segments WHERE asset_id = ?",
            (asset_id,),
        ).fetchone()
    assert row is not None and row["joined"]
    joined = row["joined"].lower()
    token = next(
        (t for t in ("test", "fixture", "generation", "multimodal") if t in joined),
        None,
    )
    assert token is not None, f"no known phrase token in transcript: {joined!r}"

    result = await handle_search(SearchInput(query=token, mode="fts", asset_id=asset_id))
    assert len(result.hits) >= 1
    hit = result.hits[0]
    assert hit.asset_id == asset_id
    assert hit.end_s > hit.start_s
    assert hit.snippet is not None and token in hit.snippet.lower()
