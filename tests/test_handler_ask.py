from __future__ import annotations

from mmrag.db.connection import connect
from mmrag.handlers.ask import handle_ask
from mmrag.models.mcp_io import AskInput, SearchHit, SearchOutput


async def test_ask_returns_evidence_only_by_default(monkeypatch, isolated_data_dir):
    from mmrag.handlers import search as search_mod

    with connect() as conn:
        conn.execute(
            "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
            "VALUES ('asset-1', 'ask-summary-hash', 'file', '{}')"
        )
        conn.execute(
            """
            INSERT INTO scenes(id, asset_id, scene_idx, start_s, end_s, summary)
            VALUES (10, 'asset-1', 0, 1.0, 2.0, 'Spoken: hello from the transcript')
            """
        )

    async def fake_search(_inp):
        return SearchOutput(
            hits=[
                SearchHit(
                    asset_id="asset-1",
                    scene_id="10",
                    start_s=1.0,
                    end_s=2.0,
                    score=0.25,
                    snippet="hello from the transcript",
                    source_stream="fts_transcript",
                )
            ]
        )

    monkeypatch.setattr(search_mod, "handle_search", fake_search)

    out = await handle_ask(AskInput(question="what happened?"))

    assert out.answer is None
    assert len(out.evidence) == 1
    assert out.evidence[0].asset_id == "asset-1"
    assert out.evidence[0].source_stream == "fts_transcript"
    assert out.evidence[0].score == 0.25
    assert out.evidence[0].summary == "Spoken: hello from the transcript"
    assert out.evidence[0].transcript_snippet == "hello from the transcript"


async def test_ask_synthesizes_only_when_requested(monkeypatch, isolated_data_dir):
    from mmrag.handlers import ask as ask_mod
    from mmrag.handlers import search as search_mod

    async def fake_search(_inp):
        return SearchOutput(
            hits=[
                SearchHit(
                    asset_id="asset-1",
                    scene_id="10",
                    start_s=1.0,
                    end_s=2.0,
                    score=0.25,
                    snippet="evidence text",
                    source_stream="fts_transcript",
                )
            ]
        )

    async def fake_generate(_inp, evidence):
        assert len(evidence) == 1
        return "The answer from evidence [1]."

    monkeypatch.setattr(search_mod, "handle_search", fake_search)
    monkeypatch.setattr(ask_mod, "_generate_answer", fake_generate)

    out = await handle_ask(AskInput(question="what happened?", synthesize=True))

    assert out.answer == "The answer from evidence [1]."
    assert out.confidence == "medium"
    assert len(out.evidence) == 1


async def test_ask_passes_time_range_into_search(monkeypatch, isolated_data_dir):
    from mmrag.handlers import search as search_mod

    async def fake_search(inp):
        assert inp.time_range == (10.0, 20.0)
        return SearchOutput(
            hits=[
                SearchHit(
                    asset_id="asset-1",
                    scene_id="10",
                    start_s=12.0,
                    end_s=13.0,
                    score=0.25,
                    snippet="evidence text",
                    source_stream="fts_transcript",
                )
            ]
        )

    monkeypatch.setattr(search_mod, "handle_search", fake_search)

    out = await handle_ask(AskInput(question="what happened?", time_range=(10.0, 20.0)))

    assert len(out.evidence) == 1
    assert out.evidence[0].start_s == 12.0
