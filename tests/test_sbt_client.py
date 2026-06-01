from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect, transaction
from mmrag.pipeline import runner


@pytest.mark.asyncio
async def test_push_to_sbt_builds_enrichment_payload(
    isolated_data_dir: Path, monkeypatch
) -> None:
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir, sbt_url="http://sbt.test"))
    asset_id = "asset-sbt"
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO assets(id, content_hash, source_kind, source_url, title, metadata_json)
            VALUES (?, 'hash-sbt', 'url', 'https://youtu.be/example', 'Example', '{}')
            """,
            (asset_id,),
        )
        conn.execute(
            "INSERT INTO scenes(id, asset_id, scene_idx, start_s, end_s, summary) "
            "VALUES (10, ?, 0, 0.0, 1.0, 'Spoken: blazar jets')",
            (asset_id,),
        )
        conn.execute(
            "INSERT INTO transcript_segments(asset_id, scene_id, seg_idx, start_s, end_s, text) "
            "VALUES (?, 10, 0, 0.0, 1.0, 'blazar jets are bright')",
            (asset_id,),
        )

    calls: list[tuple[str, dict]] = []

    async def fake_push_to_sbt(base_url: str, payload: dict) -> dict:
        calls.append((base_url, payload))
        return {"ok": True}

    monkeypatch.setattr("mmrag.sbt_client.push_to_sbt", fake_push_to_sbt)

    await runner._push_to_sbt_if_requested(asset_id, True)

    assert calls
    base_url, payload = calls[0]
    assert base_url == "http://sbt.test"
    assert payload["mmrag_asset_id"] == asset_id
    assert payload["platform"] == "youtube"
    assert "blazar jets" in payload["summary"]
    assert "blazar" in payload["transcriptText"]
    assert "blazar" in payload["topTags"]
