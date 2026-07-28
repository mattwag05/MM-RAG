from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect, transaction
from mmrag.handlers import ingest as ingest_mod
from mmrag.handlers.ingest import handle_ingest
from mmrag.models.mcp_io import IngestInput


@pytest.mark.asyncio
async def test_ingest_timeout_does_not_cancel_pipeline_task(
    isolated_data_dir: Path, monkeypatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    cancelled = False

    async def fake_run_job(_job_id: str) -> None:
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            finished.set()

    monkeypatch.setattr(ingest_mod, "run_job", fake_run_job)

    out = await handle_ingest(IngestInput(source="tests/fixtures/sample.mp4", wait_ms=1))

    assert out.status == "in_progress"
    assert started.is_set()
    assert not cancelled
    assert not finished.is_set()

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=1.0)
    assert not cancelled


@pytest.mark.asyncio
async def test_ingest_queue_only_does_not_run_pipeline_inline(
    isolated_data_dir: Path, monkeypatch
) -> None:
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir, ingest_inline=False))
    called = False

    async def fake_run_job(_job_id: str) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(ingest_mod, "run_job", fake_run_job)

    out = await handle_ingest(IngestInput(source="tests/fixtures/sample.mp4", wait_ms=0))

    assert out.status == "in_progress"
    assert out.job_id is not None
    assert not called
    with connect() as conn:
        row = conn.execute("SELECT status, stage FROM jobs WHERE id = ?", (out.job_id,)).fetchone()
    assert row is not None
    assert row["status"] == "queued"
    assert row["stage"] == "queued"


@pytest.mark.asyncio
async def test_ingest_queue_only_waits_for_external_worker_result(
    isolated_data_dir: Path, monkeypatch
) -> None:
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir, ingest_inline=False))
    monkeypatch.setattr(ingest_mod, "_EXTERNAL_RUNNER_POLL_S", 0.01)

    async def complete_queued_job() -> None:
        while True:
            with connect() as conn:
                row = conn.execute("SELECT id FROM jobs WHERE status = 'queued'").fetchone()
            if row is not None:
                with connect() as conn, transaction(conn):
                    conn.execute(
                        """
                        INSERT INTO assets(id, content_hash, source_kind, metadata_json)
                        VALUES ('external-worker-asset', 'external-worker-hash', 'file', '{}')
                        """,
                    )
                    conn.execute(
                        """
                        UPDATE jobs
                           SET status = 'done',
                               stage = 'done',
                               progress = 1.0,
                               asset_id = 'external-worker-asset'
                         WHERE id = ?
                        """,
                        (row["id"],),
                    )
                return
            await asyncio.sleep(0.01)

    worker_task = asyncio.create_task(complete_queued_job())
    try:
        out = await handle_ingest(IngestInput(source="tests/fixtures/sample.mp4", wait_ms=1000))
    finally:
        await asyncio.wait_for(worker_task, timeout=1.0)

    assert out.status == "done"
    assert out.asset_id == "external-worker-asset"
