from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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

    async def fake_run_pipeline(_job_id: str) -> None:
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            finished.set()

    monkeypatch.setattr(ingest_mod, "run_pipeline", fake_run_pipeline)

    out = await handle_ingest(IngestInput(source="tests/fixtures/sample.mp4", wait_ms=1))

    assert out.status == "in_progress"
    assert started.is_set()
    assert not cancelled
    assert not finished.is_set()

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=1.0)
    assert not cancelled
