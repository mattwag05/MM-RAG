from __future__ import annotations

import asyncio
import os
import signal

import pytest

from mmrag import worker as worker_mod
from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.connection import connect, transaction
from mmrag.handlers.status import handle_status
from mmrag.models.job import Stage
from mmrag.models.mcp_io import StatusInput
from mmrag.pipeline import runner
from mmrag.worker import _claim_pending


def _insert_job(
    job_id: str,
    *,
    status: str = "queued",
    stage: str = "queued",
    runner_id: str | None = None,
    runner_heartbeat_at: str | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> None:
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO jobs (
                id, source, mode, push_to_sbt, status, stage, progress,
                wait_ms, pipeline_state_json, runner_id, runner_heartbeat_at,
                error_kind, error_message
            )
            VALUES (?, 'fixture.mp4', 'standard', 0, ?, ?, 0.0, 0, '{}', ?, ?, ?, ?)
            """,
            (
                job_id,
                status,
                stage,
                runner_id,
                runner_heartbeat_at,
                error_kind,
                error_message,
            ),
        )


@pytest.mark.asyncio
async def test_run_pipeline_claims_job_once_for_concurrent_runners(
    isolated_data_dir, monkeypatch
) -> None:
    job_id = "job-concurrent"
    _insert_job(
        job_id,
        error_kind="stale",
        error_message="old failure should be cleared on success",
    )

    calls = 0

    async def fake_run_stage(stage, state, mode):  # noqa: ANN001
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {}

    monkeypatch.setattr(runner, "M1_STAGE_ORDER", (Stage.FETCH,))
    monkeypatch.setattr(runner, "_run_stage", fake_run_stage)

    await asyncio.gather(runner.run_pipeline(job_id), runner.run_pipeline(job_id))

    with connect() as conn:
        row = conn.execute(
            """
            SELECT status, stage, progress, error_kind, error_message,
                   runner_id, runner_heartbeat_at
              FROM jobs
             WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

    assert calls == 1
    assert row["status"] == "done"
    assert row["stage"] == "done"
    assert row["progress"] == 1.0
    assert row["error_kind"] is None
    assert row["error_message"] is None
    assert row["runner_id"] is None
    assert row["runner_heartbeat_at"] is None

    status = await handle_status(StatusInput(job_id=job_id))
    assert status.status == "done"
    assert status.error is None


@pytest.mark.asyncio
async def test_run_pipeline_resume_skips_last_completed_stage(
    isolated_data_dir, monkeypatch
) -> None:
    job_id = "job-resume"
    _insert_job(job_id, stage=Stage.TRANSCRIBE.value)
    calls: list[Stage] = []

    async def fake_run_stage(stage, state, mode):  # noqa: ANN001
        calls.append(stage)
        return {}

    monkeypatch.setattr(
        runner,
        "M1_STAGE_ORDER",
        (Stage.FETCH, Stage.NORMALIZE, Stage.TRANSCRIBE, Stage.FRAME_SAMPLE),
    )
    monkeypatch.setattr(runner, "_run_stage", fake_run_stage)

    await runner.run_pipeline(job_id)

    assert calls == [Stage.FRAME_SAMPLE]


@pytest.mark.asyncio
async def test_run_pipeline_cancel_requeues_active_job(isolated_data_dir, monkeypatch) -> None:
    job_id = "job-cancel"
    _insert_job(job_id)
    started = asyncio.Event()

    async def fake_run_stage(stage, state, mode):  # noqa: ANN001
        started.set()
        await asyncio.Event().wait()
        return {}

    monkeypatch.setattr(runner, "M1_STAGE_ORDER", (Stage.FETCH,))
    monkeypatch.setattr(runner, "_run_stage", fake_run_stage)

    task = asyncio.create_task(runner.run_pipeline(job_id))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    with connect() as conn:
        row = conn.execute(
            "SELECT status, runner_id, runner_heartbeat_at FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

    assert row["status"] == "queued"
    assert row["runner_id"] is None
    assert row["runner_heartbeat_at"] is None


def test_worker_claim_pending_skips_fresh_running_jobs(isolated_data_dir) -> None:
    _insert_job("queued-job")
    _insert_job(
        "fresh-running-job",
        status="running",
        runner_id="active-runner",
        runner_heartbeat_at="9999-01-01T00:00:00.000Z",
    )
    _insert_job(
        "stale-running-job",
        status="running",
        runner_id="dead-runner",
        runner_heartbeat_at="1970-01-01T00:00:00.000Z",
    )

    pending = set(_claim_pending())

    assert "queued-job" in pending
    assert "stale-running-job" in pending
    assert "fresh-running-job" not in pending


@pytest.mark.asyncio
async def test_worker_honors_configured_concurrency(isolated_data_dir, monkeypatch) -> None:
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir, worker_concurrency=2))
    started: list[str] = []
    started_two = asyncio.Event()
    release = asyncio.Event()

    def fake_claim_pending(limit: int = 16) -> list[str]:
        assert limit <= 2
        return [f"job-{idx}" for idx in range(limit)]

    async def fake_run_one(job_id: str) -> None:
        started.append(job_id)
        if len(started) == 2:
            started_two.set()
        await release.wait()

    monkeypatch.setattr(worker_mod, "_claim_pending", fake_claim_pending)
    monkeypatch.setattr(worker_mod, "_run_one", fake_run_one)

    task = asyncio.create_task(worker_mod.run_worker())
    try:
        await asyncio.wait_for(started_two.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert started == ["job-0", "job-1"]
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_signal_handler_cancels_running_worker(monkeypatch) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_run_worker() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(worker_mod, "run_worker", fake_run_worker)

    task = asyncio.create_task(worker_mod.run_worker_until_signalled())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.sleep(0)

    os.kill(os.getpid(), signal.SIGTERM)

    await asyncio.wait_for(task, timeout=1.0)
    assert cancelled.is_set()
