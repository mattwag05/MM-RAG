from __future__ import annotations

import asyncio

import pytest

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
