from __future__ import annotations

import asyncio
import uuid

from mmrag.db.connection import connect, transaction
from mmrag.logging import get_logger
from mmrag.models.job import JobStatus
from mmrag.models.mcp_io import IngestInput, IngestOutput
from mmrag.pipeline.runner import run_pipeline

log = get_logger("handler.ingest")


def _create_job(inp: IngestInput) -> str:
    job_id = str(uuid.uuid4())
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO jobs (
                id, source, mode, push_to_sbt,
                status, stage, progress, wait_ms, pipeline_state_json
            )
            VALUES (?, ?, ?, ?, 'queued', 'queued', 0.0, ?, '{}')
            """,
            (
                job_id,
                inp.source,
                inp.mode,
                int(inp.push_to_sbt),
                inp.wait_ms,
            ),
        )
    return job_id


def _read_job(job_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, asset_id, status, stage, error_kind, error_message FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def _read_summary(asset_id: str | None) -> str | None:
    if asset_id is None:
        return None
    # M1 has no scene summaries yet — that lands in M4. Return None
    # explicitly so the response shape stays correct.
    return None


async def handle_ingest(inp: IngestInput) -> IngestOutput:
    """Sync-if-fast, async-if-slow. Always creates a job, then races the
    pipeline against a wait_ms budget."""
    job_id = _create_job(inp)
    log.info("ingest.queued", job_id=job_id, source=inp.source, wait_ms=inp.wait_ms)

    pipeline_task = asyncio.create_task(run_pipeline(job_id))
    try:
        # wait_ms == 0 means "fire and forget, return immediately."
        if inp.wait_ms > 0:
            await asyncio.wait_for(pipeline_task, timeout=inp.wait_ms / 1000.0)
    except TimeoutError:
        # Don't cancel; let the worker (or in-process task) keep running.
        pass

    job = _read_job(job_id)
    if job is None:
        return IngestOutput(status="error", error="job vanished after enqueue")

    if job["status"] == JobStatus.DONE.value:
        return IngestOutput(
            status="done",
            asset_id=job["asset_id"],
            job_id=job_id,
            summary=_read_summary(job["asset_id"]),
        )
    if job["status"] == JobStatus.ERROR.value:
        return IngestOutput(
            status="error",
            asset_id=job["asset_id"],
            job_id=job_id,
            error=f"{job['error_kind']}: {job['error_message']}",
        )
    return IngestOutput(
        status="in_progress",
        asset_id=job["asset_id"],
        job_id=job_id,
    )
