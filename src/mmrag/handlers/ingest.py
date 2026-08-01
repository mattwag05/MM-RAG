from __future__ import annotations

import asyncio
import json
import uuid

from mmrag.config import get_settings
from mmrag.db.connection import connect, transaction
from mmrag.logging import get_logger
from mmrag.models.job import JobStatus
from mmrag.models.mcp_io import IngestInput, IngestOutput
from mmrag.pipeline.spawn import run_job

log = get_logger("handler.ingest")

_EXTERNAL_RUNNER_POLL_S = 0.1


def _create_job(inp: IngestInput) -> str:
    job_id = str(uuid.uuid4())
    # The profile rides in pipeline_state_json rather than a jobs column, so
    # adding a profile needs no migration and resume reads it back for free.
    # See pipeline.runner._stage_order.
    state = {} if inp.profile == "full" else {"profile": inp.profile}
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO jobs (
                id, source, push_to_sbt,
                status, stage, progress, wait_ms, pipeline_state_json
            )
            VALUES (?, ?, ?, 'queued', 'queued', 0.0, ?, ?)
            """,
            (
                job_id,
                inp.source,
                int(inp.push_to_sbt),
                inp.wait_ms,
                json.dumps(state),
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


async def run_job_and_wait(job_id: str, wait_ms: int) -> None:
    """Drive ``job_id`` and return once it settles or ``wait_ms`` elapses.

    "Inline" means this request owns the job, not that the pipeline runs in
    this process: it runs in a child that exits, because pipeline models are
    never reclaimed in-process (see mmrag.pipeline.spawn). When inline
    execution is off, the worker owns the job and this only polls.
    """
    if get_settings().ingest_inline:
        pipeline_task = asyncio.create_task(run_job(job_id))
        try:
            # wait_ms == 0 means "fire and forget, return immediately."
            if wait_ms > 0:
                await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=wait_ms / 1000.0)
        except TimeoutError:
            # Don't cancel; let the in-process task keep running.
            pass
        return
    deadline = asyncio.get_running_loop().time() + (wait_ms / 1000.0)
    while asyncio.get_running_loop().time() < deadline:
        job = _read_job(job_id)
        if job is None or job["status"] in {JobStatus.DONE.value, JobStatus.ERROR.value}:
            break
        await asyncio.sleep(_EXTERNAL_RUNNER_POLL_S)


def _read_summary(asset_id: str | None) -> str | None:
    if asset_id is None:
        return None
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT start_s, end_s, summary
              FROM scenes
             WHERE asset_id = ?
               AND summary IS NOT NULL
               AND summary <> ''
             ORDER BY scene_idx
             LIMIT 5
            """,
            (asset_id,),
        ).fetchall()
    if not rows:
        with connect() as conn:
            doc_rows = conn.execute(
                """
                SELECT text
                  FROM content_items
                 WHERE asset_id = ?
                   AND text IS NOT NULL
                   AND text <> ''
                 ORDER BY chunk_idx
                 LIMIT 3
                """,
                (asset_id,),
            ).fetchall()
        if not doc_rows:
            return None
        summary = " | ".join(str(r["text"]) for r in doc_rows)
        return summary[:1200] + ("…" if len(summary) > 1200 else "")
    parts = [f"{float(r['start_s']):.2f}-{float(r['end_s']):.2f}s: {r['summary']}" for r in rows]
    summary = " | ".join(parts)
    return summary[:1200] + ("…" if len(summary) > 1200 else "")


async def handle_ingest(inp: IngestInput) -> IngestOutput:
    """Create a job and optionally run it inline within the wait_ms budget.

    Mac/local dev defaults to inline execution so short ingests can complete in
    one request. Pi/tailnet deployments set MMRAG_INGEST_INLINE=false so the MCP
    service only enqueues while the worker process owns heavy pipeline stages.
    """
    job_id = _create_job(inp)
    log.info(
        "ingest.queued",
        job_id=job_id,
        source=inp.source,
        wait_ms=inp.wait_ms,
        profile=inp.profile,
        ingest_inline=get_settings().ingest_inline,
    )

    await run_job_and_wait(job_id, inp.wait_ms)

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
