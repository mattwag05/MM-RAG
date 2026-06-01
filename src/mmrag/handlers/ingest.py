from __future__ import annotations

import asyncio
import uuid

from mmrag.config import get_settings
from mmrag.db.connection import connect, transaction
from mmrag.logging import get_logger
from mmrag.models.job import JobStatus
from mmrag.models.mcp_io import IngestInput, IngestOutput
from mmrag.pipeline.runner import run_pipeline

log = get_logger("handler.ingest")

_EXTERNAL_RUNNER_POLL_S = 0.1


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
    ingest_inline = get_settings().ingest_inline
    log.info(
        "ingest.queued",
        job_id=job_id,
        source=inp.source,
        wait_ms=inp.wait_ms,
        ingest_inline=ingest_inline,
    )

    if ingest_inline:
        pipeline_task = asyncio.create_task(run_pipeline(job_id))
        try:
            # wait_ms == 0 means "fire and forget, return immediately."
            if inp.wait_ms > 0:
                await asyncio.wait_for(asyncio.shield(pipeline_task), timeout=inp.wait_ms / 1000.0)
        except TimeoutError:
            # Don't cancel; let the in-process task keep running.
            pass
    elif inp.wait_ms > 0:
        deadline = asyncio.get_running_loop().time() + (inp.wait_ms / 1000.0)
        while asyncio.get_running_loop().time() < deadline:
            job = _read_job(job_id)
            if job is None or job["status"] in {JobStatus.DONE.value, JobStatus.ERROR.value}:
                break
            await asyncio.sleep(_EXTERNAL_RUNNER_POLL_S)

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
