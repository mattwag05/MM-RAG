from __future__ import annotations

import asyncio

from mmrag.config import get_settings
from mmrag.db.connection import connect
from mmrag.logging import get_logger
from mmrag.pipeline.runner import JOB_LEASE_STALE_SECONDS, run_pipeline

log = get_logger("worker")

POLL_INTERVAL_S = 1.0


def _claim_pending(limit: int = 16) -> list[str]:
    """Return queued or stale-running job ids to drain on this tick."""
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id
              FROM jobs
             WHERE status = 'queued'
                OR (
                    status = 'running'
                    AND (
                        runner_heartbeat_at IS NULL
                        OR runner_heartbeat_at < strftime(
                            '%Y-%m-%dT%H:%M:%fZ',
                            'now',
                            '-{JOB_LEASE_STALE_SECONDS} seconds'
                        )
                    )
                )
             ORDER BY created_at
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [r["id"] for r in rows]


async def _run_one(job_id: str) -> None:
    try:
        await run_pipeline(job_id)
    except Exception:  # noqa: BLE001 — runner already records errors
        log.exception("worker.job_error", job_id=job_id)


async def run_worker() -> None:
    settings = get_settings()
    concurrency = max(1, int(settings.worker_concurrency))
    log.info("worker.start")
    active: set[asyncio.Task[None]] = set()
    try:
        while True:
            done = {task for task in active if task.done()}
            for task in done:
                active.remove(task)
                await task

            capacity = concurrency - len(active)
            ids = _claim_pending(capacity) if capacity > 0 else []
            if ids:
                log.info("worker.tick", n=len(ids), concurrency=concurrency)
                for job_id in ids:
                    active.add(asyncio.create_task(_run_one(job_id)))

            if not active:
                await asyncio.sleep(POLL_INTERVAL_S)
                continue

            await asyncio.wait(active, timeout=POLL_INTERVAL_S, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        log.info("worker.stop")
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        raise
