from __future__ import annotations

import asyncio

from mmrag.db.connection import connect
from mmrag.logging import get_logger
from mmrag.pipeline.runner import JOB_LEASE_STALE_SECONDS, run_pipeline

log = get_logger("worker")

POLL_INTERVAL_S = 1.0


def _claim_pending() -> list[str]:
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
             LIMIT 16
            """
        ).fetchall()
    return [r["id"] for r in rows]


async def run_worker() -> None:
    log.info("worker.start")
    try:
        while True:
            ids = _claim_pending()
            if not ids:
                await asyncio.sleep(POLL_INTERVAL_S)
                continue
            log.info("worker.tick", n=len(ids))
            for job_id in ids:
                try:
                    await run_pipeline(job_id)
                except Exception:  # noqa: BLE001 — runner already records errors
                    log.exception("worker.job_error", job_id=job_id)
    except asyncio.CancelledError:
        log.info("worker.stop")
        raise
