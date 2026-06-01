from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable

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


async def run_worker_until_signalled() -> None:
    """Run the worker and convert process signals into graceful cancellation.

    Docker sends SIGTERM to PID 1 during ``docker stop``. Without an explicit
    handler, Python exits immediately and active job leases remain fresh until
    the stale-lease timeout. Cancelling ``run_worker`` gives active pipelines a
    chance to release their leases before the process exits.
    """
    loop = asyncio.get_running_loop()
    worker_task = asyncio.create_task(run_worker())
    installed: list[signal.Signals] = []
    restored: list[tuple[signal.Signals, signal.Handlers | int | Callable]] = []

    def request_stop(sig: signal.Signals) -> None:
        log.info("worker.signal", signal=sig.name)
        if not worker_task.done():
            worker_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, request_stop, sig)
        except (NotImplementedError, RuntimeError):
            previous = signal.getsignal(sig)

            def handler(_signum, _frame, *, handled_sig=sig) -> None:  # noqa: ANN001
                loop.call_soon_threadsafe(request_stop, handled_sig)

            signal.signal(sig, handler)
            restored.append((sig, previous))
        else:
            installed.append(sig)

    try:
        await worker_task
    except asyncio.CancelledError:
        return
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)
        for sig, previous in restored:
            signal.signal(sig, previous)
