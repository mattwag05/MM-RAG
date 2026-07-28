"""Run one pipeline job in a child process that exits when the job is done.

The pipeline models (SigLIP, Parakeet, Florence-2) are never given back by
dropping references: measured across 8 VLMs, ``del`` + ``gc.collect()`` +
``torch.mps.empty_cache()`` left between 1.5 GB and 23 GB resident
(``docs/vlm-selection.md``, Table 3). Process exit is the only thing that
returns the memory, so the long-lived hosts — the stdio MCP server and the
worker — never import torch at all. They hand a job id to a short-lived child
and let it die.

Everything the child needs is already in SQLite (source, stage, resume state,
lease), so the job id is the entire hand-off.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from contextlib import suppress

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("pipeline.spawn")

# How long a terminated child gets to release its job lease before we stop
# waiting. Overshooting just delays shutdown; undershooting falls back to the
# stale-lease reclaim, which is the same path a hard kill takes.
_TERMINATE_GRACE_S = 10.0


async def run_job(job_id: str) -> None:
    """Run ``job_id`` to completion in a child process.

    Cancellation terminates the child, which cancels its own pipeline task and
    releases the job lease on the way out.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "mmrag",
        "run-job",
        job_id,
        # The child re-reads its config from the environment, so an in-process
        # override (tests, an embedding host) has to be handed down explicitly
        # or the child would quietly work against the default data dir.
        env={**os.environ, "MMRAG_DATA_DIR": str(get_settings().data_dir)},
        # mmrag logs to stderr, but pipeline subprocesses (yt-dlp, ffmpeg) are
        # chattier. When the parent is the stdio MCP server, anything on stdout
        # is protocol traffic — so the child never gets to write there.
        stdout=subprocess.DEVNULL,
    )
    log.info("spawn.start", job_id=job_id, pid=proc.pid)
    try:
        code = await proc.wait()
    except asyncio.CancelledError:
        log.info("spawn.cancel", job_id=job_id, pid=proc.pid)
        with suppress(ProcessLookupError):
            proc.terminate()
        with suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_GRACE_S)
        raise
    # A non-zero exit is not raised: the child records terminal job errors in
    # the jobs table itself, and callers poll that, not this return value.
    log.info("spawn.done", job_id=job_id, pid=proc.pid, exit_code=code)


async def run_job_in_process(job_id: str) -> None:
    """Child-side entry point: run the pipeline, cancelling on SIGTERM/SIGINT.

    Without the handlers a terminated child dies before ``run_pipeline``'s
    cancel path can release the lease, so the job would sit unreclaimable until
    the stale-lease timeout.
    """
    from mmrag.pipeline.runner import run_pipeline

    task = asyncio.create_task(run_pipeline(job_id))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, task.cancel)
    with suppress(asyncio.CancelledError):
        await task
