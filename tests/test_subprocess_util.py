from __future__ import annotations

import asyncio
import sys

import pytest

from mmrag.pipeline import subprocess_util


@pytest.mark.asyncio
async def test_subprocess_run_terminates_child_on_cancellation() -> None:
    started = asyncio.Event()
    proc_ref = {}
    real_create = asyncio.create_subprocess_exec

    async def tracked_create(*args, **kwargs):  # noqa: ANN002, ANN003
        proc = await real_create(*args, **kwargs)
        proc_ref["proc"] = proc
        started.set()
        return proc

    original_create = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = tracked_create
    try:
        task = asyncio.create_task(
            subprocess_util.run(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                ],
                timeout_s=120,
                grace_s=1,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        proc = proc_ref["proc"]
        await asyncio.wait_for(proc.wait(), timeout=2.0)
        assert proc.returncode is not None
    finally:
        asyncio.create_subprocess_exec = original_create
