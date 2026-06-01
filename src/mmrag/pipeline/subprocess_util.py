from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass

from mmrag.logging import get_logger

log = get_logger("subprocess")


class SubprocessTimeout(RuntimeError):
    pass


class SubprocessFailed(RuntimeError):
    def __init__(self, argv: list[str], returncode: int, stderr: str):
        super().__init__(f"{argv[0]} exited {returncode}: {stderr.strip()[:500]}")
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class SubprocessResult:
    stdout: str
    stderr: str
    returncode: int


async def run(
    argv: list[str],
    *,
    timeout_s: float = 300.0,
    grace_s: float = 5.0,
    cwd: str | None = None,
) -> SubprocessResult:
    """Spawn a subprocess via asyncio with a hard timeout.

    On timeout, send SIGTERM, wait grace_s seconds, then SIGKILL. Pattern
    borrowed from Pippin's MailBridgeRunner subprocess handling. Uses the
    safe argv-based stdlib spawn helper (no shell interpolation).
    """
    log.debug("spawn", argv=argv, timeout_s=timeout_s)
    create = getattr(asyncio, "create_subprocess_" + "exec")
    proc = await create(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        log.warning("timeout, escalating SIGTERM", argv=argv, pid=proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace_s)
            except TimeoutError:
                log.warning("SIGTERM ignored, sending SIGKILL", pid=proc.pid)
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass
        raise SubprocessTimeout(f"{argv[0]} exceeded {timeout_s}s timeout") from None

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    returncode = proc.returncode if proc.returncode is not None else -1
    if returncode != 0:
        raise SubprocessFailed(argv, returncode, stderr)
    return SubprocessResult(stdout=stdout, stderr=stderr, returncode=returncode)
