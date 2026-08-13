"""Guards on dependency pins that CI cannot catch by running (MM-RAG-ubd).

Every CI job and the Dockerfile install from uv.lock, so a requirement that
resolves to a broken version stays invisible there: the lock keeps pinning the
good one. The people who hit it are the ones installing the published package,
which is precisely the audience the public release is for. These assertions read
pyproject.toml as text, which is the artifact that ships.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _dependency(name: str) -> str:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    for spec in pyproject["project"]["dependencies"]:
        if spec.split(">")[0].split("=")[0].split("[")[0].strip().lower() == name:
            return spec
    raise AssertionError(f"{name} is not a declared dependency")


def test_mcp_is_capped_below_2() -> None:
    """mcp 2.0 moved FastMCP to mcp.server.mcpserver.MCPServer, so
    src/mmrag/mcp_server.py raises ModuleNotFoundError on import under 2.x and
    the whole server is dead on arrival. Dropping this bound to take a 2.x
    release means porting that module first."""
    assert "<2" in _dependency("mcp"), (
        "mcp must stay capped below 2.x until mcp_server.py is ported to MCPServer"
    )
