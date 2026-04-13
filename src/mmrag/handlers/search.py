from __future__ import annotations

from mmrag.models.mcp_io import SearchInput, SearchOutput

# M1 stub. FTS5 transcript search lands in M2; vector + hybrid in M3.


async def handle_search(inp: SearchInput) -> SearchOutput:
    return SearchOutput(hits=[])
