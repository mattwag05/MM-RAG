from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mmrag.handlers.ask import handle_ask
from mmrag.handlers.ingest import handle_ingest
from mmrag.handlers.search import handle_search
from mmrag.handlers.status import JobNotFound, handle_status
from mmrag.logging import configure_logging
from mmrag.models.mcp_io import (
    AskInput,
    AskOutput,
    IngestInput,
    IngestOutput,
    SearchInput,
    SearchOutput,
    StatusInput,
    StatusOutput,
)

configure_logging()

mcp = FastMCP("mmrag")


@mcp.tool()
async def ingest(
    source: str,
    mode: str = "standard",
    wait_ms: int = 30000,
    push_to_sbt: bool = False,
) -> dict:
    """Ingest a public URL or local file. Sync-if-fast (within wait_ms),
    async-if-slow. Returns a job_id you can poll with `status`."""
    inp = IngestInput(source=source, mode=mode, wait_ms=wait_ms, push_to_sbt=push_to_sbt)
    out: IngestOutput = await handle_ingest(inp)
    return out.model_dump()


@mcp.tool()
async def ask(
    question: str,
    asset_id: str | None = None,
    top_k: int = 5,
    model: str = "gemma4:e4b",
) -> dict:
    """Answer a natural-language question about an ingested asset (or the
    whole library) by retrieving top-k evidence and reasoning with Gemma 4.

    M1: returns a placeholder. Real impl lands in M4."""
    inp = AskInput(question=question, asset_id=asset_id, top_k=top_k, model=model)
    out: AskOutput = await handle_ask(inp)
    return out.model_dump()


@mcp.tool()
async def search(
    query: str,
    asset_id: str | None = None,
    top_k: int = 10,
    mode: str = "hybrid",
) -> dict:
    """Search across transcripts, OCR, and scene summaries.

    M1: returns an empty hit list. FTS5 lands in M2; vector + hybrid in M3."""
    inp = SearchInput(query=query, asset_id=asset_id, top_k=top_k, mode=mode)
    out: SearchOutput = await handle_search(inp)
    return out.model_dump()


@mcp.tool()
async def status(job_id: str) -> dict:
    """Poll the status of an ingest job by id."""
    try:
        out: StatusOutput = await handle_status(StatusInput(job_id=job_id))
    except JobNotFound:
        return {
            "status": "error",
            "stage": "queued",
            "progress": 0.0,
            "asset_id": None,
            "error": f"job not found: {job_id}",
        }
    return out.model_dump()


def run_stdio() -> None:
    mcp.run()
