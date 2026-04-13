from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from mmrag import __version__
from mmrag.db.connection import connect
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

app = FastAPI(
    title="mmrag",
    version=__version__,
    description="Edge-optimized multimodal ingestion REST mirror of the MCP surface.",
)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "version": __version__}


@app.post("/ingest", response_model=IngestOutput)
async def ingest_endpoint(inp: IngestInput) -> IngestOutput:
    return await handle_ingest(inp)


@app.post("/ask", response_model=AskOutput)
async def ask_endpoint(inp: AskInput) -> AskOutput:
    return await handle_ask(inp)


@app.post("/search", response_model=SearchOutput)
async def search_endpoint(inp: SearchInput) -> SearchOutput:
    return await handle_search(inp)


@app.get("/jobs/{job_id}", response_model=StatusOutput)
async def status_endpoint(job_id: str) -> StatusOutput:
    try:
        return await handle_status(StatusInput(job_id=job_id))
    except JobNotFound:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}") from None


@app.get("/asset/{asset_id}")
async def asset_endpoint(asset_id: str) -> JSONResponse:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM assets WHERE id = ?",
            (asset_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    out = dict(row)
    if out.get("metadata_json"):
        try:
            out["metadata"] = json.loads(out.pop("metadata_json"))
        except json.JSONDecodeError:
            out["metadata"] = {}
    return JSONResponse(out)
