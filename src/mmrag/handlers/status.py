from __future__ import annotations

from mmrag.db.connection import connect
from mmrag.models.mcp_io import StatusInput, StatusOutput


class JobNotFound(LookupError):
    pass


async def handle_status(inp: StatusInput) -> StatusOutput:
    with connect() as conn:
        row = conn.execute(
            "SELECT status, stage, progress, asset_id, error_kind, error_message "
            "FROM jobs WHERE id = ?",
            (inp.job_id,),
        ).fetchone()
    if row is None:
        raise JobNotFound(inp.job_id)

    error: str | None = None
    if row["error_kind"] or row["error_message"]:
        error = f"{row['error_kind'] or 'error'}: {row['error_message'] or ''}".strip()

    return StatusOutput(
        status=row["status"],
        stage=row["stage"],
        progress=row["progress"] or 0.0,
        asset_id=row["asset_id"],
        error=error,
    )
