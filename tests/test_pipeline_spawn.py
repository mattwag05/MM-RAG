from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.db.connection import connect, transaction
from mmrag.pipeline.spawn import run_job


@pytest.mark.asyncio
async def test_run_job_child_uses_the_parents_data_dir(isolated_data_dir: Path) -> None:
    """The child re-reads Settings from the environment, so an in-process
    override has to be handed down. Without it the child would run against the
    default data dir and this job would still be sitting at 'queued'.

    A missing source is deliberate: it fails in the fetch stage, so the child
    records a terminal error without loading a single model.
    """
    job_id = "job-spawn-datadir"
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO jobs (
                id, source, push_to_sbt, status, stage, progress,
                wait_ms, pipeline_state_json
            )
            VALUES (?, '/nonexistent/spawn-fixture.mp4', 0, 'queued', 'queued', 0.0, 0, '{}')
            """,
            (job_id,),
        )

    await run_job(job_id)

    with connect() as conn:
        row = conn.execute(
            "SELECT status, error_kind, runner_id FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()

    assert row["status"] == "error"
    assert row["error_kind"] is not None
    assert row["runner_id"] is None
