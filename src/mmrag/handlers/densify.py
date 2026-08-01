"""Re-sample an already-ingested time range at higher frame density.

Frame stride is fixed at ingest (midpoint per scene, 2s on long scenes), so
once search localizes a moment there is no way to look closer at it. Densify
runs a partial pipeline — FRAME_SAMPLE, OCR, EMBED — over the scenes that
overlap a requested window, at a caller-chosen interval.

It is a re-index, not a request-time model call: the new frames land in
``frames``/``vec_frames`` exactly like ingest-time ones, so ``search`` and
``ask`` pick them up on the next query with no special casing. Evidence-first
stays the default.

The plan (which timestamps, at which ``frame_idx``) is computed here rather
than in the stage because it has to avoid colliding with the frame indices
already in the DB; the stage itself stays DB-free.
"""

from __future__ import annotations

import json
import uuid

from mmrag.db.connection import connect, transaction
from mmrag.handlers.ingest import run_job_and_wait
from mmrag.logging import get_logger
from mmrag.models.job import JobStatus
from mmrag.models.mcp_io import DensifyInput, DensifyOutput

log = get_logger("handler.densify")

# A guard on agent-supplied input, not a tuning knob: a 10-minute window at
# 0.2s would queue 3000 ffmpeg shell-outs plus SigLIP forward passes. Refusing
# with a message the caller can act on beats silently truncating the range.
_MAX_DENSIFY_FRAMES = 200


class DensifyError(Exception):
    """Request cannot be planned — bad asset, empty range, or too many frames."""


def _load_asset(asset_id: str) -> dict:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, content_hash, mezzanine_path, source_url FROM assets WHERE id = ?",
            (asset_id,),
        ).fetchone()
    if row is None:
        raise DensifyError(f"unknown asset_id: {asset_id}")
    if not row["mezzanine_path"]:
        raise DensifyError(f"asset has no video to re-sample: {asset_id}")
    return dict(row)


def _plan(asset_id: str, start_s: float, end_s: float, interval_s: float) -> list[dict]:
    """Build per-scene sample schedules over the requested window.

    Timestamps are placed at bucket midpoints (``lo + interval/2 + k*interval``)
    so a sample never lands on a scene boundary, where the fast ffmpeg seek is
    most likely to return the neighbouring shot.

    A timestamp already covered by a frame within half an interval is dropped,
    which is what makes a repeated densify call on the same range a no-op
    instead of a second copy of every frame.
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.scene_idx, s.start_s, s.end_s,
                   COALESCE(MAX(f.frame_idx), -1) AS max_frame_idx,
                   COALESCE(group_concat(f.t_s), '') AS existing_t_s
              FROM scenes s
              LEFT JOIN frames f ON f.scene_id = s.id
             WHERE s.asset_id = ? AND s.start_s < ? AND s.end_s > ?
             GROUP BY s.id
             ORDER BY s.scene_idx
            """,
            (asset_id, end_s, start_s),
        ).fetchall()

    if not rows:
        raise DensifyError(
            f"no scenes overlap {start_s}-{end_s}s for asset {asset_id} "
            "(check the range against the asset's duration)"
        )

    plan: list[dict] = []
    n_frames = 0
    for row in rows:
        lo = max(float(row["start_s"]), start_s)
        hi = min(float(row["end_s"]), end_s)
        existing = [float(v) for v in str(row["existing_t_s"]).split(",") if v]
        times: list[float] = []
        t = lo + interval_s / 2.0
        while t < hi:
            if not any(abs(t - e) < interval_s / 2.0 for e in existing):
                times.append(round(t, 3))
            t += interval_s
        if not times:
            continue
        n_frames += len(times)
        plan.append(
            {
                "scene_idx": int(row["scene_idx"]),
                "frame_idx_start": int(row["max_frame_idx"]) + 1,
                "times": times,
            }
        )

    if n_frames > _MAX_DENSIFY_FRAMES:
        raise DensifyError(
            f"{n_frames} frames requested, limit is {_MAX_DENSIFY_FRAMES} — "
            "narrow the time_range or raise interval_s"
        )
    return plan


def _count_frames_in_range(asset_id: str, start_s: float, end_s: float) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM frames WHERE asset_id = ? AND t_s >= ? AND t_s <= ?",
            (asset_id, start_s, end_s),
        ).fetchone()
    return int(row["n"])


def _create_job(asset: dict, plan: list[dict], wait_ms: int) -> str:
    job_id = str(uuid.uuid4())
    state = {
        "densify": True,
        "densify_plan": plan,
        "asset_id": asset["id"],
        "content_hash": asset["content_hash"],
        "mezzanine_path": asset["mezzanine_path"],
    }
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO jobs (
                id, asset_id, source, push_to_sbt,
                status, stage, progress, wait_ms, pipeline_state_json
            )
            VALUES (?, ?, ?, 0, 'queued', 'queued', 0.0, ?, ?)
            """,
            (
                job_id,
                asset["id"],
                asset["source_url"] or asset["mezzanine_path"],
                wait_ms,
                json.dumps(state),
            ),
        )
    return job_id


def _read_job(job_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, asset_id, status, error_kind, error_message FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row is not None else None


async def handle_densify(inp: DensifyInput) -> DensifyOutput:
    start_s, end_s = inp.time_range
    try:
        asset = _load_asset(inp.asset_id)
        plan = _plan(inp.asset_id, start_s, end_s, inp.interval_s)
    except DensifyError as e:
        return DensifyOutput(status="error", asset_id=inp.asset_id, error=str(e))

    if not plan:
        log.info("densify.already_dense", asset_id=inp.asset_id, start_s=start_s, end_s=end_s)
        return DensifyOutput(status="done", asset_id=inp.asset_id, frames_added=0)

    before = _count_frames_in_range(inp.asset_id, start_s, end_s)
    job_id = _create_job(asset, plan, inp.wait_ms)
    log.info(
        "densify.queued",
        job_id=job_id,
        asset_id=inp.asset_id,
        start_s=start_s,
        end_s=end_s,
        interval_s=inp.interval_s,
        n_scenes=len(plan),
        n_planned=sum(len(e["times"]) for e in plan),
    )

    await run_job_and_wait(job_id, inp.wait_ms)

    job = _read_job(job_id)
    if job is None:
        return DensifyOutput(
            status="error", asset_id=inp.asset_id, error="job vanished after enqueue"
        )
    if job["status"] == JobStatus.ERROR.value:
        return DensifyOutput(
            status="error",
            asset_id=inp.asset_id,
            job_id=job_id,
            error=f"{job['error_kind']}: {job['error_message']}",
        )

    # Reported after the fact, not from the plan: near-duplicate frames are
    # dropped during sampling, so planned and added differ on static content.
    frames_added = _count_frames_in_range(inp.asset_id, start_s, end_s) - before
    return DensifyOutput(
        status="done" if job["status"] == JobStatus.DONE.value else "in_progress",
        asset_id=inp.asset_id,
        job_id=job_id,
        frames_added=frames_added,
    )
