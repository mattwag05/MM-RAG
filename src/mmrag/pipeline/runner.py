from __future__ import annotations

import json
from pathlib import Path

from mmrag.config import get_settings
from mmrag.db.connection import connect, transaction
from mmrag.logging import get_logger
from mmrag.models.job import M1_STAGE_ORDER, JobStatus, Stage
from mmrag.pipeline.stages.embed import embed
from mmrag.pipeline.stages.fetch import FetchError, fetch
from mmrag.pipeline.stages.frame_sample import frame_sample
from mmrag.pipeline.stages.normalize import NormalizeError, normalize
from mmrag.pipeline.stages.ocr import ocr
from mmrag.pipeline.stages.scene_detect import scene_detect
from mmrag.pipeline.stages.summarize import summarize
from mmrag.pipeline.stages.transcribe import transcribe

log = get_logger("runner")


def _persist_state(job_id: str, state: dict, stage: Stage, progress: float) -> None:
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            UPDATE jobs
               SET pipeline_state_json = ?,
                   stage = ?,
                   progress = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
            """,
            (json.dumps(state), stage.value, progress, job_id),
        )


def _set_status(job_id: str, status: JobStatus) -> None:
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            UPDATE jobs
               SET status = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
            """,
            (status.value, job_id),
        )


def _record_error(job_id: str, kind: str, message: str) -> None:
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            UPDATE jobs
               SET status = 'error',
                   error_kind = ?,
                   error_message = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
            """,
            (kind, message[:2000], job_id),
        )


def _persist_shots(*, asset_id: str, shots: list[dict]) -> None:
    """Upsert shot rows for an asset. Idempotent via UNIQUE(asset_id, shot_idx)."""
    if not shots:
        return
    with connect() as conn, transaction(conn):
        for shot in shots:
            conn.execute(
                """
                INSERT INTO shots (asset_id, shot_idx, start_s, end_s)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(asset_id, shot_idx) DO UPDATE SET
                    start_s = excluded.start_s,
                    end_s = excluded.end_s
                """,
                (
                    asset_id,
                    int(shot["shot_idx"]),
                    float(shot["start_s"]),
                    float(shot["end_s"]),
                ),
            )


def _persist_segments(*, asset_id: str, segments: list[dict]) -> None:
    """Upsert transcript segments + map shot_idx → shot.id for the FK.

    Idempotent via UNIQUE(asset_id, seg_idx). FTS index is kept in sync by
    the triggers on transcript_segments.
    """
    if not segments:
        return
    with connect() as conn, transaction(conn):
        shot_rows = conn.execute(
            "SELECT id, shot_idx FROM shots WHERE asset_id = ?",
            (asset_id,),
        ).fetchall()
        shot_id_by_idx: dict[int, int] = {
            int(r["shot_idx"]): int(r["id"]) for r in shot_rows
        }
        for seg in segments:
            shot_idx = seg.get("shot_idx")
            shot_id = (
                shot_id_by_idx.get(int(shot_idx)) if shot_idx is not None else None
            )
            conn.execute(
                """
                INSERT INTO transcript_segments
                    (asset_id, shot_id, seg_idx, start_s, end_s, text)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, seg_idx) DO UPDATE SET
                    shot_id = excluded.shot_id,
                    start_s = excluded.start_s,
                    end_s = excluded.end_s,
                    text = excluded.text
                """,
                (
                    asset_id,
                    shot_id,
                    int(seg["seg_idx"]),
                    float(seg["start_s"]),
                    float(seg["end_s"]),
                    str(seg["text"]),
                ),
            )


def _upsert_asset(state: dict) -> None:
    """Persist the assets row from accumulated pipeline state."""
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            INSERT INTO assets (
                id, content_hash, source_url, source_kind, title,
                duration_s, fps, width, height,
                mezzanine_path, audio_path, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_hash) DO UPDATE SET
                source_url = COALESCE(excluded.source_url, assets.source_url),
                title = COALESCE(excluded.title, assets.title),
                duration_s = COALESCE(excluded.duration_s, assets.duration_s),
                fps = COALESCE(excluded.fps, assets.fps),
                width = COALESCE(excluded.width, assets.width),
                height = COALESCE(excluded.height, assets.height),
                mezzanine_path = COALESCE(excluded.mezzanine_path, assets.mezzanine_path),
                audio_path = COALESCE(excluded.audio_path, assets.audio_path)
            """,
            (
                state["asset_id"],
                state["content_hash"],
                state.get("source_url"),
                state["source_kind"],
                state.get("title"),
                state.get("duration_s"),
                state.get("fps"),
                state.get("width"),
                state.get("height"),
                state.get("mezzanine_path"),
                state.get("audio_path"),
                json.dumps(state.get("metadata", {})),
            ),
        )
        # If the conflict path was taken, asset_id may not match what we
        # inserted; reconcile so the job points at the canonical row.
        row = conn.execute(
            "SELECT id FROM assets WHERE content_hash = ?",
            (state["content_hash"],),
        ).fetchone()
        if row and row["id"] != state["asset_id"]:
            state["asset_id"] = row["id"]
        conn.execute(
            "UPDATE jobs SET asset_id = ? WHERE id = ?",
            (state["asset_id"], state["__job_id"]),
        )


async def _run_stage(stage: Stage, state: dict, mode: str) -> dict:
    """Dispatch a stage by name. M1 only has fetch+normalize as real stages;
    everything else returns a stub patch."""
    if stage is Stage.FETCH:
        return await fetch(source=state["source"])
    if stage is Stage.NORMALIZE:
        settings = get_settings()
        return await normalize(
            raw_path=state["raw_path"],
            content_hash=state["content_hash"],
            asset_dir=settings.assets_dir / state["content_hash"],
        )
    if stage is Stage.SCENE_DETECT:
        return await scene_detect(mezzanine_path=state.get("mezzanine_path"))
    if stage is Stage.TRANSCRIBE:
        return await transcribe(
            audio_path=state.get("audio_path"),
            shots=state.get("shots", []),
        )
    if stage is Stage.FRAME_SAMPLE:
        return await frame_sample(
            mezzanine_path=state.get("mezzanine_path"), mode=mode
        )
    if stage is Stage.OCR:
        return await ocr(frames=state.get("frames", []))
    if stage is Stage.EMBED:
        return await embed(
            shots=state.get("shots", []),
            frames=state.get("frames", []),
            segments=state.get("segments", []),
        )
    if stage is Stage.SUMMARIZE:
        return await summarize(shots=state.get("shots", []))
    raise ValueError(f"unknown stage: {stage}")


async def run_pipeline(job_id: str) -> None:
    """Execute all stages for a job, persisting state after each.

    Idempotent: if the job already advanced past a stage, skipped stages
    are no-ops on next attempt.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT id, source, mode, status, stage, pipeline_state_json "
            "FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        log.warning("job_missing", job_id=job_id)
        return

    state: dict = json.loads(row["pipeline_state_json"] or "{}")
    state.setdefault("source", row["source"])
    state["__job_id"] = job_id
    completed_stage = Stage(row["stage"]) if row["stage"] else Stage.QUEUED
    mode = row["mode"] or "standard"

    _set_status(job_id, JobStatus.RUNNING)

    try:
        n_stages = len(M1_STAGE_ORDER)
        for idx, stage in enumerate(M1_STAGE_ORDER):
            # Resume past completed stages.
            already_completed = (
                completed_stage != Stage.QUEUED
                and M1_STAGE_ORDER.index(completed_stage) >= idx
                and completed_stage != stage
            )
            if already_completed:
                continue

            log.info("stage.start", job_id=job_id, stage=stage.value)
            patch = await _run_stage(stage, state, mode)
            state.update(patch or {})
            if stage is Stage.NORMALIZE and "content_hash" in state:
                # Persist the asset row as soon as we know the canonical
                # technical metadata, so that GET /asset/{id} works mid-job.
                _upsert_asset(state)
            elif stage is Stage.SCENE_DETECT and state.get("asset_id"):
                _persist_shots(
                    asset_id=state["asset_id"],
                    shots=state.get("shots", []),
                )
            elif stage is Stage.TRANSCRIBE and state.get("asset_id"):
                _persist_segments(
                    asset_id=state["asset_id"],
                    segments=state.get("segments", []),
                )
            progress = (idx + 1) / n_stages
            _persist_state(job_id, _strip_internal(state), stage, progress)
            log.info(
                "stage.done",
                job_id=job_id,
                stage=stage.value,
                progress=progress,
            )

        # After SUMMARIZE we mark the job done.
        _persist_state(job_id, _strip_internal(state), Stage.DONE, 1.0)
        _set_status(job_id, JobStatus.DONE)
        log.info("job.done", job_id=job_id, asset_id=state.get("asset_id"))
    except FetchError as e:
        log.warning("fetch.error", job_id=job_id, kind=e.kind, error=str(e))
        _record_error(job_id, e.kind, str(e))
    except NormalizeError as e:
        log.warning("normalize.error", job_id=job_id, kind=e.kind, error=str(e))
        _record_error(job_id, e.kind, str(e))
    except Exception as e:  # noqa: BLE001 — terminal job error path
        log.exception("pipeline.error", job_id=job_id)
        _record_error(job_id, "unknown", f"{type(e).__name__}: {e}")


def _strip_internal(state: dict) -> dict:
    return {k: v for k, v in state.items() if not k.startswith("__")}
