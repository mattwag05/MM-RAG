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


def _persist_scenes(*, asset_id: str, scenes: list[dict]) -> None:
    """Upsert scene rows for an asset. Idempotent via UNIQUE(asset_id, scene_idx)."""
    if not scenes:
        return
    with connect() as conn, transaction(conn):
        for scene in scenes:
            conn.execute(
                """
                INSERT INTO scenes (asset_id, scene_idx, start_s, end_s)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(asset_id, scene_idx) DO UPDATE SET
                    start_s = excluded.start_s,
                    end_s = excluded.end_s
                """,
                (
                    asset_id,
                    int(scene["scene_idx"]),
                    float(scene["start_s"]),
                    float(scene["end_s"]),
                ),
            )


def _persist_segments(*, asset_id: str, segments: list[dict]) -> None:
    """Upsert transcript segments + map scene_idx → scenes.id for the FK.

    Idempotent via UNIQUE(asset_id, seg_idx). FTS index is kept in sync by
    the triggers on transcript_segments.
    """
    if not segments:
        return
    with connect() as conn, transaction(conn):
        scene_rows = conn.execute(
            "SELECT id, scene_idx FROM scenes WHERE asset_id = ?",
            (asset_id,),
        ).fetchall()
        scene_id_by_idx: dict[int, int] = {
            int(r["scene_idx"]): int(r["id"]) for r in scene_rows
        }
        for seg in segments:
            scene_idx = seg.get("scene_idx")
            scene_id = (
                scene_id_by_idx.get(int(scene_idx)) if scene_idx is not None else None
            )
            conn.execute(
                """
                INSERT INTO transcript_segments
                    (asset_id, scene_id, seg_idx, start_s, end_s, text)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, seg_idx) DO UPDATE SET
                    scene_id = excluded.scene_id,
                    start_s = excluded.start_s,
                    end_s = excluded.end_s,
                    text = excluded.text
                """,
                (
                    asset_id,
                    scene_id,
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
            scenes=state.get("scenes", []),
        )
    if stage is Stage.FRAME_SAMPLE:
        settings = get_settings()
        content_hash = state.get("content_hash")
        if not content_hash:
            log.warning("frame_sample.no_content_hash", job_id=state.get("__job_id"))
            return {"frames": []}
        return await frame_sample(
            mezzanine_path=state.get("mezzanine_path"),
            scenes=state.get("scenes", []),
            assets_dir=settings.assets_dir,
            content_hash=content_hash,
            mode=mode,
        )
    if stage is Stage.OCR:
        return await ocr(frames=state.get("frames", []))
    if stage is Stage.EMBED:
        return await embed(
            scenes=state.get("scenes", []),
            frames=state.get("frames", []),
            segments=state.get("segments", []),
        )
    if stage is Stage.SUMMARIZE:
        return await summarize(scenes=state.get("scenes", []))
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
                _persist_scenes(
                    asset_id=state["asset_id"],
                    scenes=state.get("scenes", []),
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
