from __future__ import annotations

import json
import struct
import uuid

from mmrag.config import get_settings
from mmrag.db.connection import connect, transaction
from mmrag.db.content_items import replace_content_items_for_asset, rewrite_content_items_for_asset
from mmrag.logging import get_logger
from mmrag.models.content_item import ContentItem
from mmrag.models.job import M1_STAGE_ORDER, JobStatus, Stage
from mmrag.pipeline.stages.document import DocumentIngestError, ingest_document
from mmrag.pipeline.stages.embed import embed
from mmrag.pipeline.stages.fetch import FetchError, fetch
from mmrag.pipeline.stages.frame_sample import frame_sample
from mmrag.pipeline.stages.normalize import NormalizeError, normalize
from mmrag.pipeline.stages.ocr import ocr
from mmrag.pipeline.stages.scene_detect import scene_detect
from mmrag.pipeline.stages.summarize import summarize
from mmrag.pipeline.stages.transcribe import transcribe

log = get_logger("runner")

JOB_LEASE_STALE_SECONDS = 3600


def _stale_lease_expr() -> str:
    return f"strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-{JOB_LEASE_STALE_SECONDS} seconds')"


def _claim_job(job_id: str, runner_id: str) -> dict | None:
    """Atomically claim a queued or stale-running job for this runner."""
    stale_expr = _stale_lease_expr()
    with connect() as conn, transaction(conn):
        cur = conn.execute(
            f"""
            UPDATE jobs
               SET status = 'running',
                   runner_id = ?,
                   runner_heartbeat_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                   error_kind = NULL,
                   error_message = NULL,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
               AND (
                    status = 'queued'
                    OR (
                        status = 'running'
                        AND (
                            runner_heartbeat_at IS NULL
                            OR runner_heartbeat_at < {stale_expr}
                        )
                    )
               )
            """,
            (runner_id, job_id),
        )
        if cur.rowcount != 1:
            return None
        row = conn.execute(
            "SELECT id, source, mode, push_to_sbt, status, stage, pipeline_state_json FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        return dict(row) if row is not None else None


def _refresh_lease(job_id: str, runner_id: str) -> None:
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            UPDATE jobs
               SET runner_heartbeat_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
               AND runner_id = ?
               AND status = 'running'
            """,
            (job_id, runner_id),
        )


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
                   error_kind = CASE WHEN ? = 'done' THEN NULL ELSE error_kind END,
                   error_message = CASE WHEN ? = 'done' THEN NULL ELSE error_message END,
                   runner_id = CASE WHEN ? IN ('done', 'error') THEN NULL ELSE runner_id END,
                   runner_heartbeat_at = CASE
                       WHEN ? IN ('done', 'error') THEN NULL
                       ELSE runner_heartbeat_at
                   END,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
            """,
            (status.value, status.value, status.value, status.value, status.value, job_id),
        )


def _record_error(job_id: str, kind: str, message: str) -> None:
    with connect() as conn, transaction(conn):
        conn.execute(
            """
            UPDATE jobs
               SET status = 'error',
                   error_kind = ?,
                   error_message = ?,
                   runner_id = NULL,
                   runner_heartbeat_at = NULL,
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
        scene_id_by_idx: dict[int, int] = {int(r["scene_idx"]): int(r["id"]) for r in scene_rows}
        for seg in segments:
            scene_idx = seg.get("scene_idx")
            scene_id = scene_id_by_idx.get(int(scene_idx)) if scene_idx is not None else None
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


def _pack_vec(v: list[float]) -> bytes:
    # Explicit little-endian — sqlite-vec's vec0 expects LE float32 blobs,
    # and native-order struct packing silently produces wrong vectors on BE.
    return struct.pack(f"<{len(v)}f", *v)


def _persist_frames(
    *,
    asset_id: str,
    scene_id_by_idx: dict[int, int],
    frames: list[dict],
) -> dict[tuple[int, int], int]:
    """Upsert frames and return {(scene_idx, frame_idx): frames.id}."""
    if not frames:
        return {}
    out: dict[tuple[int, int], int] = {}
    with connect() as conn, transaction(conn):
        for frame in frames:
            scene_idx = int(frame["scene_idx"])
            scene_id = scene_id_by_idx.get(scene_idx)
            if scene_id is None:
                continue
            conn.execute(
                """
                INSERT INTO frames
                    (asset_id, scene_id, frame_idx, t_s, path, ocr_text, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id, scene_id, frame_idx) DO UPDATE SET
                    t_s = excluded.t_s,
                    path = excluded.path,
                    ocr_text = excluded.ocr_text,
                    width = excluded.width,
                    height = excluded.height
                """,
                (
                    asset_id,
                    scene_id,
                    int(frame["frame_idx"]),
                    float(frame["t_s"]),
                    str(frame["path"]),
                    frame.get("ocr_text"),
                    int(frame.get("width") or 0) or None,
                    int(frame.get("height") or 0) or None,
                ),
            )
            row = conn.execute(
                "SELECT id FROM frames WHERE asset_id=? AND scene_id=? AND frame_idx=?",
                (asset_id, scene_id, int(frame["frame_idx"])),
            ).fetchone()
            out[(scene_idx, int(frame["frame_idx"]))] = int(row["id"])
    return out


def _rewrite_fts_scenes(*, asset_id: str) -> None:
    """Rebuild every fts_scenes row for this asset's scenes from current OCR text.

    Idempotent: deletes any existing rows for the asset's scenes first, then
    inserts the fresh aggregation keyed on ``rowid = scenes.id``.
    """
    with connect() as conn, transaction(conn):
        scene_rows = conn.execute(
            "SELECT id FROM scenes WHERE asset_id = ?", (asset_id,)
        ).fetchall()
        scene_ids = [int(r["id"]) for r in scene_rows]
        if not scene_ids:
            return
        placeholders = ",".join("?" * len(scene_ids))
        conn.execute(
            f"DELETE FROM fts_scenes WHERE rowid IN ({placeholders})",
            scene_ids,
        )
        for scene_id in scene_ids:
            frame_rows = conn.execute(
                "SELECT ocr_text FROM frames WHERE scene_id = ? "
                "AND ocr_text IS NOT NULL AND ocr_text <> ''",
                (scene_id,),
            ).fetchall()
            text = " ".join(r["ocr_text"] for r in frame_rows).strip()
            if not text:
                continue
            conn.execute(
                "INSERT INTO fts_scenes(rowid, text) VALUES (?, ?)",
                (scene_id, text),
            )


def _persist_scene_summaries(*, asset_id: str, summaries: list[dict]) -> None:
    if not summaries:
        return
    with connect() as conn, transaction(conn):
        for entry in summaries:
            conn.execute(
                """
                UPDATE scenes
                   SET summary = ?
                 WHERE asset_id = ? AND scene_idx = ?
                """,
                (
                    entry.get("summary"),
                    asset_id,
                    int(entry["scene_idx"]),
                ),
            )


def _persist_vectors(
    *,
    asset_id: str,
    frame_id_map: dict[tuple[int, int], int],
    scene_id_by_idx: dict[int, int],
    segment_id_by_idx: dict[int, int],
    frame_vectors: list[dict],
    scene_vectors: list[dict],
    segment_vectors: list[dict],
) -> None:
    with connect() as conn, transaction(conn):
        for entry in frame_vectors:
            key = (int(entry["scene_idx"]), int(entry["frame_idx"]))
            frame_id = frame_id_map.get(key)
            if frame_id is None:
                continue
            conn.execute("DELETE FROM vec_frames WHERE rowid = ?", (frame_id,))
            conn.execute(
                "INSERT INTO vec_frames(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (frame_id, _pack_vec(entry["vector"]), asset_id),
            )
        for entry in scene_vectors:
            scene_id = scene_id_by_idx.get(int(entry["scene_idx"]))
            if scene_id is None:
                continue
            conn.execute("DELETE FROM vec_scenes WHERE rowid = ?", (scene_id,))
            conn.execute(
                "INSERT INTO vec_scenes(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (scene_id, _pack_vec(entry["vector"]), asset_id),
            )
        for entry in segment_vectors:
            seg_id = segment_id_by_idx.get(int(entry["seg_idx"]))
            if seg_id is None:
                continue
            conn.execute("DELETE FROM vec_transcript WHERE rowid = ?", (seg_id,))
            conn.execute(
                "INSERT INTO vec_transcript(rowid, embedding, asset_id) VALUES (?, ?, ?)",
                (seg_id, _pack_vec(entry["vector"]), asset_id),
            )


def _scene_id_by_idx(asset_id: str) -> dict[int, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, scene_idx FROM scenes WHERE asset_id = ?", (asset_id,)
        ).fetchall()
    return {int(r["scene_idx"]): int(r["id"]) for r in rows}


def _segment_id_by_idx(asset_id: str) -> dict[int, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, seg_idx FROM transcript_segments WHERE asset_id = ?",
            (asset_id,),
        ).fetchall()
    return {int(r["seg_idx"]): int(r["id"]) for r in rows}


def _frame_id_map_from_db(asset_id: str) -> dict[tuple[int, int], int]:
    """Recompute {(scene_idx, frame_idx): frames.id} from the DB.

    Used as a resume-from-crash fallback in the EMBED persist branch when
    the in-memory ``__frame_id_map`` stash from FRAME_SAMPLE has been lost
    (e.g. worker restart between FRAME_SAMPLE and EMBED).
    """
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT f.id, s.scene_idx, f.frame_idx
              FROM frames f
              JOIN scenes s ON s.id = f.scene_id
             WHERE f.asset_id = ?
            """,
            (asset_id,),
        ).fetchall()
    return {(int(r["scene_idx"]), int(r["frame_idx"])): int(r["id"]) for r in rows}


def _update_frame_ocr(*, asset_id: str, frames: list[dict]) -> None:
    if not frames:
        return
    with connect() as conn, transaction(conn):
        for frame in frames:
            conn.execute(
                """
                UPDATE frames SET ocr_text = ?
                 WHERE asset_id = ? AND frame_idx = ?
                   AND scene_id = (SELECT id FROM scenes
                                    WHERE asset_id = ? AND scene_idx = ?)
                """,
                (
                    frame.get("ocr_text"),
                    asset_id,
                    int(frame["frame_idx"]),
                    asset_id,
                    int(frame["scene_idx"]),
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


def _sbt_payload(asset_id: str) -> dict | None:
    with connect() as conn:
        asset = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        if asset is None:
            return None
        transcript = conn.execute(
            """
            SELECT group_concat(text, ' ') AS text
              FROM transcript_segments
             WHERE asset_id = ?
             ORDER BY seg_idx
            """,
            (asset_id,),
        ).fetchone()
        summaries = conn.execute(
            """
            SELECT summary
              FROM scenes
             WHERE asset_id = ?
               AND summary IS NOT NULL
               AND summary <> ''
             ORDER BY scene_idx
             LIMIT 8
            """,
            (asset_id,),
        ).fetchall()
        content_text = conn.execute(
            """
            SELECT group_concat(text, ' ') AS text
              FROM content_items
             WHERE asset_id = ?
               AND text IS NOT NULL
               AND text <> ''
             ORDER BY chunk_idx
            """,
            (asset_id,),
        ).fetchone()

    transcript_text = transcript["text"] or content_text["text"] or ""
    summary = " ".join(row["summary"] for row in summaries if row["summary"]).strip()
    if not summary:
        summary = (transcript_text[:500] + ("…" if len(transcript_text) > 500 else "")).strip()
    top_tags = _top_tags(" ".join(part for part in (summary, transcript_text) if part))
    return {
        "url": asset["source_url"] or asset["mezzanine_path"] or asset["metadata_json"],
        "platform": _platform(asset["source_url"]),
        "mmrag_asset_id": asset_id,
        "summary": summary,
        "topTags": top_tags,
        "transcriptText": transcript_text,
    }


def _platform(source_url: str | None) -> str:
    if not source_url:
        return "unknown"
    lowered = source_url.lower()
    if "instagram.com" in lowered:
        return "instagram"
    if "threads.net" in lowered:
        return "threads"
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    return "unknown"


def _top_tags(text: str) -> list[str]:
    import re
    from collections import Counter

    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
        if w.lower() not in {"about", "from", "that", "this", "with", "there", "their"}
    ]
    return [word for word, _count in Counter(words).most_common(8)]


async def _push_to_sbt_if_requested(asset_id: str | None, push_to_sbt: bool) -> None:
    if not asset_id or not push_to_sbt:
        return
    settings = get_settings()
    if not settings.sbt_url:
        log.warning("sbt.skip_no_url", asset_id=asset_id)
        return
    payload = _sbt_payload(asset_id)
    if payload is None:
        return
    from mmrag.sbt_client import SBTClientError, push_to_sbt

    try:
        await push_to_sbt(settings.sbt_url, payload)
    except SBTClientError as e:
        log.warning("sbt.push_failed", asset_id=asset_id, error=str(e))


async def _run_stage(stage: Stage, state: dict, mode: str) -> dict:
    """Dispatch a stage by name. M1 only has fetch+normalize as real stages;
    everything else returns a stub patch."""
    if stage is Stage.FETCH:
        return await fetch(source=state["source"])
    if stage is Stage.NORMALIZE:
        if state.get("is_document"):
            return await ingest_document(raw_path=state["raw_path"], asset_id=state["asset_id"])
        settings = get_settings()
        return await normalize(
            raw_path=state["raw_path"],
            content_hash=state["content_hash"],
            asset_dir=settings.assets_dir / state["content_hash"],
        )
    if stage is Stage.SCENE_DETECT:
        if state.get("is_document"):
            return {"scenes": []}
        return await scene_detect(mezzanine_path=state.get("mezzanine_path"))
    if stage is Stage.TRANSCRIBE:
        if state.get("is_document"):
            return {"segments": []}
        return await transcribe(
            audio_path=state.get("audio_path"),
            scenes=state.get("scenes", []),
        )
    if stage is Stage.FRAME_SAMPLE:
        if state.get("is_document"):
            return {"frames": []}
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
        if state.get("is_document"):
            return {"frames": state.get("frames", [])}
        return await ocr(frames=state.get("frames", []))
    if stage is Stage.EMBED:
        if state.get("is_document"):
            return {"frame_vectors": [], "scene_vectors": [], "segment_vectors": [], "vectors_written": 0}
        return await embed(
            scenes=state.get("scenes", []),
            frames=state.get("frames", []),
            segments=state.get("segments", []),
        )
    if stage is Stage.SUMMARIZE:
        if state.get("is_document"):
            return {"summaries": []}
        return await summarize(
            scenes=state.get("scenes", []),
            segments=state.get("segments", []),
            frames=state.get("frames", []),
        )
    raise ValueError(f"unknown stage: {stage}")


async def run_pipeline(job_id: str) -> None:
    """Execute all stages for a job, persisting state after each.

    Idempotent: if the job already advanced past a stage, skipped stages
    are no-ops on next attempt.
    """
    runner_id = str(uuid.uuid4())
    row = _claim_job(job_id, runner_id)
    if row is None:
        log.info("job_claim_skipped", job_id=job_id)
        return

    state: dict = json.loads(row["pipeline_state_json"] or "{}")
    state.setdefault("source", row["source"])
    state["__job_id"] = job_id
    completed_stage = Stage(row["stage"]) if row["stage"] else Stage.QUEUED
    mode = row["mode"] or "standard"
    push_to_sbt = bool(row["push_to_sbt"])

    try:
        n_stages = len(M1_STAGE_ORDER)
        for idx, stage in enumerate(M1_STAGE_ORDER):
            _refresh_lease(job_id, runner_id)
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
                if state.get("is_document"):
                    items = [ContentItem(**item) for item in state.get("document_items", [])]
                    replace_content_items_for_asset(state["asset_id"], items)
            elif stage is Stage.SCENE_DETECT and state.get("asset_id") and not state.get("is_document"):
                _persist_scenes(
                    asset_id=state["asset_id"],
                    scenes=state.get("scenes", []),
                )
                rewrite_content_items_for_asset(state["asset_id"])
            elif stage is Stage.TRANSCRIBE and state.get("asset_id") and not state.get("is_document"):
                _persist_segments(
                    asset_id=state["asset_id"],
                    segments=state.get("segments", []),
                )
                rewrite_content_items_for_asset(state["asset_id"])
            elif stage is Stage.FRAME_SAMPLE and state.get("asset_id") and not state.get("is_document"):
                scene_id_by_idx = _scene_id_by_idx(state["asset_id"])
                frame_id_map = _persist_frames(
                    asset_id=state["asset_id"],
                    scene_id_by_idx=scene_id_by_idx,
                    frames=state.get("frames", []),
                )
                # Stash the maps on the state dict under internal keys so
                # the EMBED persist step can look them up. Keys are stripped
                # from the JSON by _strip_internal before state is saved.
                state["__frame_id_map"] = {f"{k[0]}:{k[1]}": v for k, v in frame_id_map.items()}
                state["__scene_id_by_idx"] = {str(k): v for k, v in scene_id_by_idx.items()}
                rewrite_content_items_for_asset(state["asset_id"])
            elif stage is Stage.OCR and state.get("asset_id") and not state.get("is_document"):
                _update_frame_ocr(
                    asset_id=state["asset_id"],
                    frames=state.get("frames", []),
                )
                _rewrite_fts_scenes(asset_id=state["asset_id"])
                rewrite_content_items_for_asset(state["asset_id"])
            elif stage is Stage.EMBED and state.get("asset_id") and not state.get("is_document"):
                raw_frame_stash = state.get("__frame_id_map") or {}
                if raw_frame_stash:
                    frame_id_map = {
                        tuple(int(x) for x in k.split(":")): v for k, v in raw_frame_stash.items()
                    }
                else:
                    # Resume-from-crash fallback: stash was lost because
                    # FRAME_SAMPLE committed but the worker restarted before
                    # EMBED ran. Recompute from the frames/scenes tables.
                    frame_id_map = _frame_id_map_from_db(state["asset_id"])
                raw_scene_stash = state.get("__scene_id_by_idx") or {}
                if raw_scene_stash:
                    scene_id_by_idx = {int(k): v for k, v in raw_scene_stash.items()}
                else:
                    scene_id_by_idx = _scene_id_by_idx(state["asset_id"])
                segment_id_by_idx = _segment_id_by_idx(state["asset_id"])
                _persist_vectors(
                    asset_id=state["asset_id"],
                    frame_id_map=frame_id_map,
                    scene_id_by_idx=scene_id_by_idx,
                    segment_id_by_idx=segment_id_by_idx,
                    frame_vectors=state.get("frame_vectors", []),
                    scene_vectors=state.get("scene_vectors", []),
                    segment_vectors=state.get("segment_vectors", []),
                )
            elif stage is Stage.SUMMARIZE and state.get("asset_id") and not state.get("is_document"):
                _persist_scene_summaries(
                    asset_id=state["asset_id"],
                    summaries=state.get("summaries", []),
                )
                rewrite_content_items_for_asset(state["asset_id"])
            progress = (idx + 1) / n_stages
            _persist_state(job_id, _strip_internal(state), stage, progress)
            _refresh_lease(job_id, runner_id)
            log.info(
                "stage.done",
                job_id=job_id,
                stage=stage.value,
                progress=progress,
            )

        # After SUMMARIZE we mark the job done.
        _persist_state(job_id, _strip_internal(state), Stage.DONE, 1.0)
        _set_status(job_id, JobStatus.DONE)
        await _push_to_sbt_if_requested(state.get("asset_id"), push_to_sbt)
        log.info("job.done", job_id=job_id, asset_id=state.get("asset_id"))
    except FetchError as e:
        log.warning("fetch.error", job_id=job_id, kind=e.kind, error=str(e))
        _record_error(job_id, e.kind, str(e))
    except NormalizeError as e:
        log.warning("normalize.error", job_id=job_id, kind=e.kind, error=str(e))
        _record_error(job_id, e.kind, str(e))
    except DocumentIngestError as e:
        log.warning("document.error", job_id=job_id, kind=e.kind, error=str(e))
        _record_error(job_id, e.kind, str(e))
    except Exception as e:  # noqa: BLE001 — terminal job error path
        log.exception("pipeline.error", job_id=job_id)
        _record_error(job_id, "unknown", f"{type(e).__name__}: {e}")


def _strip_internal(state: dict) -> dict:
    return {k: v for k, v in state.items() if not k.startswith("__")}
