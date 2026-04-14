"""MCP `search` tool handler.

Hybrid retrieval fuses four streams via reciprocal rank fusion (k=60):

  1. FTS5 BM25 over transcript_segments (via fts_transcript)
  2. FTS5 BM25 over aggregated scene OCR (via fts_scenes)
  3. SigLIP text-tower cosine over vec_frames
  4. SigLIP text-tower cosine over vec_transcript

Each stream emits up to 20 candidates keyed on ``scenes.id``. ``hybrid``
mode sums the RRF contributions. ``vector`` mode skips BM25 and returns
raw SigLIP cosine similarity as the score (so callers can threshold
against it — the M3 acceptance test does). ``fts`` mode skips vector
streams.

FTS hits carry segment-level start_s/end_s (the timestamp of the matched
text, not the enclosing scene boundary). This preserves backward
compatibility with tests that check segment-precision timestamps.
Vector/hybrid hits carry scene-level start_s/end_s from scenes table.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from mmrag.db.connection import connect
from mmrag.logging import get_logger
from mmrag.models.mcp_io import SearchHit, SearchInput, SearchOutput

log = get_logger("handler.search")

_RRF_K = 60
_PER_STREAM_TOP = 20


@dataclass
class _StreamHit:
    scene_id: int
    score: float  # cosine for vec streams, -bm25 for fts streams
    snippet: str | None
    # For FTS hits: segment-level timestamps (higher precision).
    # For vec hits: None (will be filled from scenes table).
    start_s: float | None = field(default=None)
    end_s: float | None = field(default=None)
    # Source tag — used by FTS dedup to prefer fts_transcript over fts_scenes
    # when both match the same scene_id (fts_transcript has segment timestamps).
    source: str = field(default="")


async def _encode_query_text(query: str) -> list[float]:
    """Encode query via the SigLIP text tower. Monkey-patched in tests."""
    import asyncio

    from mmrag.pipeline.stages.embed import _encode_texts_sync

    vecs = await asyncio.to_thread(_encode_texts_sync, [query])
    return vecs[0]


def _pack(v: list[float]) -> bytes:
    # Explicit little-endian — sqlite-vec's vec0 expects LE float32 blobs.
    return struct.pack(f"<{len(v)}f", *v)


def _fts_transcript_stream(query: str, asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT ts.scene_id   AS scene_id,
               ts.start_s    AS start_s,
               ts.end_s      AS end_s,
               -bm25(fts_transcript) AS score,
               snippet(fts_transcript, 0, '', '', '…', 24) AS snippet
          FROM fts_transcript
          JOIN transcript_segments ts ON ts.id = fts_transcript.rowid
         WHERE fts_transcript MATCH ?
    """
    params: list = [query]
    if asset_id is not None:
        sql += " AND ts.asset_id = ?"
        params.append(asset_id)
    sql += f" ORDER BY score DESC LIMIT {_PER_STREAM_TOP}"
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("fts_transcript.failed", error=str(e))
            return []
    return [
        _StreamHit(
            scene_id=int(r["scene_id"]),
            score=float(r["score"]),
            snippet=r["snippet"],
            start_s=float(r["start_s"]),
            end_s=float(r["end_s"]),
            source="fts_transcript",
        )
        for r in rows
        if r["scene_id"] is not None
    ]


def _fts_scenes_stream(query: str, asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT s.id      AS scene_id,
               s.start_s AS start_s,
               s.end_s   AS end_s,
               -bm25(fts_scenes) AS score,
               snippet(fts_scenes, 0, '', '', '…', 24) AS snippet
          FROM fts_scenes
          JOIN scenes s ON s.id = fts_scenes.rowid
         WHERE fts_scenes MATCH ?
    """
    params: list = [query]
    if asset_id is not None:
        sql += " AND s.asset_id = ?"
        params.append(asset_id)
    sql += f" ORDER BY score DESC LIMIT {_PER_STREAM_TOP}"
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("fts_scenes.failed", error=str(e))
            return []
    return [
        _StreamHit(
            scene_id=int(r["scene_id"]),
            score=float(r["score"]),
            snippet=r["snippet"],
            start_s=float(r["start_s"]),
            end_s=float(r["end_s"]),
            source="fts_scenes",
        )
        for r in rows
    ]


def _vec_frames_stream(qvec: list[float], asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT f.scene_id AS scene_id,
               vf.distance AS distance
          FROM vec_frames vf
          JOIN frames f ON f.id = vf.rowid
         WHERE vf.embedding MATCH ?
           AND k = ?
    """
    params: list = [_pack(qvec), _PER_STREAM_TOP]
    if asset_id is not None:
        sql += " AND f.asset_id = ?"
        params.append(asset_id)
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("vec_frames.failed", error=str(e))
            return []
    return [
        _StreamHit(
            scene_id=int(r["scene_id"]),
            # sqlite-vec returns squared L2 distance on L2-normalized vecs;
            # for unit vectors, cosine_sim = 1 - distance^2 / 2.
            score=1.0 - (float(r["distance"]) ** 2) / 2.0,
            snippet=None,
            source="vec_frames",
        )
        for r in rows
    ]


def _vec_transcript_stream(qvec: list[float], asset_id: str | None) -> list[_StreamHit]:
    sql = """
        SELECT ts.scene_id AS scene_id,
               ts.text     AS text,
               vt.distance AS distance
          FROM vec_transcript vt
          JOIN transcript_segments ts ON ts.id = vt.rowid
         WHERE vt.embedding MATCH ?
           AND k = ?
    """
    params: list = [_pack(qvec), _PER_STREAM_TOP]
    if asset_id is not None:
        sql += " AND ts.asset_id = ?"
        params.append(asset_id)
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("vec_transcript.failed", error=str(e))
            return []
    out: list[_StreamHit] = []
    for r in rows:
        if r["scene_id"] is None:
            continue
        text = r["text"] or ""
        snippet = text[:80] + ("…" if len(text) > 80 else "")
        out.append(
            _StreamHit(
                scene_id=int(r["scene_id"]),
                score=1.0 - (float(r["distance"]) ** 2) / 2.0,
                snippet=snippet,
                source="vec_transcript",
            )
        )
    return out


def _scene_timing(scene_ids: list[int]) -> dict[int, tuple[str, float, float]]:
    """Return {scene_id: (asset_id, start_s, end_s)} for the given scene IDs."""
    if not scene_ids:
        return {}
    placeholders = ",".join("?" * len(scene_ids))
    sql = (
        f"SELECT id, asset_id, start_s, end_s FROM scenes WHERE id IN ({placeholders})"
    )
    with connect() as conn:
        rows = conn.execute(sql, scene_ids).fetchall()
    return {
        int(r["id"]): (str(r["asset_id"]), float(r["start_s"]), float(r["end_s"]))
        for r in rows
    }


def _rrf_fuse(
    streams: list[list[_StreamHit]], top_k: int
) -> list[tuple[int, float, str | None]]:
    """Return [(scene_id, fused_score, best_snippet), ...] top_k."""
    fused: dict[int, float] = {}
    snippets: dict[int, tuple[float, str | None]] = {}
    for hits in streams:
        for rank, hit in enumerate(hits):
            fused[hit.scene_id] = fused.get(hit.scene_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            cur = snippets.get(hit.scene_id)
            # Snippet is picked by "highest raw score among contributing streams."
            # BM25 scores (negated, so positive) and cosine scores (0.0–1.0) live
            # on different scales; the comparison is apples-to-oranges but is a
            # reasonable heuristic: snippets are advisory output, not the ranking.
            if hit.snippet and (cur is None or hit.score > cur[0]):
                snippets[hit.scene_id] = (hit.score, hit.snippet)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(sid, score, snippets.get(sid, (0.0, None))[1]) for sid, score in ordered]


async def handle_search(inp: SearchInput) -> SearchOutput:
    streams: list[list[_StreamHit]] = []

    if inp.mode in ("fts", "hybrid"):
        streams.append(_fts_transcript_stream(inp.query, inp.asset_id))
        streams.append(_fts_scenes_stream(inp.query, inp.asset_id))

    if inp.mode in ("vector", "hybrid"):
        try:
            qvec = await _encode_query_text(inp.query)
        except Exception as e:  # noqa: BLE001
            log.warning("query_encode.failed", error=str(e))
            qvec = []
        if qvec:
            streams.append(_vec_frames_stream(qvec, inp.asset_id))
            streams.append(_vec_transcript_stream(qvec, inp.asset_id))

    if inp.mode == "fts":
        # FTS-only: return segment-level timestamps for precision.
        # Flatten streams (fts_transcript first, then fts_scenes), dedup by
        # scene_id. When both streams match the same scene, always prefer the
        # fts_transcript hit — it carries segment-level start_s/end_s. The
        # fts_scenes hit has only scene-level boundaries and would lose
        # precision. Score for ranking uses whichever hit wins the source
        # preference (or the higher score when both have the same source).
        flat: dict[int, _StreamHit] = {}
        for hits in streams:
            for hit in hits:
                cur = flat.get(hit.scene_id)
                if cur is None:
                    flat[hit.scene_id] = hit
                    continue
                # Prefer fts_transcript (has segment-level start_s/end_s).
                if cur.source == "fts_transcript" and hit.source != "fts_transcript":
                    continue  # keep the transcript-sourced hit
                if hit.source == "fts_transcript" and cur.source != "fts_transcript":
                    flat[hit.scene_id] = hit  # replace with transcript-sourced
                    continue
                # Same source — keep the higher score.
                if hit.score > cur.score:
                    flat[hit.scene_id] = hit
        ordered = sorted(flat.values(), key=lambda h: h.score, reverse=True)[: inp.top_k]
        scene_meta = _scene_timing([h.scene_id for h in ordered])
        return SearchOutput(
            hits=[
                SearchHit(
                    asset_id=scene_meta[h.scene_id][0],
                    scene_id=str(h.scene_id),
                    # Use segment-level timestamps when available (fts_transcript),
                    # fall back to scene-level (fts_scenes).
                    start_s=h.start_s if h.start_s is not None else scene_meta[h.scene_id][1],
                    end_s=h.end_s if h.end_s is not None else scene_meta[h.scene_id][2],
                    score=h.score,
                    snippet=h.snippet,
                )
                for h in ordered
                if h.scene_id in scene_meta
            ]
        )

    if inp.mode == "vector":
        flat_v: dict[int, _StreamHit] = {}
        for hits in streams:
            for hit in hits:
                cur = flat_v.get(hit.scene_id)
                if cur is None or hit.score > cur.score:
                    flat_v[hit.scene_id] = hit
        ordered_v = sorted(flat_v.values(), key=lambda h: h.score, reverse=True)[: inp.top_k]
        scene_meta = _scene_timing([h.scene_id for h in ordered_v])
        return SearchOutput(
            hits=[
                SearchHit(
                    asset_id=scene_meta[h.scene_id][0],
                    scene_id=str(h.scene_id),
                    start_s=scene_meta[h.scene_id][1],
                    end_s=scene_meta[h.scene_id][2],
                    score=h.score,
                    snippet=h.snippet or "[visual match]",
                )
                for h in ordered_v
                if h.scene_id in scene_meta
            ]
        )

    # hybrid: RRF fusion over all streams
    fused = _rrf_fuse(streams, inp.top_k)
    scene_meta = _scene_timing([sid for sid, _, _ in fused])
    return SearchOutput(
        hits=[
            SearchHit(
                asset_id=scene_meta[sid][0],
                scene_id=str(sid),
                start_s=scene_meta[sid][1],
                end_s=scene_meta[sid][2],
                score=score,
                snippet=snippet or "[visual match]",
            )
            for sid, score, snippet in fused
            if sid in scene_meta
        ]
    )
