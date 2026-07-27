"""MCP `search` tool handler.

Hybrid retrieval fuses five streams via reciprocal rank fusion (k=60):

  1. FTS5 BM25 over transcript_segments (via fts_transcript)
  2. FTS5 BM25 over aggregated scene OCR (via fts_scenes)
  3. SigLIP text-tower cosine over vec_frames
  4. SigLIP text-tower cosine over vec_transcript
  5. FTS5 BM25 over the content_items projection (via fts_content_items)

Every stream goes through fusion. Nothing is ranked on its raw stream
score, because those scores are not comparable: cosine runs 0.0–1.0 while
negated BM25 runs ~1–20, so any list concatenated onto fused results
(~0.016–0.05) and re-sorted wins the whole result set by scale alone.

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

import re
from dataclasses import dataclass, field

from mmrag.config import get_settings
from mmrag.db.connection import connect
from mmrag.db.graph import expand_search_hits
from mmrag.logging import get_logger
from mmrag.models.mcp_io import SearchHit, SearchInput, SearchOutput
from mmrag.vector_backends import QdrantBackend, SqliteVecBackend, VectorBackend

log = get_logger("handler.search")

_RRF_K = 60
_PER_STREAM_TOP = 20
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_FTS_BINARY_OPS = {"AND", "OR"}


@dataclass
class _StreamHit:
    # None only for content_items rows that carry no scene (document/page
    # items). Every other stream filters those out before constructing a hit.
    scene_id: int | None
    score: float  # cosine for vec streams, -bm25 for fts streams
    snippet: str | None
    # For FTS hits: segment-level timestamps (higher precision).
    # For vec hits: None (will be filled from scenes table).
    start_s: float | None = field(default=None)
    end_s: float | None = field(default=None)
    frame_id: int | None = field(default=None)
    # Source tag — used by FTS dedup to prefer fts_transcript over fts_scenes
    # when both match the same scene_id (fts_transcript has segment timestamps).
    source: str = field(default="")
    # Set only by the content_items stream; scene-less items fuse under their
    # own key and supply their own asset_id/timestamps.
    content_item_id: str | None = field(default=None)
    asset_id: str | None = field(default=None)

    @property
    def key(self) -> tuple[str, str]:
        """Fusion identity. Scene-bearing hits from any stream share a key."""
        if self.scene_id is not None:
            return ("scene", str(self.scene_id))
        return ("item", str(self.content_item_id))


async def _encode_query_text(query: str) -> list[float]:
    """Encode query via the SigLIP text tower. Monkey-patched in tests."""
    import asyncio

    from mmrag.pipeline.stages.embed import _encode_texts_sync

    vecs = await asyncio.to_thread(_encode_texts_sync, [query])
    return vecs[0]


def _vector_backend() -> VectorBackend:
    settings = get_settings()
    if settings.vector_backend == "qdrant":
        return QdrantBackend(settings.qdrant_url)
    return SqliteVecBackend()


def _fts_query(query: str) -> str | None:
    """Convert caller text into a safe FTS5 expression.

    Natural-language questions often contain punctuation that has syntax
    meaning in FTS5. Tokenize to words, quote terms, and preserve explicit
    AND/OR operators when users supply them.
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    if not tokens:
        return None
    out: list[str] = []
    need_op = False
    for token in tokens:
        upper = token.upper()
        if upper in _FTS_BINARY_OPS and need_op:
            out.append(upper)
            need_op = False
            continue
        if need_op:
            out.append("OR")
        out.append(f'"{token}"')
        need_op = True
    while out and out[-1] in _FTS_BINARY_OPS:
        out.pop()
    return " ".join(out) if out else None


def _time_overlap_clause(alias: str) -> str:
    return f" AND {alias}.end_s >= ? AND {alias}.start_s <= ?"


def _append_time_params(params: list, time_range: tuple[float, float] | None) -> None:
    if time_range is None:
        return
    start, end = time_range
    params.extend([start, end])


def _fts_transcript_stream(
    query: str, asset_id: str | None, time_range: tuple[float, float] | None
) -> list[_StreamHit]:
    fts_query = _fts_query(query)
    if fts_query is None:
        return []
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
    params: list = [fts_query]
    if asset_id is not None:
        sql += " AND ts.asset_id = ?"
        params.append(asset_id)
    if time_range is not None:
        sql += _time_overlap_clause("ts")
        _append_time_params(params, time_range)
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


def _fts_scenes_stream(
    query: str, asset_id: str | None, time_range: tuple[float, float] | None
) -> list[_StreamHit]:
    fts_query = _fts_query(query)
    if fts_query is None:
        return []
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
    params: list = [fts_query]
    if asset_id is not None:
        sql += " AND s.asset_id = ?"
        params.append(asset_id)
    if time_range is not None:
        sql += _time_overlap_clause("s")
        _append_time_params(params, time_range)
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


def _vec_frames_stream(
    qvec: list[float], asset_id: str | None, time_range: tuple[float, float] | None
) -> list[_StreamHit]:
    try:
        hits = _vector_backend().frame_hits(qvec, asset_id, time_range, _PER_STREAM_TOP)
    except Exception as e:  # noqa: BLE001
        log.warning("vec_frames.failed", error=str(e))
        return []
    return [
        _StreamHit(
            scene_id=int(hit.scene_id),
            score=hit.score,
            snippet=hit.snippet,
            frame_id=hit.frame_id,
            source=hit.source,
        )
        for hit in hits
        if hit.scene_id is not None
    ]


def _vec_transcript_stream(
    qvec: list[float], asset_id: str | None, time_range: tuple[float, float] | None
) -> list[_StreamHit]:
    try:
        hits = _vector_backend().transcript_hits(qvec, asset_id, time_range, _PER_STREAM_TOP)
    except Exception as e:  # noqa: BLE001
        log.warning("vec_transcript.failed", error=str(e))
        return []
    return [
        _StreamHit(
            scene_id=int(hit.scene_id),
            score=hit.score,
            snippet=hit.snippet,
            source=hit.source,
        )
        for hit in hits
        if hit.scene_id is not None
    ]


def _content_items_hits(
    query: str,
    asset_id: str | None,
    time_range: tuple[float, float] | None,
    top_k: int,
) -> list[SearchHit]:
    fts_query = _fts_query(query)
    if fts_query is None:
        return []
    sql = """
        SELECT f.item_id, f.asset_id, f.item_type,
               -bm25(fts_content_items) AS score,
               snippet(fts_content_items, 3, '', '', '…', 24) AS snippet,
               ci.scene_id, ci.frame_id, ci.start_s, ci.end_s
          FROM fts_content_items f
          JOIN content_items ci ON ci.id = f.item_id
         WHERE fts_content_items MATCH ?
    """
    params: list = [fts_query]
    if asset_id is not None:
        sql += " AND f.asset_id = ?"
        params.append(asset_id)
    if time_range is not None:
        sql += _time_overlap_clause("ci")
        _append_time_params(params, time_range)
    sql += " ORDER BY score DESC LIMIT ?"
    params.append(top_k)
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001
            log.warning("fts_content_items.failed", error=str(e))
            return []
    return [
        SearchHit(
            asset_id=row["asset_id"],
            content_item_id=row["item_id"],
            scene_id=str(row["scene_id"]) if row["scene_id"] is not None else None,
            frame_id=str(row["frame_id"]) if row["frame_id"] is not None else None,
            start_s=float(row["start_s"] or 0.0),
            end_s=float(row["end_s"] if row["end_s"] is not None else row["start_s"] or 0.0),
            score=float(row["score"]),
            snippet=row["snippet"],
            source_stream="content_items",
        )
        for row in rows
    ]


def _content_items_stream(
    query: str, asset_id: str | None, time_range: tuple[float, float] | None
) -> list[_StreamHit]:
    """content_items as a fusion stream (hybrid mode).

    Reuses ``_content_items_hits``' SQL and re-keys it for RRF. In hybrid mode
    these hits must enter fusion rather than being concatenated onto fused
    results: raw ``-bm25`` runs ~1–20 while an RRF score runs ~0.016–0.05, so
    concatenating and re-sorting put every content_items hit above every fused
    hit regardless of relevance.
    """
    return [
        _StreamHit(
            scene_id=int(hit.scene_id) if hit.scene_id is not None else None,
            score=hit.score,
            snippet=hit.snippet,
            start_s=hit.start_s,
            end_s=hit.end_s,
            frame_id=int(hit.frame_id) if hit.frame_id is not None else None,
            source="content_items",
            content_item_id=hit.content_item_id,
            asset_id=hit.asset_id,
        )
        for hit in _content_items_hits(query, asset_id, time_range, _PER_STREAM_TOP)
    ]


def _scene_timing(scene_ids: list[int]) -> dict[int, tuple[str, float, float]]:
    """Return {scene_id: (asset_id, start_s, end_s)} for the given scene IDs."""
    if not scene_ids:
        return {}
    placeholders = ",".join("?" * len(scene_ids))
    sql = f"SELECT id, asset_id, start_s, end_s FROM scenes WHERE id IN ({placeholders})"
    with connect() as conn:
        rows = conn.execute(sql, scene_ids).fetchall()
    return {
        int(r["id"]): (str(r["asset_id"]), float(r["start_s"]), float(r["end_s"])) for r in rows
    }


def _rrf_fuse(
    streams: list[list[_StreamHit]], top_k: int
) -> list[tuple[float, _StreamHit, int | None]]:
    """Return [(fused_score, representative_hit, frame_id), ...] top_k, sorted.

    The representative hit supplies both the snippet and the ``source`` label,
    chosen together: the highest-contribution hit that actually *has* a snippet,
    falling back to the highest-contribution hit overall when none does. They
    must agree because ``handlers/ask.py`` files the snippet under
    ``ocr_snippet`` or ``transcript_snippet`` purely by ``source_stream`` — a
    hit labelled ``vec_frames`` carrying an FTS-transcript snippet gets that
    transcript text reported as on-screen text.

    Everything is compared by RRF contribution, never by raw stream score:
    cosine (0.0–1.0) and negated BM25 (~1–20) are not on the same scale.

    Contributions are **weighted by within-stream normalised score**, and a
    key scores **once per stream** (its best rank). Textbook RRF does neither,
    and both matter here:

    - Rank-only fusion discards how *well* a hit matched. BM25 already ranked
      a verbatim match at 10.31 against 5.39 for a hit sharing only "the" and
      "does", then RRF flattened both to 1/61. Normalising by the stream's own
      top score puts that discrimination back without reintroducing a
      cross-stream scale (every stream's best hit is worth 1.0 of its rank
      contribution).
    - MM-RAG's streams are redundant — content_items is a *projection* of
      scenes/segments/frames — so one scene can match several rows of one
      stream, and un-deduped RRF counted each. That rewarded assets with more
      indexed rows rather than better matches.

    Measured on eval/smoke.jsonl (gold ranks / MRR): rank-only 0.259,
    +dedup 0.444, +dedup+score-weight **1.000**. A stopword list was also
    tried (+dedup+stopwords 0.778) and is deliberately NOT used: score
    weighting subsumes it, and a hardcoded English stoplist would be wrong
    for a plugin that indexes 25 languages.
    """
    fused: dict[tuple[str, str], float] = {}
    best: dict[tuple[str, str], tuple[float, _StreamHit]] = {}
    best_snippet: dict[tuple[str, str], tuple[float, _StreamHit]] = {}
    best_frame: dict[tuple[str, str], tuple[float, int]] = {}
    for hits in streams:
        top_score = max((hit.score for hit in hits), default=0.0)
        scored: set[tuple[str, str]] = set()
        for rank, hit in enumerate(hits):
            contribution = 1.0 / (_RRF_K + rank + 1)
            if top_score > 0:
                # Negative cosine means "not similar" — clamp rather than
                # letting abs() turn it into a strong match.
                contribution *= max(hit.score, 0.0) / top_score
            key = hit.key
            # Best rank per stream only. Later duplicates may still supply a
            # snippet or frame_id below, they just do not score again.
            if key not in scored:
                scored.add(key)
                fused[key] = fused.get(key, 0.0) + contribution
            cur = best.get(key)
            if cur is None or contribution > cur[0]:
                best[key] = (contribution, hit)
            if hit.snippet:
                cur_snip = best_snippet.get(key)
                if cur_snip is None or contribution > cur_snip[0]:
                    best_snippet[key] = (contribution, hit)
            if hit.frame_id is not None:
                cur_frame = best_frame.get(key)
                if cur_frame is None or contribution > cur_frame[0]:
                    best_frame[key] = (contribution, hit.frame_id)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out: list[tuple[float, _StreamHit, int | None]] = []
    for key, score in ordered:
        rep = best_snippet.get(key) or best.get(key)
        if rep is None:  # pragma: no cover — keys only exist because a hit made them
            continue
        frame = best_frame.get(key)
        out.append((score, rep[1], frame[1] if frame else None))
    return out


async def handle_search(inp: SearchInput) -> SearchOutput:
    streams: list[list[_StreamHit]] = []
    base_mode = "hybrid" if inp.mode == "hybrid_graph" else inp.mode
    vector_enabled = get_settings().query_vector_enabled

    if base_mode in ("fts", "hybrid"):
        streams.append(_fts_transcript_stream(inp.query, inp.asset_id, inp.time_range))
        streams.append(_fts_scenes_stream(inp.query, inp.asset_id, inp.time_range))

    if base_mode in ("vector", "hybrid") and vector_enabled:
        try:
            qvec = await _encode_query_text(inp.query)
        except Exception as e:  # noqa: BLE001
            log.warning("query_encode.failed", error=str(e))
            qvec = []
        if qvec:
            streams.append(_vec_frames_stream(qvec, inp.asset_id, inp.time_range))
            streams.append(_vec_transcript_stream(qvec, inp.asset_id, inp.time_range))
    elif base_mode in ("vector", "hybrid"):
        log.info("query_vector.disabled", mode=inp.mode)

    if base_mode == "hybrid":
        # Fifth stream. In fts mode content_items stays a concatenated list
        # below — there both sides are raw BM25, so the scales already agree.
        streams.append(_content_items_stream(inp.query, inp.asset_id, inp.time_range))

    content_hits = (
        _content_items_hits(inp.query, inp.asset_id, inp.time_range, inp.top_k)
        if base_mode == "fts"
        else []
    )

    if base_mode == "fts":
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
        scene_hits = [
            SearchHit(
                asset_id=scene_meta[h.scene_id][0],
                scene_id=str(h.scene_id),
                frame_id=str(h.frame_id) if h.frame_id is not None else None,
                # Use segment-level timestamps when available (fts_transcript),
                # fall back to scene-level (fts_scenes).
                start_s=h.start_s if h.start_s is not None else scene_meta[h.scene_id][1],
                end_s=h.end_s if h.end_s is not None else scene_meta[h.scene_id][2],
                score=h.score,
                snippet=h.snippet,
                source_stream=h.source,
            )
            for h in ordered
            if h.scene_id in scene_meta
        ]
        hits = sorted([*scene_hits, *content_hits], key=lambda h: h.score, reverse=True)[
            : inp.top_k
        ]
        if inp.mode == "hybrid_graph":
            hits = _with_graph_expansion(hits, inp)
        return SearchOutput(hits=hits)

    if base_mode == "vector":
        flat_v: dict[int, _StreamHit] = {}
        for hits in streams:
            for hit in hits:
                cur = flat_v.get(hit.scene_id)
                if cur is None or hit.score > cur.score:
                    flat_v[hit.scene_id] = hit
        ordered_v = sorted(flat_v.values(), key=lambda h: h.score, reverse=True)[: inp.top_k]
        scene_meta = _scene_timing([h.scene_id for h in ordered_v])
        hits = [
            SearchHit(
                asset_id=scene_meta[h.scene_id][0],
                scene_id=str(h.scene_id),
                frame_id=str(h.frame_id) if h.frame_id is not None else None,
                start_s=scene_meta[h.scene_id][1],
                end_s=scene_meta[h.scene_id][2],
                score=h.score,
                snippet=h.snippet or "[visual match]",
                source_stream=h.source,
            )
            for h in ordered_v
            if h.scene_id in scene_meta
        ]
        return SearchOutput(hits=hits)

    # hybrid: RRF fusion over all streams, content_items included. Already
    # sorted by fused score — no re-sort, because there is now only one scale.
    fused = _rrf_fuse(streams, inp.top_k)
    scene_meta = _scene_timing([h.scene_id for _, h, _ in fused if h.scene_id is not None])
    hits = []
    for score, hit, frame_id in fused:
        if hit.scene_id is not None:
            if hit.scene_id not in scene_meta:
                continue
            asset_id, start_s, end_s = scene_meta[hit.scene_id]
        elif hit.asset_id is not None:
            # Scene-less content_items row: it carries its own timing.
            asset_id, start_s, end_s = hit.asset_id, hit.start_s or 0.0, hit.end_s or 0.0
        else:
            continue
        hits.append(
            SearchHit(
                asset_id=asset_id,
                content_item_id=hit.content_item_id,
                scene_id=str(hit.scene_id) if hit.scene_id is not None else None,
                frame_id=str(frame_id) if frame_id is not None else None,
                start_s=start_s,
                end_s=end_s,
                score=score,
                snippet=hit.snippet or "[visual match]",
                source_stream=hit.source,
            )
        )
    hits = hits[: inp.top_k]
    if inp.mode == "hybrid_graph":
        hits = _with_graph_expansion(hits, inp)
    return SearchOutput(hits=hits)


def _with_graph_expansion(hits: list[SearchHit], inp: SearchInput) -> list[SearchHit]:
    if not get_settings().graph_enabled:
        return hits
    expanded = expand_search_hits(
        hits,
        top_k=max(inp.top_k - len(hits), inp.top_k),
        asset_id=inp.asset_id,
        time_range=inp.time_range,
    )
    merged: list[SearchHit] = []
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    for hit in [*hits, *expanded]:
        key = (hit.asset_id, hit.content_item_id, hit.scene_id, hit.frame_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
        if len(merged) >= inp.top_k:
            break
    return merged
