from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Protocol

from mmrag.db.connection import connect


@dataclass(frozen=True)
class VectorHit:
    scene_id: int | None
    score: float
    source: str
    snippet: str | None = None
    frame_id: int | None = None
    # Transcript hits carry their segment's asset and timing so the search
    # handler can attribute them to every scene they overlap, not just the one
    # the stored FK points at (MM-RAG-s0l).
    asset_id: str | None = None
    start_s: float | None = None
    end_s: float | None = None


class VectorBackend(Protocol):
    def frame_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]: ...

    def transcript_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]: ...

    def scene_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]: ...


def _pack(v: list[float]) -> bytes:
    return struct.pack(f"<{len(v)}f", *v)


def _cosine(distance: float) -> float:
    return 1.0 - (distance**2) / 2.0


class SqliteVecBackend:
    def frame_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]:
        sql = """
            SELECT f.scene_id AS scene_id,
                   vf.rowid AS frame_id,
                   vf.distance AS distance
              FROM vec_frames vf
              JOIN frames f ON f.id = vf.rowid
             WHERE {asset_filter}
               {time_filter}
               vf.embedding MATCH ?
               AND k = ?
        """
        asset_filter = "vf.asset_id = ? AND" if asset_id is not None else ""
        time_filter = "f.t_s >= ? AND f.t_s <= ? AND" if time_range is not None else ""
        sql = sql.format(asset_filter=asset_filter, time_filter=time_filter)
        params: list = []
        if asset_id is not None:
            params.append(asset_id)
        if time_range is not None:
            params.extend([time_range[0], time_range[1]])
        params.extend([_pack(qvec), limit])
        with connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            VectorHit(
                scene_id=int(r["scene_id"]),
                score=_cosine(float(r["distance"])),
                frame_id=int(r["frame_id"]),
                source="vec_frames",
            )
            for r in rows
        ]

    def transcript_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]:
        sql = """
            SELECT ts.scene_id AS scene_id,
                   ts.asset_id AS asset_id,
                   ts.start_s  AS start_s,
                   ts.end_s    AS end_s,
                   ts.text     AS text,
                   vt.distance AS distance
              FROM vec_transcript vt
              JOIN transcript_segments ts ON ts.id = vt.rowid
             WHERE {asset_filter}
               {time_filter}
               vt.embedding MATCH ?
               AND k = ?
        """
        asset_filter = "vt.asset_id = ? AND" if asset_id is not None else ""
        time_filter = "ts.end_s >= ? AND ts.start_s <= ? AND" if time_range is not None else ""
        sql = sql.format(asset_filter=asset_filter, time_filter=time_filter)
        params: list = []
        if asset_id is not None:
            params.append(asset_id)
        if time_range is not None:
            params.extend([time_range[0], time_range[1]])
        params.extend([_pack(qvec), limit])
        with connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[VectorHit] = []
        for row in rows:
            # A NULL scene_id is no longer fatal: the handler re-attributes by
            # time overlap, so the segment is still reachable (MM-RAG-s0l).
            text = row["text"] or ""
            out.append(
                VectorHit(
                    scene_id=int(row["scene_id"]) if row["scene_id"] is not None else None,
                    score=_cosine(float(row["distance"])),
                    snippet=text[:80] + ("…" if len(text) > 80 else ""),
                    source="vec_transcript",
                    asset_id=row["asset_id"],
                    start_s=float(row["start_s"]) if row["start_s"] is not None else None,
                    end_s=float(row["end_s"]) if row["end_s"] is not None else None,
                )
            )
        return out

    def scene_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]:
        sql = """
            SELECT s.id      AS scene_id,
                   s.summary AS summary,
                   vs.distance AS distance
              FROM vec_scenes vs
              JOIN scenes s ON s.id = vs.rowid
             WHERE {asset_filter}
               {time_filter}
               vs.embedding MATCH ?
               AND k = ?
        """
        asset_filter = "vs.asset_id = ? AND" if asset_id is not None else ""
        time_filter = "s.end_s >= ? AND s.start_s <= ? AND" if time_range is not None else ""
        sql = sql.format(asset_filter=asset_filter, time_filter=time_filter)
        params: list = []
        if asset_id is not None:
            params.append(asset_id)
        if time_range is not None:
            params.extend([time_range[0], time_range[1]])
        params.extend([_pack(qvec), limit])
        with connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[VectorHit] = []
        for row in rows:
            summary = row["summary"] or ""
            out.append(
                VectorHit(
                    scene_id=int(row["scene_id"]),
                    score=_cosine(float(row["distance"])),
                    snippet=(summary[:80] + ("…" if len(summary) > 80 else "")) or None,
                    source="vec_scenes",
                )
            )
        return out


class QdrantBackend:
    def __init__(self, url: str | None):
        self.url = url

    def frame_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]:
        return []

    def transcript_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]:
        return []

    def scene_hits(
        self,
        qvec: list[float],
        asset_id: str | None,
        time_range: tuple[float, float] | None,
        limit: int,
    ) -> list[VectorHit]:
        return []
