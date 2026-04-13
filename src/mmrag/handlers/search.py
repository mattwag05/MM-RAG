"""MCP `search` tool handler.

M2 ships FTS5 BM25 transcript search. The `vector` and `hybrid` modes are
still M3 territory and fall through to an empty hit list for now so the
surface stays stable while the implementation catches up.
"""

from __future__ import annotations

from mmrag.db.connection import connect
from mmrag.logging import get_logger
from mmrag.models.mcp_io import SearchHit, SearchInput, SearchOutput

log = get_logger("handler.search")


def _fts_search(
    query: str,
    asset_id: str | None,
    top_k: int,
) -> list[SearchHit]:
    # FTS5 bm25() returns a NEGATIVE score where lower (more negative)
    # means a better match. Negating gives the caller higher-is-better,
    # which matches the SearchHit.score convention used elsewhere.
    sql = """
        SELECT
            ts.asset_id   AS asset_id,
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
    sql += " ORDER BY score DESC LIMIT ?"
    params.append(top_k)

    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:  # noqa: BLE001 — malformed MATCH expression, etc.
            log.warning("fts.query_failed", query=query, error=str(e))
            return []

    return [
        SearchHit(
            asset_id=row["asset_id"],
            scene_id=None,  # Populated once M4 adds scene summaries.
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            score=float(row["score"]),
            snippet=row["snippet"],
        )
        for row in rows
    ]


async def handle_search(inp: SearchInput) -> SearchOutput:
    if inp.mode == "fts":
        return SearchOutput(hits=_fts_search(inp.query, inp.asset_id, inp.top_k))
    # `vector` and `hybrid` arrive in M3 with SigLIP + sqlite-vec. Until
    # then we degrade gracefully to FTS-only so the tool surface works.
    if inp.mode in ("vector", "hybrid"):
        return SearchOutput(hits=_fts_search(inp.query, inp.asset_id, inp.top_k))
    return SearchOutput(hits=[])
