"""Ingest-time knowledge graph over content_items.

Write-only as of MM-RAG-88j: the retrieval half (``expand_search_hits``, the
``hybrid_graph`` mode) was removed because it cost precision, and the cause was
the quality of these nodes rather than the fusion code — ``_topic_terms`` is a
regex, so stopwords and OCR misreads become first-class entities. Building is
off by default now; MM-RAG-gje is the plan to replace the regex with real
entity extraction, at which point a measured retrieval path can come back.
"""

from __future__ import annotations

import hashlib
import json
import re

from mmrag.db.connection import connect, transaction

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "from",
    "have",
    "into",
    "that",
    "their",
    "there",
    "this",
    "with",
    "would",
}


def _edge_id(source: str, target: str, edge_type: str) -> str:
    h = hashlib.sha1(f"{source}\0{target}\0{edge_type}".encode()).hexdigest()
    return f"edge:{h}"


def _insert_node(
    conn, node_id: str, node_type: str, asset_id: str | None, label: str, metadata=None
) -> None:
    conn.execute(
        """
        INSERT INTO nodes(id, node_type, asset_id, label, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            node_type = excluded.node_type,
            asset_id = excluded.asset_id,
            label = excluded.label,
            metadata_json = excluded.metadata_json
        """,
        (node_id, node_type, asset_id, label, json.dumps(metadata or {})),
    )


def _insert_edge(
    conn, source: str, target: str, edge_type: str, weight: float = 1.0, metadata=None
) -> None:
    conn.execute(
        """
        INSERT INTO edges(id, source_node_id, target_node_id, edge_type, weight, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_node_id, target_node_id, edge_type) DO UPDATE SET
            weight = excluded.weight,
            metadata_json = excluded.metadata_json
        """,
        (
            _edge_id(source, target, edge_type),
            source,
            target,
            edge_type,
            weight,
            json.dumps(metadata or {}),
        ),
    )


def _topic_terms(text: str | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in _TOKEN_RE.findall(text or ""):
        term = token.lower()
        if term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= 6:
            break
    return out


def rebuild_graph_for_asset(asset_id: str) -> None:
    with connect() as conn, transaction(conn):
        node_ids = [
            r["id"]
            for r in conn.execute("SELECT id FROM nodes WHERE asset_id = ?", (asset_id,)).fetchall()
        ]
        if node_ids:
            placeholders = ",".join("?" * len(node_ids))
            conn.execute(
                f"DELETE FROM edges WHERE source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders})",
                [*node_ids, *node_ids],
            )
        conn.execute("DELETE FROM nodes WHERE asset_id = ?", (asset_id,))

        asset = conn.execute(
            "SELECT id, title, source_url FROM assets WHERE id = ?", (asset_id,)
        ).fetchone()
        if asset is None:
            return
        asset_node = f"asset:{asset_id}"
        _insert_node(
            conn,
            asset_node,
            "asset",
            asset_id,
            asset["title"] or asset["source_url"] or asset_id,
        )

        rows = conn.execute(
            """
            SELECT id, item_type, chunk_idx, scene_id, frame_id, segment_id,
                   page_idx, text, caption, file_path
              FROM content_items
             WHERE asset_id = ?
             ORDER BY chunk_idx, id
            """,
            (asset_id,),
        ).fetchall()

        prev_item_node: str | None = None
        for row in rows:
            item_node = f"item:{row['id']}"
            label = (row["text"] or row["caption"] or row["file_path"] or row["id"])[:120]
            _insert_node(
                conn,
                item_node,
                row["item_type"],
                asset_id,
                label,
                {"content_item_id": row["id"], "page_idx": row["page_idx"]},
            )
            _insert_edge(conn, asset_node, item_node, "contains")
            _insert_edge(conn, item_node, asset_node, "part_of")

            if row["scene_id"] is not None:
                scene_node = f"scene:{row['scene_id']}"
                _insert_node(conn, scene_node, "scene", asset_id, f"scene:{row['scene_id']}")
                _insert_edge(conn, scene_node, item_node, "contains")
                _insert_edge(conn, item_node, scene_node, "part_of")

            if prev_item_node is not None:
                _insert_edge(conn, prev_item_node, item_node, "adjacent", 0.7)
                _insert_edge(conn, item_node, prev_item_node, "adjacent", 0.7)
            prev_item_node = item_node

            for term in _topic_terms(
                " ".join(part for part in (row["text"], row["caption"]) if part)
            ):
                topic_node = f"topic:{asset_id}:{term}"
                _insert_node(conn, topic_node, "topic", asset_id, term)
                _insert_edge(conn, item_node, topic_node, "mentions", 0.5)
                _insert_edge(conn, topic_node, item_node, "mentioned_by", 0.5)
