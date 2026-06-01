from __future__ import annotations

import json

from mmrag.db.connection import connect, transaction
from mmrag.models.content_item import ContentItem


def _scene_items(conn, asset_id: str) -> list[ContentItem]:
    rows = conn.execute(
        """
        SELECT id, scene_idx, start_s, end_s, summary
          FROM scenes
         WHERE asset_id = ?
         ORDER BY scene_idx
        """,
        (asset_id,),
    ).fetchall()
    return [
        ContentItem(
            id=f"scene:{r['id']}",
            type="video_segment",
            source_id=asset_id,
            chunk_idx=int(r["scene_idx"]),
            asset_id=asset_id,
            scene_id=int(r["id"]),
            start_s=float(r["start_s"]),
            end_s=float(r["end_s"]),
            text=r["summary"],
        )
        for r in rows
    ]


def _segment_items(conn, asset_id: str) -> list[ContentItem]:
    rows = conn.execute(
        """
        SELECT id, scene_id, seg_idx, start_s, end_s, text
          FROM transcript_segments
         WHERE asset_id = ?
         ORDER BY seg_idx
        """,
        (asset_id,),
    ).fetchall()
    return [
        ContentItem(
            id=f"segment:{r['id']}",
            type="audio_segment",
            source_id=asset_id,
            chunk_idx=int(r["seg_idx"]),
            asset_id=asset_id,
            scene_id=int(r["scene_id"]) if r["scene_id"] is not None else None,
            segment_id=int(r["id"]),
            start_s=float(r["start_s"]),
            end_s=float(r["end_s"]),
            text=r["text"],
        )
        for r in rows
    ]


def _frame_items(conn, asset_id: str) -> list[ContentItem]:
    rows = conn.execute(
        """
        SELECT id, scene_id, frame_idx, t_s, path, ocr_text, width, height
          FROM frames
         WHERE asset_id = ?
         ORDER BY scene_id, frame_idx
        """,
        (asset_id,),
    ).fetchall()
    return [
        ContentItem(
            id=f"frame:{r['id']}",
            type="image",
            source_id=asset_id,
            chunk_idx=int(r["frame_idx"]),
            asset_id=asset_id,
            scene_id=int(r["scene_id"]),
            frame_id=int(r["id"]),
            start_s=float(r["t_s"]),
            end_s=float(r["t_s"]),
            text=r["ocr_text"],
            file_path=r["path"],
            metadata={"width": r["width"], "height": r["height"]},
        )
        for r in rows
    ]


def rewrite_content_items_for_asset(asset_id: str) -> int:
    """Rewrite the content_items projection for one asset.

    The projection is derived from canonical tables, so rebuilding is simpler
    and safer than trying to maintain row-level triggers across staged writes.
    """
    with connect() as conn, transaction(conn):
        items = [
            *_scene_items(conn, asset_id),
            *_segment_items(conn, asset_id),
            *_frame_items(conn, asset_id),
        ]
        conn.execute("DELETE FROM content_items WHERE asset_id = ?", (asset_id,))
        for item in items:
            conn.execute(
                """
                INSERT INTO content_items (
                    id, asset_id, item_type, source_id, chunk_idx, scene_id,
                    frame_id, segment_id, page_idx, start_s, end_s, text,
                    caption, file_path, metadata_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (
                    item.id,
                    item.asset_id,
                    item.type,
                    item.source_id,
                    item.chunk_idx,
                    item.scene_id,
                    item.frame_id,
                    item.segment_id,
                    item.page_idx,
                    item.start_s,
                    item.end_s,
                    item.text,
                    item.caption,
                    item.file_path,
                    json.dumps(item.metadata),
                ),
            )
    return len(items)
