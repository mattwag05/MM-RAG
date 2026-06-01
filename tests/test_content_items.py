from __future__ import annotations

import json
import uuid

import pytest

from mmrag.db.connection import connect
from mmrag.db.content_items import rewrite_content_items_for_asset
from mmrag.pipeline.runner import _persist_frames, _persist_scenes, _persist_segments

pytestmark = pytest.mark.m3_visual


def test_rewrite_content_items_projects_current_pipeline_tables(isolated_data_dir):
    asset_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets(id, content_hash, source_kind, metadata_json) "
            "VALUES (?, ?, 'file', '{}')",
            (asset_id, "content-items-hash"),
        )

    _persist_scenes(
        asset_id=asset_id,
        scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 4.0}],
    )
    _persist_segments(
        asset_id=asset_id,
        segments=[{"scene_idx": 0, "seg_idx": 0, "start_s": 0.5, "end_s": 1.5, "text": "hello"}],
    )
    with connect() as conn:
        scene_id = conn.execute(
            "SELECT id FROM scenes WHERE asset_id = ? AND scene_idx = 0",
            (asset_id,),
        ).fetchone()["id"]
    _persist_frames(
        asset_id=asset_id,
        scene_id_by_idx={0: scene_id},
        frames=[
            {
                "scene_idx": 0,
                "frame_idx": 0,
                "t_s": 1.0,
                "path": "/tmp/frame.jpg",
                "ocr_text": "visible text",
                "width": 320,
                "height": 240,
            }
        ],
    )

    count = rewrite_content_items_for_asset(asset_id)

    assert count == 3
    with connect() as conn:
        rows = conn.execute(
            "SELECT item_type, text, file_path, metadata_json "
            "FROM content_items WHERE asset_id = ? ORDER BY item_type",
            (asset_id,),
        ).fetchall()

    by_type = {r["item_type"]: r for r in rows}
    assert by_type["audio_segment"]["text"] == "hello"
    assert by_type["image"]["text"] == "visible text"
    assert by_type["image"]["file_path"] == "/tmp/frame.jpg"
    assert json.loads(by_type["image"]["metadata_json"]) == {"width": 320, "height": 240}
    assert by_type["video_segment"]["text"] is None
