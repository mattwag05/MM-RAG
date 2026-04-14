"""Stage 3: scene detection via PySceneDetect's ContentDetector.

PySceneDetect's Python API runs synchronously (OpenCV-backed frame reads),
so we hop to a worker thread via ``asyncio.to_thread`` to keep the event
loop responsive. For uniform clips with no detected cuts we fall back to a
single scene spanning the full duration so downstream stages can always
rely on ``len(scenes) >= 1`` when a mezzanine exists.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mmrag.logging import get_logger

log = get_logger("stage.scene_detect")


def _detect_scenes_sync(mezzanine_path: str) -> list[dict]:
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(mezzanine_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    scene_manager.detect_scenes(video=video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    scenes: list[dict] = []
    if not scene_list:
        duration_s = float(video.duration.get_seconds()) if video.duration else 0.0
        scenes.append({"scene_idx": 0, "start_s": 0.0, "end_s": duration_s})
        return scenes

    for idx, (start, end) in enumerate(scene_list):
        scenes.append(
            {
                "scene_idx": idx,
                "start_s": float(start.get_seconds()),
                "end_s": float(end.get_seconds()),
            }
        )
    return scenes


async def scene_detect(*, mezzanine_path: str | None) -> dict:
    if mezzanine_path is None:
        return {"scenes": []}
    if not Path(mezzanine_path).exists():
        log.warning("mezzanine_missing", path=mezzanine_path)
        return {"scenes": []}

    log.info("detect.start", path=mezzanine_path)
    scenes = await asyncio.to_thread(_detect_scenes_sync, mezzanine_path)
    log.info("detect.done", path=mezzanine_path, n_scenes=len(scenes))
    return {"scenes": scenes}
