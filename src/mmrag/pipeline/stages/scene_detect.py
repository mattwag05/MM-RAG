"""Stage 3: scene detection via PySceneDetect's ContentDetector.

PySceneDetect's Python API runs synchronously (OpenCV-backed frame reads),
so we hop to a worker thread via ``asyncio.to_thread`` to keep the event
loop responsive. For uniform clips with no detected cuts we fall back to a
single scene spanning the full duration so downstream stages can always
rely on ``len(shots) >= 1`` when a mezzanine exists.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mmrag.logging import get_logger

log = get_logger("stage.scene_detect")


def _detect_shots_sync(mezzanine_path: str) -> list[dict]:
    # Imported lazily so `mmrag` still imports cleanly if scenedetect is
    # somehow absent at runtime (e.g. a stripped M1-era install).
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(mezzanine_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    scene_manager.detect_scenes(video=video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    shots: list[dict] = []
    if not scene_list:
        # Uniform content — emit a single scene covering the whole clip so
        # downstream stages don't have to special-case the zero-shot path.
        duration_s = float(video.duration.get_seconds()) if video.duration else 0.0
        shots.append({"shot_idx": 0, "start_s": 0.0, "end_s": duration_s})
        return shots

    for idx, (start, end) in enumerate(scene_list):
        shots.append(
            {
                "shot_idx": idx,
                "start_s": float(start.get_seconds()),
                "end_s": float(end.get_seconds()),
            }
        )
    return shots


async def scene_detect(*, mezzanine_path: str | None) -> dict:
    if mezzanine_path is None:
        return {"shots": []}
    if not Path(mezzanine_path).exists():
        log.warning("mezzanine_missing", path=mezzanine_path)
        return {"shots": []}

    log.info("detect.start", path=mezzanine_path)
    shots = await asyncio.to_thread(_detect_shots_sync, mezzanine_path)
    log.info("detect.done", path=mezzanine_path, n_shots=len(shots))
    return {"shots": shots}
