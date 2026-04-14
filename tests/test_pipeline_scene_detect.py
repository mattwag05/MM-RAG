"""Stage 3: scene_detect — PySceneDetect ContentDetector."""

from __future__ import annotations

import pytest

from mmrag.pipeline.stages.scene_detect import scene_detect
from tests.conftest import MULTISHOT_MP4, SAMPLE_MP4


@pytest.mark.asyncio
async def test_scene_detect_finds_cut_in_multishot_clip() -> None:
    result = await scene_detect(mezzanine_path=str(MULTISHOT_MP4))
    scenes = result["scenes"]
    # Two hard-cut 2s solid-color scenes → at least 2 scenes, cut near 2.0s.
    assert len(scenes) >= 2
    # Scene indices are monotonic from 0.
    assert [s["scene_idx"] for s in scenes] == list(range(len(scenes)))
    # Boundaries cover the whole clip, no gaps.
    assert scenes[0]["start_s"] == pytest.approx(0.0, abs=0.05)
    for prev, curr in zip(scenes, scenes[1:], strict=False):
        assert curr["start_s"] == pytest.approx(prev["end_s"], abs=0.1)
    # The cut should land somewhere in the middle of the 4s clip.
    assert 1.5 < scenes[0]["end_s"] < 2.6


@pytest.mark.asyncio
async def test_scene_detect_single_scene_fallback() -> None:
    """A uniform testsrc clip has no cuts → return a single-scene fallback
    covering the full duration rather than an empty list, so downstream
    stages can always assume at least one scene exists."""
    result = await scene_detect(mezzanine_path=str(SAMPLE_MP4))
    scenes = result["scenes"]
    assert len(scenes) == 1
    assert scenes[0]["scene_idx"] == 0
    assert scenes[0]["start_s"] == pytest.approx(0.0, abs=0.05)
    assert scenes[0]["end_s"] > 2.0


@pytest.mark.asyncio
async def test_scene_detect_missing_path_returns_empty() -> None:
    """If the mezzanine path is None (audio-only asset), no scenes."""
    result = await scene_detect(mezzanine_path=None)
    assert result["scenes"] == []
