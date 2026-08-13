"""Stage 7 caption: scoping predicate, and one real Florence-2 pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.pipeline.stages.caption import _frames_needing_caption, caption

pytestmark = pytest.mark.m3_visual


def _scene(idx: int, start: float, end: float) -> dict:
    return {"scene_idx": idx, "start_s": start, "end_s": end}


def _frame(scene_idx: int, path: str, ocr: str = "", frame_idx: int = 0) -> dict:
    return {
        "scene_idx": scene_idx,
        "frame_idx": frame_idx,
        "t_s": float(scene_idx),
        "path": path,
        "ocr_text": ocr,
    }


def test_selects_only_scenes_with_neither_speech_nor_on_screen_text() -> None:
    scenes = [_scene(0, 0.0, 2.0), _scene(1, 2.0, 4.0), _scene(2, 4.0, 6.0)]
    segments = [{"scene_idx": 0, "start_s": 0.0, "end_s": 2.0, "text": "someone is talking"}]
    frames = [
        _frame(0, "/tmp/a.jpg"),  # has speech -> skip
        _frame(1, "/tmp/b.jpg", ocr="Chapter 1"),  # has on-screen text -> skip
        _frame(2, "/tmp/c.jpg"),  # silent and blank -> caption this one
    ]

    selected = _frames_needing_caption(scenes=scenes, segments=segments, frames=frames)

    assert [f["path"] for f in selected] == ["/tmp/c.jpg"]


def test_selects_only_the_scene_midpoint_frame() -> None:
    """Long scenes sample extra frames; captioning them all would be wasteful."""
    scenes = [_scene(0, 0.0, 30.0)]
    frames = [
        _frame(0, "/tmp/mid.jpg", frame_idx=0),
        _frame(0, "/tmp/extra1.jpg", frame_idx=1),
        _frame(0, "/tmp/extra2.jpg", frame_idx=2),
    ]

    selected = _frames_needing_caption(scenes=scenes, segments=[], frames=frames)

    assert [f["path"] for f in selected] == ["/tmp/mid.jpg"]


def test_whitespace_only_transcript_does_not_count_as_speech() -> None:
    scenes = [_scene(0, 0.0, 2.0)]
    segments = [{"scene_idx": 0, "start_s": 0.0, "end_s": 2.0, "text": "   "}]
    frames = [_frame(0, "/tmp/a.jpg", ocr="  ")]

    selected = _frames_needing_caption(scenes=scenes, segments=segments, frames=frames)

    assert [f["path"] for f in selected] == ["/tmp/a.jpg"]


async def test_caption_disabled_leaves_frames_untouched_and_loads_no_model(tmp_path) -> None:
    try:
        reset_settings_for_tests(Settings(data_dir=tmp_path, caption_enabled=False))
        frames = [_frame(0, "/nonexistent/x.jpg")]

        patch = await caption(scenes=[_scene(0, 0.0, 2.0)], segments=[], frames=frames)

        # A nonexistent path would raise or warn if the model had been reached.
        assert patch["frames"][0]["caption"] == ""
    finally:
        reset_settings_for_tests(Settings())


async def test_caption_describes_a_silent_scene_end_to_end(tmp_path: Path) -> None:
    """Real Florence-2 inference — no mocks, matching the suite's SigLIP tests.

    Guards the part that is easy to get subtly wrong: the raw decode is a
    tagged string, so without processor.post_process_generation this returns
    markup instead of a caption.
    """
    from PIL import Image, ImageDraw

    # A scene with no speech and no readable text, but real visual content.
    p = tmp_path / "silent.jpg"
    img = Image.new("RGB", (640, 360), (30, 90, 180))
    draw = ImageDraw.Draw(img)
    draw.ellipse([220, 90, 420, 290], fill=(240, 200, 40))
    img.save(p, "JPEG", quality=92)

    patch = await caption(
        scenes=[_scene(0, 0.0, 2.0)],
        segments=[],
        frames=[_frame(0, str(p))],
    )

    text = patch["frames"][0]["caption"]
    assert text, "silent scene produced no caption"
    assert "<" not in text, f"caption still contains task markup: {text!r}"
    # Florence-2's <DETAILED_CAPTION> head emits prose, not a label.
    assert len(text.split()) >= 4, f"caption too short to be a description: {text!r}"


async def test_unreadable_frame_degrades_to_empty_rather_than_failing(tmp_path) -> None:
    patch = await caption(
        scenes=[_scene(0, 0.0, 2.0)],
        segments=[],
        frames=[_frame(0, str(tmp_path / "missing.jpg"))],
    )

    assert patch["frames"][0]["caption"] == ""
