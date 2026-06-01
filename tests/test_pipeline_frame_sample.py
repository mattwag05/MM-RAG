"""Stage 5 frame_sample: midpoint sample per scene, 2s stride on scenes >10s."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mmrag.pipeline.stages.frame_sample import frame_sample

pytestmark = pytest.mark.m3_visual


def _make_test_video(path: Path, duration: int = 6) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=160x120:rate=24",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


async def test_frame_sample_midpoint_per_scene(tmp_path):
    video = tmp_path / "testsrc.mp4"
    _make_test_video(video, duration=6)
    scenes = [
        {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
        {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
        {"scene_idx": 2, "start_s": 4.0, "end_s": 6.0},
    ]
    patch = await frame_sample(
        mezzanine_path=str(video),
        scenes=scenes,
        assets_dir=tmp_path,
        content_hash="testhash",
        mode="standard",
    )
    frames = patch["frames"]
    assert len(frames) == 3
    t_values = [f["t_s"] for f in frames]
    assert t_values == [1.0, 3.0, 5.0]
    # Import PIL inside the test (not at module top level) so collection
    # survives on a core-only install: this module is marked m3_visual and the
    # conftest skips it, but a top-level `from PIL import Image` would raise at
    # COLLECTION time — before the skip can apply. Matches tests/test_pipeline_ocr.py.
    from PIL import Image

    for f in frames:
        p = Path(f["path"])
        assert p.exists() and p.stat().st_size > 0
        with Image.open(p) as img:
            assert f["width"] == img.width
            assert f["height"] == img.height


async def test_frame_sample_long_scene_strides_every_2s(tmp_path):
    video = tmp_path / "testsrc_long.mp4"
    _make_test_video(video, duration=15)
    scenes = [{"scene_idx": 0, "start_s": 0.0, "end_s": 15.0}]
    patch = await frame_sample(
        mezzanine_path=str(video),
        scenes=scenes,
        assets_dir=tmp_path,
        content_hash="longhash",
        mode="standard",
    )
    frames = patch["frames"]
    # 1 midpoint (7.5) + strides starting at start_s+1.0 with 2s step up to
    # end_s-0.5 => [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0] (7 strides; 7.5
    # midpoint does NOT collide with any integer stride so no dedup applies).
    assert len(frames) == 8
    assert frames[0]["t_s"] == pytest.approx(7.5, abs=0.1)
    stride_ts = [f["t_s"] for f in frames[1:]]
    assert stride_ts == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])


async def test_frame_sample_dedupes_midpoint_stride_collision(tmp_path):
    """Even-duration long scene: midpoint coincides with a stride sample.

    A 14-second scene has midpoint 7.0 which would also be emitted as a
    stride sample (start_s=0 + 1.0, 3.0, 5.0, 7.0, ...). The stage must
    deduplicate to a single 7.0 entry so downstream frame_idx values don't
    point at identical pixel data.
    """
    video = tmp_path / "testsrc_14.mp4"
    _make_test_video(video, duration=14)
    scenes = [{"scene_idx": 0, "start_s": 0.0, "end_s": 14.0}]
    patch = await frame_sample(
        mezzanine_path=str(video),
        scenes=scenes,
        assets_dir=tmp_path,
        content_hash="14hash",
        mode="standard",
    )
    frames = patch["frames"]
    t_values = [f["t_s"] for f in frames]
    # midpoint 7.0 + strides 1.0, 3.0, 5.0, 9.0, 11.0, 13.0 (7.0 deduped)
    assert len(set(t_values)) == len(t_values), f"duplicate t_s: {t_values}"
    assert 7.0 in t_values
    assert t_values.count(7.0) == 1
    assert len(frames) == 7  # not 8


async def test_frame_sample_none_mezzanine_returns_empty(tmp_path):
    patch = await frame_sample(
        mezzanine_path=None,
        scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 1.0}],
        assets_dir=tmp_path,
        content_hash="h",
        mode="standard",
    )
    assert patch == {"frames": []}


async def test_frame_sample_empty_scenes_returns_empty(tmp_path):
    video = tmp_path / "testsrc.mp4"
    _make_test_video(video, duration=2)
    patch = await frame_sample(
        mezzanine_path=str(video),
        scenes=[],
        assets_dir=tmp_path,
        content_hash="h",
        mode="standard",
    )
    assert patch == {"frames": []}


async def test_frame_sample_missing_mezzanine_file_returns_empty(tmp_path):
    patch = await frame_sample(
        mezzanine_path=str(tmp_path / "does_not_exist.mp4"),
        scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 1.0}],
        assets_dir=tmp_path,
        content_hash="h",
        mode="standard",
    )
    assert patch == {"frames": []}
