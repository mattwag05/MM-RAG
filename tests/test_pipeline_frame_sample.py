"""Stage 5 frame_sample: midpoint sample per scene, 2s stride on scenes >10s."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from mmrag.pipeline.stages.frame_sample import frame_sample

pytestmark = pytest.mark.m3_visual


def _make_test_video(path: Path, duration: int = 6) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size=160x120:rate=24",
            "-pix_fmt", "yuv420p",
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
    # end_s-0.5 => 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 (7 strides)
    assert len(frames) == 8
    # Index 0 is the midpoint (emitted first).
    assert frames[0]["t_s"] == pytest.approx(7.5, abs=0.1)
