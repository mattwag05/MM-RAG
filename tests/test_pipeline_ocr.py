"""Stage 6 ocr: extract burned-in text from a generated JPEG."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.pipeline.stages.ocr import ocr

pytestmark = pytest.mark.m3_visual


def _make_text_frame(path: Path, text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 120), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
            )
        except OSError:
            font = ImageFont.load_default()
    draw.text((10, 30), text, fill="black", font=font)
    img.save(path, "JPEG", quality=95)


async def test_ocr_extracts_burned_in_text(tmp_path):
    p = tmp_path / "hello.jpg"
    _make_text_frame(p, "HELLO WORLD")
    frames = [
        {
            "scene_idx": 0,
            "frame_idx": 0,
            "t_s": 0.0,
            "path": str(p),
            "width": 400,
            "height": 120,
        }
    ]
    patch = await ocr(frames=frames)
    out_frames = patch["frames"]
    assert len(out_frames) == 1
    assert "HELLO" in out_frames[0]["ocr_text"].upper()
    assert "WORLD" in out_frames[0]["ocr_text"].upper()


async def test_ocr_on_empty_frames_returns_empty_list():
    patch = await ocr(frames=[])
    assert patch["frames"] == []


async def test_ocr_survives_single_frame_failure(tmp_path):
    good = tmp_path / "good.jpg"
    _make_text_frame(good, "OK")
    frames = [
        {"scene_idx": 0, "frame_idx": 0, "t_s": 0.0, "path": str(good), "width": 400, "height": 120},
        {"scene_idx": 0, "frame_idx": 1, "t_s": 1.0, "path": str(tmp_path / "missing.jpg"), "width": 400, "height": 120},
    ]
    patch = await ocr(frames=frames)
    assert patch["frames"][0]["ocr_text"]
    assert patch["frames"][1]["ocr_text"] == ""
