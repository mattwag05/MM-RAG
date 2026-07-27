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
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
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
    assert "HELLO" in out_frames[0]["ocr_text"]
    assert "WORLD" in out_frames[0]["ocr_text"]


async def test_ocr_on_empty_frames_returns_empty_list():
    patch = await ocr(frames=[])
    assert patch["frames"] == []


async def test_ocr_survives_single_frame_failure(tmp_path):
    good = tmp_path / "good.jpg"
    # Not a 2-character string: PSM 3 needs a little page structure to
    # segment, and drops an isolated token like "OK". Anything from roughly
    # "Chapter 1" upward reads fine. This frame only has to be readable —
    # what is under test is that a sibling frame's failure does not kill it.
    _make_text_frame(good, "Chapter 1")
    frames = [
        {
            "scene_idx": 0,
            "frame_idx": 0,
            "t_s": 0.0,
            "path": str(good),
            "width": 400,
            "height": 120,
        },
        {
            "scene_idx": 0,
            "frame_idx": 1,
            "t_s": 1.0,
            "path": str(tmp_path / "missing.jpg"),
            "width": 400,
            "height": 120,
        },
    ]
    patch = await ocr(frames=frames)
    assert patch["frames"][0]["ocr_text"]
    assert patch["frames"][1]["ocr_text"] == ""


async def test_ocr_raises_oc_rerror_when_tesseract_binary_missing(monkeypatch):
    """A missing tesseract binary is a hard setup error, not per-frame failure."""
    from mmrag.pipeline import stages
    from mmrag.pipeline.m3_errors import OCRError

    # Nuke the cached success flag so the availability check runs again.
    monkeypatch.setattr(stages.ocr, "_TESSERACT_AVAILABLE", False)
    monkeypatch.setattr(stages.ocr.shutil, "which", lambda _name: None)

    with pytest.raises(OCRError) as excinfo:
        await ocr(
            frames=[
                {
                    "scene_idx": 0,
                    "frame_idx": 0,
                    "t_s": 0.0,
                    "path": "/tmp/fake.jpg",
                    "width": 1,
                    "height": 1,
                }
            ]
        )
    assert excinfo.value.kind == "binary_missing"
    assert "install" in excinfo.value.message.lower()


async def test_ocr_timeout_uses_kill_capable_subprocess_wrapper(monkeypatch, tmp_path):
    from mmrag.pipeline import stages
    from mmrag.pipeline.subprocess_util import SubprocessTimeout

    p = tmp_path / "slow.jpg"
    _make_text_frame(p, "SLOW")
    seen = {}

    async def fake_run(argv, *, timeout_s, **kwargs):
        seen["argv"] = argv
        seen["timeout_s"] = timeout_s
        raise SubprocessTimeout("tesseract exceeded timeout")

    monkeypatch.setattr(stages.ocr, "_TESSERACT_AVAILABLE", True)
    monkeypatch.setattr(stages.ocr, "run", fake_run)

    patch = await ocr(
        frames=[
            {
                "scene_idx": 0,
                "frame_idx": 0,
                "t_s": 0.0,
                "path": str(p),
                "width": 400,
                "height": 120,
            }
        ]
    )

    assert patch["frames"][0]["ocr_text"] == ""
    # PSM is pinned to 3, not 6, on purpose: PSM 6 asserts a uniform text
    # block exists and so hallucinates text out of video texture. Do not
    # "fix" this back to 6 — see the measurement table in stages/ocr.py.
    assert seen["argv"] == ["tesseract", str(p), "stdout", "--psm", "3"]
    assert seen["timeout_s"] == 10.0
