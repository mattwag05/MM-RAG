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
    _make_text_frame(good, "OK")
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
    """If pytesseract.get_tesseract_version fails, ocr must raise OCRError
    with kind='binary_missing' instead of downgrading to empty strings."""
    import pytesseract

    from mmrag.pipeline import stages
    from mmrag.pipeline.m3_errors import OCRError

    # Nuke the cached success flag so the availability check runs again.
    monkeypatch.setattr(stages.ocr, "_TESSERACT_AVAILABLE", False)

    def _raise(*args, **kwargs):
        raise FileNotFoundError("tesseract is not installed")

    monkeypatch.setattr(pytesseract, "get_tesseract_version", _raise)

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
