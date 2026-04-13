"""Stage 2: normalize — ffmpeg mezzanine + 16 kHz mono wav."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import get_settings
from mmrag.pipeline.stages.fetch import fetch
from mmrag.pipeline.stages.normalize import normalize
from tests.conftest import SAMPLE_MP4


@pytest.mark.asyncio
async def test_normalize_produces_mezzanine_and_audio(isolated_data_dir: Path) -> None:
    fetched = await fetch(source=str(SAMPLE_MP4))
    asset_dir = get_settings().assets_dir / fetched["content_hash"]

    result = await normalize(
        raw_path=fetched["raw_path"],
        content_hash=fetched["content_hash"],
        asset_dir=asset_dir,
    )

    mezz = Path(result["mezzanine_path"])
    audio = Path(result["audio_path"]) if result["audio_path"] else None

    assert mezz.exists() and mezz.suffix == ".mp4"
    assert audio is not None and audio.exists() and audio.suffix == ".wav"
    assert result["width"] == 320
    assert result["height"] == 240
    # ffmpeg testsrc lavfi at duration=3 produces ~3s; allow slack
    assert result["duration_s"] is not None
    assert 2.5 < result["duration_s"] < 3.5
    assert result["fps"] is not None and 29.0 < result["fps"] < 31.0


@pytest.mark.asyncio
async def test_normalize_idempotent(isolated_data_dir: Path) -> None:
    fetched = await fetch(source=str(SAMPLE_MP4))
    asset_dir = get_settings().assets_dir / fetched["content_hash"]

    a = await normalize(
        raw_path=fetched["raw_path"],
        content_hash=fetched["content_hash"],
        asset_dir=asset_dir,
    )
    mtime_before = Path(a["mezzanine_path"]).stat().st_mtime

    b = await normalize(
        raw_path=fetched["raw_path"],
        content_hash=fetched["content_hash"],
        asset_dir=asset_dir,
    )
    mtime_after = Path(b["mezzanine_path"]).stat().st_mtime

    assert a["mezzanine_path"] == b["mezzanine_path"]
    # second pass should be a no-op (file already exists, ffmpeg not re-run)
    assert mtime_before == mtime_after
