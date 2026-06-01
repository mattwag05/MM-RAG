"""Stage 1: fetch — local file path. URL ingestion is exercised manually
with a real network in M1; checking it in CI would couple us to yt-dlp
extractor stability."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import get_settings
from mmrag.pipeline.stages import fetch as fetch_mod
from mmrag.pipeline.stages.fetch import FetchError, fetch
from tests.conftest import SAMPLE_MP4


@pytest.mark.asyncio
async def test_fetch_local_file_creates_hash_keyed_copy(isolated_data_dir: Path) -> None:
    out = await fetch(source=str(SAMPLE_MP4))

    assert out["source_kind"] == "file"
    assert out["source_url"] is None
    assert out["title"] == SAMPLE_MP4.stem
    assert len(out["content_hash"]) == 64  # sha256 hex
    raw = Path(out["raw_path"])
    assert raw.exists()
    assert raw.parent.name == out["content_hash"]


@pytest.mark.asyncio
async def test_fetch_missing_file_raises(isolated_data_dir: Path) -> None:
    with pytest.raises(FetchError) as exc:
        await fetch(source="/definitely/not/a/real/path.mp4")
    assert exc.value.kind == "source_not_found"


@pytest.mark.asyncio
async def test_fetch_idempotent(isolated_data_dir: Path) -> None:
    """Re-fetching the same file should produce the same content hash and
    not duplicate the raw file (idempotency by hash)."""
    a = await fetch(source=str(SAMPLE_MP4))
    b = await fetch(source=str(SAMPLE_MP4))
    assert a["content_hash"] == b["content_hash"]
    assert a["raw_path"] == b["raw_path"]


@pytest.mark.asyncio
async def test_fetch_duplicate_url_cleans_second_raw_download(
    isolated_data_dir: Path, monkeypatch
) -> None:
    payload = b"same downloaded bytes"
    calls = 0

    async def fake_fetch_url(_source: str, dest_dir: Path):
        nonlocal calls
        calls += 1
        path = dest_dir / f"video-{calls}.mp4"
        path.write_bytes(payload)
        return path, {
            "title": "Fixture",
            "duration_s": 1.0,
            "extractor": "fixture",
            "webpage_url": "https://example.test/video",
        }

    monkeypatch.setattr(fetch_mod, "_fetch_url", fake_fetch_url)

    first = await fetch(source="https://example.test/video")
    second = await fetch(source="https://example.test/video")

    assert first["content_hash"] == second["content_hash"]
    assert first["raw_path"] == second["raw_path"]
    assert Path(first["raw_path"]).exists()
    assert list((get_settings().assets_dir / "raw").iterdir()) == []
