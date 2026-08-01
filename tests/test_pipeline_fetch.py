"""Stage 1: fetch — local file path. URL ingestion is exercised manually
with a real network in M1; checking it in CI would couple us to yt-dlp
extractor stability."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import get_settings
from mmrag.pipeline.stages import fetch as fetch_mod
from mmrag.pipeline.stages.fetch import FetchError, _format_selector, fetch
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


def test_subtitles_from_info_prefers_original_language() -> None:
    """Manual caption track selection from a yt-dlp info dict (MM-RAG-8vj)."""
    from mmrag.pipeline.stages.fetch import _subtitles_from_info

    info = {
        "language": "de",
        "requested_subtitles": {
            "en": {"filepath": "/tmp/v.en.vtt"},
            "de": {"filepath": "/tmp/v.de.vtt"},
        },
    }
    assert _subtitles_from_info(info) == ("/tmp/v.de.vtt", "de")


def test_subtitles_from_info_falls_back_to_first_track() -> None:
    from mmrag.pipeline.stages.fetch import _subtitles_from_info

    info = {
        "language": "fr",
        "requested_subtitles": {"en": {"filepath": "/tmp/v.en.vtt"}},
    }
    assert _subtitles_from_info(info) == ("/tmp/v.en.vtt", "en")
    assert _subtitles_from_info({"requested_subtitles": {}}) is None
    assert _subtitles_from_info({}) is None


def test_format_selector_prefers_a_dash_merge_over_progressive() -> None:
    """The old selector was ``best[ext=mp4]/best``, which on YouTube only ever
    matches progressive streams — capped at itag 18 = 640x360 — so every ingest
    silently ran at 360p while 2160p was on offer. On-screen text does not
    survive that, so the merge branch must come first (MM-RAG-7rm).
    """
    selector = _format_selector(1080)
    merge_at = selector.index("bestvideo[height<=?1080]+bestaudio")
    progressive_at = selector.index("best[ext=mp4]")
    assert merge_at < progressive_at


def test_format_selector_applies_the_height_cap_to_every_capped_branch() -> None:
    """A cap that only guards the first branch silently pulls 4K whenever the
    merge branch does not match."""
    selector = _format_selector(720)
    capped = [part for part in selector.split("/") if part.startswith(("bestvideo", "best["))]
    assert [p for p in capped if "height<=?720" in p] == capped[:2]
    # Non-strict comparison, so a format with no height metadata stays eligible.
    assert "height<=?720" in selector
    assert "height<=720" not in selector.replace("height<=?720", "")
