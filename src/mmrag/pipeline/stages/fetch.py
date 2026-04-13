from __future__ import annotations

import asyncio
import hashlib
import shutil
import uuid
from pathlib import Path

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("stage.fetch")


class FetchError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def _fetch_url(source: str, dest_dir: Path) -> tuple[Path, dict]:
    """Download with yt-dlp into dest_dir. Returns (downloaded_path, info)."""
    # Imported lazily so that local-file ingest doesn't require yt-dlp at all.
    import yt_dlp  # type: ignore[import-untyped]

    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")
    opts = {
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": "best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "restrictfilenames": True,
    }

    def _do_download() -> dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(source, download=True)

    try:
        info = await asyncio.to_thread(_do_download)
    except Exception as e:  # yt-dlp raises a soup of error types
        raise FetchError("source_unreachable", f"yt-dlp: {e}") from e

    requested = info.get("requested_downloads") or []
    if requested:
        path = Path(requested[0]["filepath"])
    else:
        path = Path(info.get("_filename") or info.get("filename") or "")
    if not path.exists():
        raise FetchError(
            "source_unreachable",
            f"yt-dlp reported success but file not found: {path}",
        )
    return path, {
        "title": info.get("title"),
        "duration_s": info.get("duration"),
        "extractor": info.get("extractor"),
        "webpage_url": info.get("webpage_url") or source,
    }


async def fetch(*, source: str) -> dict:
    """Stage 1: produce a local raw file + content hash + minimal metadata.

    Returns a dict of fields that get merged into the job's pipeline_state and
    later promoted to the assets row in the persist step.
    """
    settings = get_settings()
    raw_dir = settings.assets_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if _is_url(source):
        log.info("fetch.url", source=source)
        downloaded, info = await _fetch_url(source, raw_dir)
        source_kind = "url"
        source_url = info.get("webpage_url")
        title = info.get("title")
        # Move into a hash-keyed location after we know the hash.
    else:
        src_path = Path(source).expanduser().resolve()
        if not src_path.exists() or not src_path.is_file():
            raise FetchError("source_not_found", f"file not found: {src_path}")
        log.info("fetch.local", path=str(src_path))
        downloaded = src_path
        source_kind = "file"
        source_url = None
        title = src_path.stem

    content_hash = await asyncio.to_thread(_hash_file, downloaded)
    asset_dir = settings.assets_dir / content_hash
    asset_dir.mkdir(parents=True, exist_ok=True)
    raw_dest = asset_dir / f"raw{downloaded.suffix or ''}"
    if not raw_dest.exists():
        if source_kind == "url":
            shutil.move(str(downloaded), str(raw_dest))
        else:
            shutil.copy2(str(downloaded), str(raw_dest))

    return {
        "asset_id": str(uuid.uuid4()),
        "content_hash": content_hash,
        "source_kind": source_kind,
        "source_url": source_url,
        "title": title,
        "raw_path": str(raw_dest),
    }
