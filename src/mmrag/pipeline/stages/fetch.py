from __future__ import annotations

import asyncio
import hashlib
import shutil
import uuid
from pathlib import Path

from mmrag.config import get_settings
from mmrag.logging import get_logger
from mmrag.pipeline.stages.document import is_document_source

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


def _subtitles_from_info(info: dict) -> tuple[str, str] | None:
    """Pick the manual caption track to use from a yt-dlp info dict.

    Prefers the video's original language, falls back to the first written
    track. Returns (filepath, lang) or None when no track was written.
    """
    subs = info.get("requested_subtitles") or {}
    tracks = {lang: t.get("filepath") for lang, t in subs.items() if t and t.get("filepath")}
    if not tracks:
        return None
    lang = info.get("language")
    if lang in tracks:
        return str(tracks[lang]), str(lang)
    first = next(iter(tracks))
    return str(tracks[first]), str(first)


def _format_selector(max_height: int) -> str:
    """yt-dlp format string capped at ``max_height`` (MM-RAG-7rm).

    The obvious-looking ``best[ext=mp4]/best`` is a trap on YouTube: ``best``
    only considers *progressive* streams (video and audio already muxed into
    one file), and YouTube's progressive mp4 tops out at itag 18 = 640x360.
    Everything above that is DASH, video-only, and reachable solely through an
    explicit ``bestvideo+bestaudio`` merge. Measured on the reference asset:
    the old selector took 640x360 while 2160p was on offer, and at 360p the
    on-screen text OCR exists to read is not present in the pixels at all.

    ``<=?`` is the non-strict comparison — a format with no height metadata
    stays eligible rather than dropping out of the running.
    """
    return (
        f"bestvideo[height<=?{max_height}]+bestaudio/best[height<=?{max_height}]/best[ext=mp4]/best"
    )


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
        "format": _format_selector(get_settings().max_video_height),
        "merge_output_format": "mp4",
        "restrictfilenames": True,
    }
    if get_settings().subtitles_enabled:
        # Manual captions only — never auto-captions, which are worse than the
        # local ASR (Parakeet ~6.3% WER). transcribe uses the track when
        # present and falls back to ASR otherwise (MM-RAG-8vj).
        opts.update(
            {
                "writesubtitles": True,
                "subtitlesformat": "vtt/best",
                "subtitleslangs": ["all"],
            }
        )

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
    subtitles = _subtitles_from_info(info)
    return path, {
        "title": info.get("title"),
        "duration_s": info.get("duration"),
        "extractor": info.get("extractor"),
        "webpage_url": info.get("webpage_url") or source,
        "subtitle_path": subtitles[0] if subtitles else None,
        "subtitle_lang": subtitles[1] if subtitles else None,
    }


async def fetch(*, source: str) -> dict:
    """Stage 1: produce a local raw file + content hash + minimal metadata.

    Returns a dict of fields that get merged into the job's pipeline_state and
    later promoted to the assets row in the persist step.
    """
    settings = get_settings()
    raw_dir = settings.assets_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    subtitle_path: str | None = None
    subtitle_lang: str | None = None
    if _is_url(source):
        log.info("fetch.url", source=source)
        downloaded, info = await _fetch_url(source, raw_dir)
        source_kind = "url"
        source_url = info.get("webpage_url")
        title = info.get("title")
        subtitle_path = info.get("subtitle_path")
        subtitle_lang = info.get("subtitle_lang")
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
    elif source_kind == "url" and downloaded.resolve() != raw_dest.resolve():
        downloaded.unlink(missing_ok=True)

    # Keep the caption track with the asset, like the raw media above.
    if subtitle_path is not None:
        sub_src = Path(subtitle_path)
        sub_dest = asset_dir / f"subs{''.join(sub_src.suffixes[-2:])}"
        if sub_src.exists() and sub_src.resolve() != sub_dest.resolve():
            shutil.move(str(sub_src), str(sub_dest))
        subtitle_path = str(sub_dest) if sub_dest.exists() else None

    return {
        "asset_id": str(uuid.uuid4()),
        "content_hash": content_hash,
        "source_kind": source_kind,
        "source_url": source_url,
        "title": title,
        "raw_path": str(raw_dest),
        "subtitle_path": subtitle_path,
        "subtitle_lang": subtitle_lang,
        "is_document": is_document_source(str(raw_dest)),
        "document_type": raw_dest.suffix.lower().lstrip(".") or None,
    }
