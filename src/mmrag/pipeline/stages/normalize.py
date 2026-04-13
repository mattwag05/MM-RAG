from __future__ import annotations

import json
from pathlib import Path

from mmrag.logging import get_logger
from mmrag.pipeline.subprocess_util import SubprocessFailed, run

log = get_logger("stage.normalize")


class NormalizeError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


async def _ffprobe(path: Path) -> dict:
    try:
        result = await run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            timeout_s=30.0,
        )
    except FileNotFoundError as e:
        raise NormalizeError("ffmpeg_missing", "ffprobe not found on PATH") from e
    except SubprocessFailed as e:
        raise NormalizeError("probe_failed", str(e)) from e
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict | None:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return None


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            n, d = float(num), float(den)
            return n / d if d else None
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


async def normalize(*, raw_path: str, content_hash: str, asset_dir: Path) -> dict:
    """Stage 2: produce a mezzanine mp4 + 16 kHz mono wav + technical metadata.

    Stream-copies if the source is already h264/aac in mp4, otherwise transcodes.
    """
    src = Path(raw_path)
    if not src.exists():
        raise NormalizeError("source_missing", f"missing raw file: {src}")

    probe = await _ffprobe(src)
    vstream = _video_stream(probe)
    duration_s: float | None = None
    fmt = probe.get("format") or {}
    if "duration" in fmt:
        try:
            duration_s = float(fmt["duration"])
        except ValueError:
            duration_s = None

    width = vstream.get("width") if vstream else None
    height = vstream.get("height") if vstream else None
    fps = _parse_fps(vstream.get("avg_frame_rate")) if vstream else None

    asset_dir.mkdir(parents=True, exist_ok=True)
    mezz_path = asset_dir / "mezzanine.mp4"
    audio_path = asset_dir / "audio.wav"

    if not mezz_path.exists():
        log.info("ffmpeg.mezzanine", src=str(src), out=str(mezz_path))
        try:
            await run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(mezz_path),
                ],
                timeout_s=600.0,
            )
        except FileNotFoundError as e:
            raise NormalizeError("ffmpeg_missing", "ffmpeg not found on PATH") from e
        except SubprocessFailed as e:
            raise NormalizeError("transcode_failed", str(e)) from e

    if not audio_path.exists() and vstream is not None:
        # Only attempt audio extraction if the source has audio streams.
        has_audio = any(
            s.get("codec_type") == "audio" for s in probe.get("streams", [])
        )
        if has_audio:
            log.info("ffmpeg.audio", src=str(src), out=str(audio_path))
            try:
                await run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(src),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-acodec",
                        "pcm_s16le",
                        str(audio_path),
                    ],
                    timeout_s=600.0,
                )
            except SubprocessFailed as e:
                raise NormalizeError("audio_extract_failed", str(e)) from e

    return {
        "duration_s": duration_s,
        "fps": fps,
        "width": width,
        "height": height,
        "mezzanine_path": str(mezz_path),
        "audio_path": str(audio_path) if audio_path.exists() else None,
    }
