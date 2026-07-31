"""Stage 4: transcription via onnx-asr (Parakeet TDT 0.6b v3, int8).

``.with_vad()`` is mandatory, not an optimisation: these models cap at
20-30 s of audio and long-form recognition only works through VAD, so a
no-VAD path silently truncates every real asset. Silero also carries the
hallucination guard the old faster-whisper ``vad_filter=True`` provided —
junk decoded over music otherwise lands in ``fts_transcript`` as real
evidence and masks the silent scenes the caption stage is scoped to.
(Measured on Big Buck Bunny, 634 s of music: faster-whisper emitted 8
segments all reading "fa"; this emits 1, a real "Mm." at 54.7 s.)

Weights land in the shared HF hub cache alongside SigLIP (embed.py) and
Florence-2 (caption.py), deliberately NOT under ``settings.data_dir`` —
tests get a fresh isolated data dir each, which would re-download 640 MB
on every run. (The Dockerfile points ``HF_HOME`` inside its volume.)

``_MAX_SPEECH_S`` overrides Silero's 20 s default. A hit's snippet and time
range ARE the answer here, and a 20 s block spanning several sentences is a
worse answer than a sentence. Measured on the 89-scene reference asset
(5fa36205, 268 s of dense speech):

===========  ====  ======  =====  ============  ======  ====
config       segs  median    p90  scenes w/ FK  silent  wall
===========  ====  ======  =====  ============  ======  ====
whisper        60   ~4.5s      -            51       0     -
default        24   12.2s  20.0s            22       3   22s
=8             46    5.9s   8.0s            42       3   13s
=5             70    4.0s   5.0s            55       3   11s
===========  ====  ======  =====  ============  ======  ====

5 wins on every axis: it restores faster-whisper's granularity, attributes
more scenes than faster-whisper did, and runs 2x faster than the default
(short uniform windows batch without padding to 20 s). Lower risks
splitting mid-sentence.

The stage is structured in two layers:

- ``_run_speech_to_text`` is the primitive speech-to-text call that loads
  the model lazily and returns raw ``[{"start","end","text"}]`` dicts in
  source order. Tests monkey-patch this with a fake so the stage logic can
  be exercised without loading the model.
- ``transcribe`` is the stage entry point. It trims empty output, assigns a
  ``seg_idx``, and associates each segment with a scene via ``_assign_scene``.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("stage.transcribe")

_MODEL = None
_MAX_SPEECH_S = 5  # see module docstring for the measurement table

# "HH:MM:SS.mmm" with optional hours ("MM:SS.mmm"), as WebVTT allows both.
_VTT_CUE_RE = re.compile(
    r"^\s*((?:\d+:)?\d{2}:\d{2}\.\d{3})\s+-->\s+((?:\d+:)?\d{2}:\d{2}\.\d{3})"
)
_VTT_TAG_RE = re.compile(r"<[^>]+>")


def _vtt_ts(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def _parse_vtt(path: Path) -> list[dict]:
    """Parse a WebVTT caption file into raw ``[{"start","end","text"}]`` dicts.

    Platform caption tracks fetched by the fetch stage (MM-RAG-8vj). Styling
    tags are stripped; multi-line cue payloads join with spaces.
    """
    cues: list[dict] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = _VTT_CUE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        start, end = _vtt_ts(m.group(1)), _vtt_ts(m.group(2))
        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(_VTT_TAG_RE.sub("", lines[i]).strip())
            i += 1
        text = " ".join(t for t in text_lines if t).strip()
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def _get_model():
    global _MODEL
    if _MODEL is None:
        import onnx_asr

        settings = get_settings()
        log.info("asr.load", model=settings.transcribe_model)
        # ponytail: no providers= override — onnxruntime picks its own
        # per-platform default, which already includes CoreML on Apple
        # silicon. Pin it only if some platform needs forcing.
        _MODEL = onnx_asr.load_model(
            settings.transcribe_model,
            quantization=settings.transcribe_quantization,
        ).with_vad(onnx_asr.load_vad("silero"), max_speech_duration_s=_MAX_SPEECH_S)
    return _MODEL


def _run_speech_to_text(audio_path: str) -> list[dict]:
    """Return VAD-delimited speech segments in source order.

    Language is detected by the model rather than pinned — Parakeet TDT v3
    covers 25 languages, and MM-RAG ships as a general-purpose plugin.
    """
    return [
        {"start": float(r.start), "end": float(r.end), "text": r.text}
        for r in _get_model().recognize(audio_path)
    ]


def _assign_scene(start_s: float, scenes: list[dict]) -> int | None:
    """Return the scene_idx whose [start_s, end_s) contains start_s, else None."""
    if not scenes:
        return None
    for s in scenes:
        if s["start_s"] <= start_s < s["end_s"]:
            return int(s["scene_idx"])
    # Past the last scene's start? Snap to the final scene.
    if start_s >= scenes[-1]["start_s"]:
        return int(scenes[-1]["scene_idx"])
    return None


def _to_segments(raw: list[dict], scenes: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for raw_seg in raw:
        text = (raw_seg.get("text") or "").strip()
        if not text:
            continue
        start_s = float(raw_seg["start"])
        end_s = float(raw_seg["end"])
        segments.append(
            {
                "seg_idx": len(segments),
                "start_s": start_s,
                "end_s": end_s,
                "text": text,
                "scene_idx": _assign_scene(start_s, scenes),
            }
        )
    return segments


async def transcribe(
    *, audio_path: str | None, scenes: list[dict], subtitle_path: str | None = None
) -> dict:
    """``transcript_source`` records provenance: ``captions`` when a platform
    subtitle track replaced ASR, ``asr`` otherwise (persisted with the job's
    pipeline_state)."""
    # Platform caption track wins over ASR: manual captions are authored, and
    # skipping ASR makes captioned-URL ingest dramatically cheaper (MM-RAG-8vj).
    # The fetch stage only requests MANUAL subs — auto-captions are worse than
    # Parakeet (~6.3% WER) and are never fetched.
    if subtitle_path is not None:
        if Path(subtitle_path).exists():
            cues = _parse_vtt(Path(subtitle_path))
            if cues:
                segments = _to_segments(cues, scenes)
                log.info(
                    "transcribe.captions", path=subtitle_path, n_segments=len(segments)
                )
                return {"segments": segments, "transcript_source": "captions"}
            log.warning("subtitles.empty", path=subtitle_path)
        else:
            log.warning("subtitles.missing", path=subtitle_path)

    if audio_path is None:
        return {"segments": [], "transcript_source": "asr"}
    if not Path(audio_path).exists():
        log.warning("audio_missing", path=audio_path)
        return {"segments": [], "transcript_source": "asr"}

    log.info("transcribe.start", path=audio_path, n_scenes=len(scenes))
    raw = await asyncio.to_thread(_run_speech_to_text, audio_path)
    segments = _to_segments(raw, scenes)
    log.info("transcribe.done", path=audio_path, n_segments=len(segments))
    return {"segments": segments, "transcript_source": "asr"}
