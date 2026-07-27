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
from pathlib import Path

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("stage.transcribe")

_MODEL = None
_MAX_SPEECH_S = 5  # see module docstring for the measurement table


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


async def transcribe(*, audio_path: str | None, scenes: list[dict]) -> dict:
    if audio_path is None:
        return {"segments": []}
    if not Path(audio_path).exists():
        log.warning("audio_missing", path=audio_path)
        return {"segments": []}

    log.info("transcribe.start", path=audio_path, n_scenes=len(scenes))
    raw = await asyncio.to_thread(_run_speech_to_text, audio_path)

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

    log.info("transcribe.done", path=audio_path, n_segments=len(segments))
    return {"segments": segments}
