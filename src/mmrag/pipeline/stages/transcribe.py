"""Stage 4: transcription via faster-whisper (ctranslate2 int8).

The stage is structured in two layers:

- ``_run_speech_to_text`` is the primitive speech-to-text call that loads
  the model lazily and returns raw ``[{"start","end","text"}]`` dicts in
  source order. Tests monkey-patch this with a fake so the stage logic can
  be exercised without loading a 40 MB model.
- ``transcribe`` is the stage entry point. It trims empty output, assigns a
  ``seg_idx``, and associates each segment with a scene via ``_assign_scene``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mmrag.config import get_settings
from mmrag.logging import get_logger

log = get_logger("stage.transcribe")

_WHISPER_MODEL = None
_WHISPER_MODEL_SIZE = "tiny.en"


def _get_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel

        settings = get_settings()
        cache_dir = settings.data_dir / "models" / "faster-whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("whisper.load", model=_WHISPER_MODEL_SIZE, cache=str(cache_dir))
        _WHISPER_MODEL = WhisperModel(
            _WHISPER_MODEL_SIZE,
            compute_type="int8",
            download_root=str(cache_dir),
        )
    return _WHISPER_MODEL


def _run_speech_to_text(audio_path: str) -> list[dict]:
    model = _get_model()
    segs, _ = model.transcribe(
        audio_path,
        language="en",
        beam_size=1,
        vad_filter=False,
    )
    return [{"start": float(s.start), "end": float(s.end), "text": s.text} for s in segs]


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
