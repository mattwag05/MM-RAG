"""Stage 8: deterministic per-scene summaries.

This is an indexing artifact, not request-time reasoning. It intentionally
does not call an LLM: summaries are distilled from transcript segments and
frame OCR already produced by earlier stages, so Pi deployments keep the same
small runtime footprint.
"""

_MAX_TRANSCRIPT_CHARS = 220
_MAX_OCR_CHARS = 160
# Florence-2 <DETAILED_CAPTION> averages ~66 tokens; this keeps a whole
# caption rather than clipping mid-sentence.
_MAX_CAPTION_CHARS = 400


def _clean_text(text: object) -> str:
    return " ".join(str(text or "").split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return f"{clipped or text[: max_chars - 1]}…"


def _overlaps_scene(item: dict, scene: dict) -> bool:
    if item.get("scene_idx") is not None:
        return int(item["scene_idx"]) == int(scene["scene_idx"])
    if item.get("start_s") is None or item.get("end_s") is None:
        return False
    return float(item["end_s"]) >= float(scene["start_s"]) and float(item["start_s"]) <= float(
        scene["end_s"]
    )


def _summarize_scene(*, scene: dict, segments: list[dict], frames: list[dict]) -> str:
    transcript_text = _clean_text(
        " ".join(
            _clean_text(seg.get("text"))
            for seg in segments
            if _overlaps_scene(seg, scene) and _clean_text(seg.get("text"))
        )
    )
    ocr_text = _clean_text(
        " ".join(
            _clean_text(frame.get("ocr_text"))
            for frame in frames
            if _overlaps_scene(frame, scene) and _clean_text(frame.get("ocr_text"))
        )
    )

    caption_text = _clean_text(
        " ".join(
            _clean_text(frame.get("caption"))
            for frame in frames
            if _overlaps_scene(frame, scene) and _clean_text(frame.get("caption"))
        )
    )

    parts = []
    if transcript_text:
        parts.append(f"Spoken: {_truncate(transcript_text, _MAX_TRANSCRIPT_CHARS)}")
    if ocr_text:
        parts.append(f"Visible text: {_truncate(ocr_text, _MAX_OCR_CHARS)}")
    # Only reached for scenes with neither — the caption stage captions
    # exactly that population, so this is the branch that used to be the
    # dead-end constant.
    if not parts and caption_text:
        parts.append(f"Scene shows: {_truncate(caption_text, _MAX_CAPTION_CHARS)}")
    if not parts:
        parts.append("No transcript or OCR text detected.")
    return " ".join(parts)


async def summarize(*, scenes: list[dict], segments: list[dict], frames: list[dict]) -> dict:
    return {
        "summaries": [
            {
                "scene_idx": int(scene["scene_idx"]),
                "summary": _summarize_scene(scene=scene, segments=segments, frames=frames),
            }
            for scene in scenes
        ]
    }
