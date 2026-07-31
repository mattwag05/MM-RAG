"""Stage 5: frame sampling.

Samples one frame at the midpoint of every scene. For scenes longer than
10 seconds, additionally samples every 2 seconds starting at start_s+1.0
so we don't miss content in long static shots without blowing the frame
budget on the Pi.

Frames are written to ``{assets_dir}/{content_hash}/frames/{scene_idx:04d}_{frame_idx:02d}.jpg``
via a single-frame ffmpeg shell-out per sample point. Width/height are
read from the resulting JPEG via Pillow.

Runs of near-identical frames within a scene (held slides, static shots)
are deduplicated by 16x16 grayscale MAD before OCR/caption/embed pay for
them — see ``_drop_near_duplicates`` for why the comparison is
temporal-chain rather than all-pairs (MM-RAG-mdn).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mmrag.logging import get_logger
from mmrag.pipeline.subprocess_util import run

log = get_logger("stage.frame_sample")

_LONG_SCENE_THRESHOLD_S = 10.0
_STRIDE_S = 2.0
_FRAME_TIMEOUT_S = 15.0

# Near-duplicate cutoff for 16x16 grayscale mean-absolute-difference
# (0-255 scale). Measured: static lavfi sources 0.0-0.11 between 2s stride
# frames, moving testsrc/testsrc2 10-18. Held slides with camera/compression
# noise land well under 3; distinct content well over it (MM-RAG-mdn).
_DEDUP_MAD_THRESHOLD = 3.0


def _sample_times(start_s: float, end_s: float) -> list[float]:
    """Return sample timestamps for a scene: midpoint first, then 2s strides
    on long scenes (start_s+1.0, start_s+3.0, ...) up to end_s-0.5.

    Deduped while preserving order so an even-duration scene (e.g. 0..14s)
    doesn't emit the midpoint twice via a colliding stride sample.
    """
    midpoint = (start_s + end_s) / 2.0
    candidates: list[float] = [midpoint]
    if end_s - start_s > _LONG_SCENE_THRESHOLD_S:
        t = start_s + 1.0
        while t < end_s - 0.5:
            candidates.append(t)
            t += _STRIDE_S
    # Dedup while preserving order — uses rounded key so float drift
    # from repeated +2.0 additions doesn't bypass the set membership check.
    seen: set[float] = set()
    out: list[float] = []
    for t in candidates:
        key = round(t, 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


async def _write_one_frame(mezzanine_path: str, t_s: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fast seek: `-ss` before `-i` lands on the nearest keyframe before t_s.
    # Trades pixel-exact accuracy for speed (~10x faster on long files).
    # Acceptable because OCR and SigLIP tolerate ±1-2 frame drift, and the
    # Pi target cannot afford the decode cost of slow seek (`-ss` after `-i`).
    await run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t_s:.3f}",
            "-i",
            mezzanine_path,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(out_path),
        ],
        timeout_s=_FRAME_TIMEOUT_S,
    )


def _read_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


def _dedup_thumb(path: Path) -> list[int]:
    """16x16 grayscale thumbnail as a flat pixel list, for MAD comparison."""
    from PIL import Image

    with Image.open(path) as img:
        return list(img.convert("L").resize((16, 16)).getdata())


def _mad(a: list[int], b: list[int]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def _drop_near_duplicates(candidates: list[dict]) -> tuple[list[dict], int]:
    """Temporal-chain dedup within one scene: walk frames in time order and
    drop any whose thumb is near-identical to the last temporally KEPT frame.

    Deliberately NOT all-pairs: content can be periodic or carry small
    regions (counters, captions, slide text) that vanish at 16x16, so two
    far-apart frames can look alike while genuinely differing. Only a run of
    consecutive near-identical frames (held slide, static shot) is redundant.
    Survivors are returned in the original sample order (midpoint first).
    """
    kept_keys: set[int] = set()
    last_thumb: list[int] | None = None
    for cand in sorted(candidates, key=lambda c: c["t_s"]):
        if last_thumb is not None and _mad(cand["thumb"], last_thumb) < _DEDUP_MAD_THRESHOLD:
            continue
        kept_keys.add(cand["frame_idx"])
        last_thumb = cand["thumb"]
    kept = [c for c in candidates if c["frame_idx"] in kept_keys]
    return kept, len(candidates) - len(kept)


async def frame_sample(
    *,
    mezzanine_path: str | None,
    scenes: list[dict],
    assets_dir: Path,
    content_hash: str,
) -> dict:
    if mezzanine_path is None or not scenes:
        return {"frames": []}
    if not Path(mezzanine_path).exists():
        log.warning("mezzanine_missing", path=mezzanine_path)
        return {"frames": []}

    frames_dir = Path(assets_dir) / content_hash / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    out: list[dict] = []
    n_deduped = 0
    for scene in scenes:
        scene_idx = int(scene["scene_idx"])
        start_s = float(scene["start_s"])
        end_s = float(scene["end_s"])
        times = _sample_times(start_s, end_s)
        candidates: list[dict] = []
        for frame_idx, t_s in enumerate(times):
            out_path = frames_dir / f"{scene_idx:04d}_{frame_idx:02d}.jpg"
            try:
                await _write_one_frame(mezzanine_path, t_s, out_path)
            except Exception as e:  # noqa: BLE001 — per-frame failure is non-fatal
                log.warning(
                    "frame_sample.write_failed",
                    scene_idx=scene_idx,
                    frame_idx=frame_idx,
                    t_s=t_s,
                    error=str(e),
                )
                continue
            if not out_path.exists():
                continue
            thumb = await asyncio.to_thread(_dedup_thumb, out_path)
            candidates.append(
                {"frame_idx": frame_idx, "t_s": t_s, "path": out_path, "thumb": thumb}
            )

        # Drop near-identical frame runs (held slides, static shots) before
        # OCR/caption/embed pay for them (MM-RAG-mdn).
        kept, dropped = _drop_near_duplicates(candidates)
        n_deduped += dropped
        kept_paths = {c["path"] for c in kept}
        for cand in candidates:
            if cand["path"] not in kept_paths:
                cand["path"].unlink(missing_ok=True)
        for cand in kept:
            w, h = await asyncio.to_thread(_read_dimensions, cand["path"])
            out.append(
                {
                    "scene_idx": scene_idx,
                    "frame_idx": cand["frame_idx"],
                    "t_s": cand["t_s"],
                    "path": str(cand["path"]),
                    "width": w,
                    "height": h,
                }
            )

    log.info("frame_sample.done", n_frames=len(out), n_deduped=n_deduped)
    return {"frames": out}
