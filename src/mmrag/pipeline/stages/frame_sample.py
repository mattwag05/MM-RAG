"""Stage 5: frame sampling.

Samples one frame at the midpoint of every scene. For scenes longer than
10 seconds, additionally samples every 2 seconds starting at start_s+1.0
so we don't miss content in long static shots without blowing the frame
budget on the Pi.

Frames are written to ``{assets_dir}/{content_hash}/frames/{scene_idx:04d}_{frame_idx:02d}.jpg``
via a single-frame ffmpeg shell-out per sample point. Width/height are
read from the resulting JPEG via Pillow.
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
    for scene in scenes:
        scene_idx = int(scene["scene_idx"])
        start_s = float(scene["start_s"])
        end_s = float(scene["end_s"])
        times = _sample_times(start_s, end_s)
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
            w, h = await asyncio.to_thread(_read_dimensions, out_path)
            out.append(
                {
                    "scene_idx": scene_idx,
                    "frame_idx": frame_idx,
                    "t_s": t_s,
                    "path": str(out_path),
                    "width": w,
                    "height": h,
                }
            )

    log.info("frame_sample.done", n_frames=len(out))
    return {"frames": out}
