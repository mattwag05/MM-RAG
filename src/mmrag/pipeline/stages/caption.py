"""Stage 7: dense captions for scenes that have no other evidence.

For a scene with speech or on-screen text the evidence pack is fine. For a
**silent scene with no burned-in text** the entire payload handed to the
calling agent used to be an asset id, a scene id, a timestamp, and
``summarize.py``'s constant "No transcript or OCR text detected." SigLIP
cross-modal retrieval found the right moment and the evidence layer threw
the content away.

A caption written here is an **indexing artifact**, exactly like OCR:
deterministic, cached, produced once at ingest, never at request time.
``synthesize=false`` remains the default at every layer.

Model choice is settled in ``docs/vlm-selection.md`` (MM-RAG-jyq), which
benchmarked 8 candidates on real MM-RAG keyframes. Do not change these
constants without re-running ``make bench-vlm``:

- ``florence-community/Florence-2-base`` with ``<DETAILED_CAPTION>`` —
  469 MB, MIT, ~200 ms/frame batched, and the repo contains zero ``.py``
  files so no ``trust_remote_code`` (which matters for a public plugin,
  and especially given CVE-2026-4372).
- ``float32``. **Never bfloat16** — MPS lacks optimised bf16 conv kernels
  and silently emulates them, and Florence's DaViT vision tower is
  conv-heavy.
- ``attn_implementation="sdpa"``; ``flash_attention_2`` does not exist on MPS.
- **Batch 8.** Peak memory scales with batch size, not model size: this
  0.23B model peaks at 21.3 GB at batch 32 versus 5.8 GB at batch 8, for
  4.5% more speed. Do not raise it.

Failure handling mirrors ``ocr.py``: a frame that cannot be read or
captioned degrades to an empty caption and a structured warning rather
than failing the stage. A missing ``m3-visual`` extra is a hard setup
error, not a per-frame failure.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mmrag.config import get_settings
from mmrag.logging import get_logger
from mmrag.pipeline.m3_errors import M3ExtraMissingError
from mmrag.pipeline.stages.summarize import _clean_text, _overlaps_scene

log = get_logger("stage.caption")

_MODEL = None
_PROCESSOR = None
_MODEL_NAME = "florence-community/Florence-2-base"
_TASK = "<DETAILED_CAPTION>"
_BATCH = 8
# Measured mean output is ~66 tokens and the head self-terminates, so this
# is headroom rather than a cap that shapes cost.
_MAX_NEW_TOKENS = 80


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model() -> tuple[Any, Any]:
    """Create-and-cache the captioner. Raises if the m3-visual extra is absent."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise M3ExtraMissingError(stage="caption") from e

    device = _device()
    log.info("caption.model_load", model=_MODEL_NAME, device=device)
    # AutoModelForImageTextToText, not AutoModelForCausalLM: transformers 5.x
    # registers Florence2Config under the image-text-to-text auto class, and
    # AutoModelForCausalLM raises "Unrecognized configuration class". This
    # matches scripts/vlm_bench.py, which is what produced the benchmark.
    model = AutoModelForImageTextToText.from_pretrained(
        _MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    # Inference mode. Mirrors embed.py: per-call no_grad rather than a
    # module-level grad toggle another call path could revert.
    model.train(False)
    processor = AutoProcessor.from_pretrained(_MODEL_NAME)

    _MODEL, _PROCESSOR = model, processor
    log.info("caption.model_ready", model=_MODEL_NAME, device=device)
    return _MODEL, _PROCESSOR


def _caption_paths_sync(paths: list[str]) -> dict[str, str]:
    """Caption each image path. Unreadable/failed frames map to ""."""
    import torch
    from PIL import Image

    model, processor = _load_model()
    device = _device()
    out: dict[str, str] = dict.fromkeys(paths, "")

    for start in range(0, len(paths), _BATCH):
        chunk = paths[start : start + _BATCH]
        loaded: list[tuple[str, Any]] = []
        for path in chunk:
            try:
                loaded.append((path, Image.open(path).convert("RGB")))
            except Exception as e:  # noqa: BLE001
                log.warning("caption.frame_unreadable", path=path, error=str(e))
        if not loaded:
            continue
        try:
            images = [img for _, img in loaded]
            inputs = processor(text=[_TASK] * len(images), images=images, return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                )
            # skip_special_tokens=False: post_process_generation parses the
            # task tags, so stripping them first loses the caption.
            decoded = processor.batch_decode(generated, skip_special_tokens=False)
            for (path, img), raw in zip(loaded, decoded, strict=True):
                parsed = processor.post_process_generation(raw, task=_TASK, image_size=img.size)
                out[path] = " ".join(str(parsed.get(_TASK, "")).split())
        except Exception as e:  # noqa: BLE001
            # A whole batch failing must not fail the ingest — these frames
            # simply keep the empty-scene behaviour they had before.
            log.warning("caption.batch_failed", n_frames=len(loaded), error=str(e))
        finally:
            for _, img in loaded:
                img.close()
    return out


def _frames_needing_caption(
    *, scenes: list[dict], segments: list[dict], frames: list[dict]
) -> list[dict]:
    """Midpoint frame of each scene that has neither speech nor on-screen text.

    This is exactly the population that makes ``summarize.py`` emit its
    empty-scene constant, which is the hole this stage exists to fill.

    Scoped deliberately narrowly. Measured on the 89-scene reference asset
    after MM-RAG-xvg: 8 scenes qualify here, versus 28 with no transcript at
    all (the rest have real on-screen text and already return evidence).
    Broadening to the no-transcript population is a follow-up to justify
    with retrieval eval, not an assumption.
    """
    selected: list[dict] = []
    for scene in scenes:
        if any(_overlaps_scene(seg, scene) and _clean_text(seg.get("text")) for seg in segments):
            continue
        scene_frames = [f for f in frames if _overlaps_scene(f, scene)]
        if any(_clean_text(f.get("ocr_text")) for f in scene_frames):
            continue
        midpoint = next((f for f in scene_frames if int(f.get("frame_idx", 0)) == 0), None)
        if midpoint and midpoint.get("path"):
            selected.append(midpoint)
    return selected


async def caption(*, scenes: list[dict], segments: list[dict], frames: list[dict]) -> dict:
    if not frames:
        return {"frames": []}
    if not get_settings().caption_enabled:
        log.info("caption.disabled")
        return {"frames": [{**f, "caption": f.get("caption") or ""} for f in frames]}

    targets = _frames_needing_caption(scenes=scenes, segments=segments, frames=frames)
    if not targets:
        log.info("caption.no_silent_scenes", n_scenes=len(scenes))
        return {"frames": [{**f, "caption": f.get("caption") or ""} for f in frames]}

    paths = [str(f["path"]) for f in targets]
    captions = await asyncio.to_thread(_caption_paths_sync, paths)

    out = [{**f, "caption": captions.get(str(f.get("path")), "") or ""} for f in frames]
    n_captioned = sum(1 for c in captions.values() if c)
    log.info(
        "caption.done",
        n_scenes=len(scenes),
        n_selected=len(targets),
        n_captioned=n_captioned,
    )
    return {"frames": out}
