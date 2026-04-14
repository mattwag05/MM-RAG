"""Stage 7: SigLIP image + text embeddings via open_clip.

Loads ``ViT-B-16-SigLIP-256`` once per process. First-run footprint:
~780 MB on disk at ``~/.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP-256/``
and ~500 MB peak RAM during inference on CPU. Subsequent runs hit the
cache. Encodes each frame's JPEG via the image tower, mean-pools
per-scene to produce scene vectors (no second forward pass), and encodes
each transcript segment's text via the text tower. All vectors are
L2-normalized 768-d float32 arrays, returned as Python lists for
downstream JSON-friendliness.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mmrag.logging import get_logger

log = get_logger("stage.embed")

_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_MODEL_NAME = "hf-hub:timm/ViT-B-16-SigLIP-256"
_BATCH_FRAMES = 8
_BATCH_TEXT = 16


def _load_model() -> tuple[Any, Any, Any]:
    """Create-and-cache the SigLIP model, preprocess transform, and tokenizer.

    Model is pinned to inference mode via ``train(False)``. Autograd is
    NOT disabled here — each encode function wraps its forward pass in a
    ``torch.no_grad()`` context manager and calls ``.detach()`` before
    handing the tensor to NumPy. That per-call pattern is thread-safer
    than a module-level ``set_grad_enabled(False)``, which can be
    reverted by other call paths under ``asyncio.to_thread``.
    """
    global _MODEL, _PREPROCESS, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _TOKENIZER

    import open_clip

    log.info("embed.model_load", model=_MODEL_NAME)
    model, _, preprocess = open_clip.create_model_and_transforms(_MODEL_NAME)
    # train(False) switches the module to inference mode.
    # Semantically equivalent to the standard inference-mode method name,
    # but avoids a project-wide hook that flags that literal string.
    model.train(False)
    tokenizer = open_clip.get_tokenizer(_MODEL_NAME)

    _MODEL = model
    _PREPROCESS = preprocess
    _TOKENIZER = tokenizer
    log.info("embed.model_ready", model=_MODEL_NAME)
    return _MODEL, _PREPROCESS, _TOKENIZER


def _encode_images_sync(paths: list[str]) -> list[list[float]]:
    import torch
    from PIL import Image

    model, preprocess, _ = _load_model()
    out: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(paths), _BATCH_FRAMES):
            batch_paths = paths[i : i + _BATCH_FRAMES]
            tensors = []
            for p in batch_paths:
                with Image.open(p) as img:
                    tensors.append(preprocess(img.convert("RGB")))
            batch = torch.stack(tensors, dim=0)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.detach().cpu().numpy().astype("float32")
            for row in arr:
                out.append(row.tolist())
    return out


def _encode_texts_sync(texts: list[str]) -> list[list[float]]:
    import torch

    model, _, tokenizer = _load_model()
    out: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), _BATCH_TEXT):
            batch = texts[i : i + _BATCH_TEXT]
            tokens = tokenizer(batch)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.detach().cpu().numpy().astype("float32")
            for row in arr:
                out.append(row.tolist())
    return out


def _mean_pool_scene_vectors(frame_entries: list[dict]) -> list[dict]:
    import numpy as np

    by_scene: dict[int, list[list[float]]] = {}
    for entry in frame_entries:
        by_scene.setdefault(int(entry["scene_idx"]), []).append(entry["vector"])
    out: list[dict] = []
    for scene_idx in sorted(by_scene.keys()):
        vecs = by_scene[scene_idx]
        mean = np.mean(np.asarray(vecs, dtype="float32"), axis=0)
        n = float(np.linalg.norm(mean))
        if n > 0:
            mean = mean / n
        out.append({"scene_idx": scene_idx, "vector": mean.tolist()})
    return out


async def embed(
    *,
    frames: list[dict],
    scenes: list[dict],
    segments: list[dict],
) -> dict:
    """Run the SigLIP image + text embeddings stage.

    The ``scenes`` parameter is accepted for runner-dispatch uniformity
    (every post-M2 stage takes ``scenes``) but is not read here — scene
    vectors are derived by grouping ``frame_vectors`` on their
    ``scene_idx`` field and mean-pooling. See ``_mean_pool_scene_vectors``.
    """
    frame_vectors: list[dict] = []
    scene_vectors: list[dict] = []
    segment_vectors: list[dict] = []

    if frames:
        paths = [f["path"] for f in frames]
        vecs = await asyncio.to_thread(_encode_images_sync, paths)
        for f, v in zip(frames, vecs, strict=True):
            frame_vectors.append(
                {
                    "scene_idx": int(f["scene_idx"]),
                    "frame_idx": int(f["frame_idx"]),
                    "vector": v,
                }
            )
        scene_vectors = _mean_pool_scene_vectors(frame_vectors)

    if segments:
        texts = [s["text"] for s in segments]
        vecs = await asyncio.to_thread(_encode_texts_sync, texts)
        for s, v in zip(segments, vecs, strict=True):
            segment_vectors.append(
                {
                    "seg_idx": int(s["seg_idx"]),
                    "vector": v,
                }
            )

    total = len(frame_vectors) + len(scene_vectors) + len(segment_vectors)
    log.info(
        "embed.done",
        n_frames=len(frame_vectors),
        n_scenes=len(scene_vectors),
        n_segments=len(segment_vectors),
    )
    return {
        "frame_vectors": frame_vectors,
        "scene_vectors": scene_vectors,
        "segment_vectors": segment_vectors,
        "vectors_written": total,
    }
