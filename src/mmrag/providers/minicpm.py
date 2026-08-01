"""MiniCPM-V-4.6 provider for the opt-in ``synthesize=true`` path (MM-RAG-thx).

The evidence-first default does not change: ``synthesize=false`` remains the
default at every layer and this module is never imported unless a caller opts
in *and* selects this backend. It exists for consumers that do not have a
capable model on the other end of MCP — for everyone else, reasoning over the
evidence pack with your own model is still the better answer.

Chosen over the Ollama/Gemma default because Gemma-4 caps at 30 s of audio and
~60 s of frames per call, while MiniCPM-V-4.6 (1.3B, Apache-2.0) had the best
caption quality measured across both benchmark rounds and can take retrieved
frame JPEGs alongside the evidence text. See ``docs/vlm-selection.md``.

**Not for edge deployments.** ~5.9 GB resident at fp32 — a Raspberry Pi cannot
run this, and it is not a drop-in for the Pi profile. Ingest-time captioning is
explicitly out of scope; Florence-2 stays there.

**On quantization.** ``MMRAG_SYNTHESIZE_MODEL`` points this at any compatible
repo, so a smaller variant is a config change. Be aware of what is actually
available on this hardware: 4-bit MiniCPM-V-4.6 exists as
``mlx-community/MiniCPM-V-4.6-4bit``, but MLX weights need the ``mlx-vlm``
runtime rather than transformers, and MLX is Apple-silicon only — the same
objection that ruled out FluidAudio for the ASR slot in a cross-platform
plugin. The usual CUDA quantizers (bitsandbytes, most GPTQ/AWQ kernels) do not
run on MPS either, so on Apple silicon fp32/fp16 through transformers is the
portable option and MLX is the fast one. Wiring an MLX adapter behind an
optional extra is a reasonable future move; it is deliberately not done here.

**No ``trust_remote_code``.** transformers 5.x supports MiniCPM-V-4.6
natively, so this keeps the same property that helped select Florence-2 for the
caption stage: no code from the model repo is executed. It is still opt-in
twice over — the caller must pass ``synthesize=true`` and the deployment must
set ``MMRAG_SYNTHESIZE_PROVIDER=minicpm``.

MPS rules, all measured rather than assumed (``docs/vlm-selection.md``):

- ``float32``. **Never bfloat16** — MPS has no optimised bf16 conv kernels and
  silently emulates them.
- ``attn_implementation="sdpa"``; ``flash_attention_2`` does not exist on MPS.
- ``.to(device)``, never ``device_map="auto"``, which is broken on MPS.
- **Images must be letterboxed to square.** On a native 16:9 frame MiniCPM
  raises ``RuntimeError: shape '[3, 1034, 1152]' is invalid for input of size
  3575808`` — its adaptive slicing miscounts visual tokens by exactly 2. This
  reproduces on the correct ``apply_chat_template`` path, so it is a
  transformers bug, not a harness artefact. Every MM-RAG keyframe is 16:9, so
  this is the normal case, not an edge case. Letterbox, never crop: cropping a
  16:9 frame to square throws away a third of the picture.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from io import BytesIO
from typing import Any

from mmrag.config import get_settings
from mmrag.logging import get_logger
from mmrag.providers.base import GenerateConfig, Message, ModelProvider, StreamChunk

log = get_logger("provider.minicpm")


_MODEL: Any = None
_PROCESSOR: Any = None


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load() -> tuple[Any, Any]:
    """Create-and-cache the model. Raises if the m3-visual extra is absent."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    from mmrag.pipeline.m3_errors import M3ExtraMissingError

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise M3ExtraMissingError(stage="synthesize") from e

    repo = get_settings().synthesize_model
    device = _device()
    log.info("minicpm.model_load", model=repo, device=device)
    # AutoModelForImageTextToText, not AutoModel: the latter resolves to the
    # bare MiniCPMV4_6Model, which has no `generate`. Matches scripts/vlm_bench.py,
    # which is what produced the published numbers.
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.train(False)
    processor = AutoProcessor.from_pretrained(repo)
    _MODEL, _PROCESSOR = model, processor
    log.info("minicpm.model_ready", model=repo, device=device)
    return _MODEL, _PROCESSOR


def letterbox_to_square(img: Any) -> Any:
    """Pad an image to square on a black canvas, preserving the whole frame.

    Public because it is the workaround for a real upstream constraint, not an
    implementation detail — see the module docstring.
    """
    from PIL import Image

    side = max(img.size)
    if img.size == (side, side):
        return img
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


def _to_images(messages: list[Message]) -> list[Any]:
    from PIL import Image

    out = []
    for message in messages:
        for raw in message.images:
            with Image.open(BytesIO(raw)) as img:
                out.append(letterbox_to_square(img.convert("RGB")))
    return out


def _to_chat(messages: list[Message], n_images: int) -> list[dict]:
    """Flatten to MiniCPM's chat shape, with images attached to the first user turn.

    A system message becomes a prefix on that turn: MiniCPM's template does not
    take a system role, and silently dropping it would discard the
    answer-only-from-evidence instruction that makes synthesis safe to ship.
    """
    system = " ".join(m.content for m in messages if m.role == "system").strip()
    user = "\n\n".join(m.content for m in messages if m.role != "system").strip()
    text = f"{system}\n\n{user}".strip() if system else user
    content: list[dict] = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _generate_sync(messages: list[Message], config: GenerateConfig) -> str:
    import torch

    model, processor = _load()
    device = _device()
    images = _to_images(messages)
    chat = _to_chat(messages, len(images))

    text = processor.apply_chat_template(chat, add_generation_prompt=True)
    if not isinstance(text, str):  # some processor versions return token ids
        text = processor.decode(text)
    inputs = processor(
        text=[text], images=[images] if images else None, return_tensors="pt"
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=config.max_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature or None,
            num_beams=1,
            use_cache=True,
        )
    prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    # Decode only the continuation — the prompt carries the whole evidence
    # pack, and echoing it back would swamp the answer.
    return processor.batch_decode(
        generated[:, prompt_len:], skip_special_tokens=True
    )[0].strip()


class MiniCPMProvider(ModelProvider):
    """Local MiniCPM-V-4.6 for opt-in synthesis over an evidence pack + frames."""

    async def generate(
        self, messages: list[Message], config: GenerateConfig
    ) -> AsyncIterator[StreamChunk]:
        # Non-streaming: transformers' generate is blocking, so it runs off the
        # event loop and lands as one chunk. Callers already accumulate chunks.
        text = await asyncio.to_thread(_generate_sync, messages, config)
        yield StreamChunk(delta=text, done=True)
