#!/usr/bin/env python3
"""VLM captioning benchmark — the harness behind MM-RAG-jyq.

Measures per-frame MPS latency, resident memory, download size and caption
output for candidate image-captioning VLMs, on real MM-RAG keyframes drawn from
silent scenes (the case where `summarize.py` emits "No transcript or OCR text
detected.").

Usage
-----
    make bench-vlm-corpus          # build eval/vlm-frames.jsonl from the store
    make bench-vlm                 # bench every candidate, print the table
    python scripts/vlm_bench.py --model florence-base-more   # one candidate
    python scripts/vlm_bench.py --dump-html out/captions.html

MPS constraints (copy these verbatim into pipeline/stages/caption.py — they are
the difference between ~0.3s and ~3s per frame, and getting one wrong reads as
"the model is slow" rather than "the config was wrong"):

  * dtype=torch.float32 — NEVER bfloat16. PyTorch MPS has no optimized bf16
    conv kernels and silently emulates, up to 10x slower. Conv is exactly where
    Florence-2's DaViT vision tower lives.
  * attn_implementation="sdpa" — flash_attention_2 does not exist on MPS.
  * .to("mps") — NOT device_map="auto", which is broken on MPS.
  * num_beams=1 — beam search is ~3x the decoder work for marginal gain.
  * PYTORCH_ENABLE_MPS_FALLBACK=1 as a net, but a *triggered* fallback is a bug,
    not a pass. This script records triggered fallbacks per candidate.

Design notes
------------
One model per process (`--all` re-execs this file per candidate). The MPS
caching allocator does not return memory to the OS, so two models in one
process measures A+B and the second model's resident number is fabricated.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Must be set before torch is imported anywhere in this process.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "eval" / "vlm-frames.jsonl"
EMPTY_SCENE_SUMMARY = "No transcript or OCR text detected."
CAPTION_PROMPT = "Describe this image in one or two factual sentences."

# Round 2 (MM-RAG-58v): transcript-conditioned captioning, VideoRAG-style.
# ``{transcript}`` is filled per frame from the manifest's ``transcript``
# field (build the corpus with ``--build-corpus ... --with-transcript``).
# Florence-2 is a fixed-task model and cannot take this prompt at all — that
# structural limit is half the reason this round exists.
COND_PROMPT = (
    "The transcript of this moment in the video:\n{transcript}\n"
    "Describe this image in one or two factual sentences, grounded in the "
    "transcript where it helps identify what is shown."
)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    key: str
    repo: str
    adapter: str  # "florence" | "chat"
    license: str
    params: str
    prompt: str = CAPTION_PROMPT
    square_pad: bool = False  # letterbox to square before processing


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "florence-base-cap",
        "florence-community/Florence-2-base",
        "florence",
        "MIT",
        "0.23B",
        "<CAPTION>",
    ),
    Candidate(
        "florence-base-detailed",
        "florence-community/Florence-2-base",
        "florence",
        "MIT",
        "0.23B",
        "<DETAILED_CAPTION>",
    ),
    Candidate(
        "florence-base-more",
        "florence-community/Florence-2-base",
        "florence",
        "MIT",
        "0.23B",
        "<MORE_DETAILED_CAPTION>",
    ),
    Candidate(
        "florence-large-more",
        "florence-community/Florence-2-large",
        "florence",
        "MIT",
        "0.77B",
        "<MORE_DETAILED_CAPTION>",
    ),
    Candidate("smolvlm-500m", "HuggingFaceTB/SmolVLM-500M-Instruct", "chat", "Apache-2.0", "0.5B"),
    Candidate("qwen3.5-2b", "Qwen/Qwen3.5-2B", "chat", "Apache-2.0", "2B"),
    # square_pad: MiniCPM-V-4.6 on transformers 5.14.1 raises
    # "shape '[3, 1034, 1152]' is invalid for input of size 3575808" on wide
    # (16:9) images — its adaptive slicing miscounts visual tokens by 2. Square
    # inputs work. Every MM-RAG keyframe is 16:9, so it needs letterboxing.
    Candidate(
        "minicpm-v-4.6", "openbmb/MiniCPM-V-4_6", "chat", "Apache-2.0", "1.3B", square_pad=True
    ),
    Candidate("gemma4-e2b", "google/gemma-4-E2B-it", "chat", "Apache-2.0", "5.1B"),
    Candidate("gemma4-e4b", "google/gemma-4-E4B-it", "chat", "Apache-2.0", "8B"),
    # --- round 2 (MM-RAG-58v): text-promptable captioners, with and without
    # transcript conditioning. -cond variants need a --with-transcript corpus.
    Candidate(
        "smolvlm2-2.2b", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "chat", "Apache-2.0", "2.2B"
    ),
    Candidate(
        "smolvlm2-2.2b-cond",
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "chat",
        "Apache-2.0",
        "2.2B",
        COND_PROMPT,
    ),
    Candidate("qwen3.5-2b-cond", "Qwen/Qwen3.5-2B", "chat", "Apache-2.0", "2B", COND_PROMPT),
    Candidate(
        "minicpm-v-4.6-cond",
        "openbmb/MiniCPM-V-4_6",
        "chat",
        "Apache-2.0",
        "1.3B",
        COND_PROMPT,
        square_pad=True,
    ),
)

BY_KEY = {c.key: c for c in CANDIDATES}


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------


class _RUsage4(ctypes.Structure):
    """rusage_info_v4 — uuid then a run of uint64 fields.

    Field 7 after the uuid is ri_phys_footprint, macOS's honest per-process
    number. `ps` RSS double-counts shared pages; ru_maxrss is peak-not-current.

    The slot count is deliberately over-allocated: the kernel writes
    sizeof(rusage_info_v4) bytes into whatever we hand it, so a struct that is
    too small is a heap overflow and a SIGSEGV, not a short read. v4 has ~36
    uint64s; 96 is free insurance against a future v5/v6.
    """

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(96)]


_LIBPROC = None


def phys_footprint_mb() -> float:
    global _LIBPROC
    if _LIBPROC is None:
        path = ctypes.util.find_library("proc")
        _LIBPROC = ctypes.CDLL(path) if path else False
    if not _LIBPROC:
        return float("nan")
    buf = _RUsage4()
    if _LIBPROC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
        return float("nan")
    return buf.f7 / 1e6


def mps_allocated_mb() -> float:
    try:
        import torch

        return torch.mps.driver_allocated_memory() / 1e6
    except Exception:
        return float("nan")


def repo_disk_mb(repo: str) -> float:
    """Measured from the HF cache, not quoted from a model card."""
    try:
        from huggingface_hub import scan_cache_dir

        for r in scan_cache_dir().repos:
            if r.repo_id == repo:
                return r.size_on_disk / 1e6
    except Exception:
        pass
    return float("nan")


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def ocr_word_ratio(text: str) -> float:
    """Fraction of OCR tokens that look like real words (>=3 alpha chars).

    Tesseract on video keyframes emits noise, not text — 'aig sa = a ee Se 2 !'
    scores 0.0, 'SN semen 2S 1-3 yh' scores 0.45 and is still garbage. Used to
    report how much of a frame's "burned-in text" is actually junk.
    """
    import re

    toks = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    if not toks:
        return 0.0
    good = [w for w in toks if len(w) >= 3 and re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", w)]
    return len(good) / len(toks)


def build_corpus(out: Path, limit: int, with_transcript: bool = False) -> int:
    """Select midpoint frames from scenes with no transcript.

    MM-RAG-yzt scopes captioning to frame_idx=0 (the scene midpoint), so this
    corpus is the frame set the future caption stage will act on.

    NOTE on the predicate. The bead specifies "transcript AND OCR both empty",
    matching `summarize.py`'s empty-scene constant. Measured on a real ingest,
    that yields 1 frame out of 89 — not because the visual-only gap is rare
    (38/89 scenes have no transcript at all) but because Tesseract emits noise
    on nearly every frame, so `ocr_text` is almost never empty. Selecting on
    "no transcript" captures the actual failure case; `ocr_ratio` is recorded
    per frame so the decision doc can show how much of that OCR is junk.
    """
    import hashlib
    import sqlite3

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mmrag.config import get_settings

    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    # with_transcript flips the predicate: round 2 (MM-RAG-58v) benches
    # transcript-conditioned captioning, which needs scenes that HAVE speech.
    predicate = "EXISTS" if with_transcript else "NOT EXISTS"
    rows = conn.execute(
        f"""
        SELECT f.id AS frame_id, f.asset_id, s.scene_idx, f.frame_idx,
               f.t_s, f.path, COALESCE(f.ocr_text, '') AS ocr_text,
               a.source_url, a.content_hash, s.summary,
               (SELECT GROUP_CONCAT(t2.text, ' ') FROM transcript_segments t2
                 WHERE t2.scene_id = s.id) AS transcript
          FROM frames f
          JOIN scenes s ON s.id = f.scene_id
          JOIN assets a ON a.id = f.asset_id
         WHERE f.frame_idx = 0
           AND {predicate} (
                 SELECT 1 FROM transcript_segments t
                  WHERE t.scene_id = s.id AND TRIM(t.text) != ''
               )
         ORDER BY f.asset_id, s.scene_idx
        """
    ).fetchall()
    conn.close()

    # Round-robin across assets so a single long video can't dominate the
    # corpus — 40 frames all from one source benchmarks "can it caption *this*
    # video" rather than the general case.
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["asset_id"], []).append(r)
    interleaved = []
    for i in range(max((len(v) for v in grouped.values()), default=0)):
        for g in grouped.values():
            if i < len(g):
                interleaved.append(g[i])
    rows = interleaved

    out.parent.mkdir(parents=True, exist_ok=True)
    written, strict = 0, 0
    by_asset: dict[str, int] = {}
    with out.open("w") as fh:
        for r in rows:
            if written >= limit:
                break
            p = Path(r["path"])
            if not p.exists():
                continue
            if r["summary"] == EMPTY_SCENE_SUMMARY:
                strict += 1
            fh.write(
                json.dumps(
                    {
                        "frame_id": r["frame_id"],
                        "asset_source": r["source_url"],
                        "content_hash": r["content_hash"],
                        "scene_idx": r["scene_idx"],
                        "frame_idx": r["frame_idx"],
                        "t_s": r["t_s"],
                        # Relative to settings.assets_dir — keeps absolute home
                        # paths out of a public repo and makes the manifest
                        # portable across machines and MMRAG_DATA_DIR values.
                        "rel_path": str(p.relative_to(settings.assets_dir)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "ocr_text": r["ocr_text"][:300],
                        "ocr_ratio": round(ocr_word_ratio(r["ocr_text"]), 3),
                        "strict_empty_scene": r["summary"] == EMPTY_SCENE_SUMMARY,
                        "transcript": (r["transcript"] or "")[:500],
                    }
                )
                + "\n"
            )
            by_asset[r["asset_id"]] = by_asset.get(r["asset_id"], 0) + 1
            written += 1

    kind = "with-transcript" if with_transcript else "no-transcript"
    print(f"{kind} midpoint frames: {len(rows)}; written: {written} -> {out}")
    print(f"  of which strictly empty (transcript AND OCR both empty): {strict}")
    for aid, n in by_asset.items():
        print(f"  {aid[:12]}… {n} frames")
    if written < 40:
        print(f"WARNING: only {written} frames (<40). Ingest another video.")
    return written


def load_corpus(manifest: Path, limit: int | None = None) -> list[dict]:
    """Resolve the manifest's assets_dir-relative paths against this machine."""
    if not manifest.exists():
        sys.exit(f"no corpus manifest at {manifest} — run `make bench-vlm-corpus` first")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mmrag.config import get_settings

    assets_dir = get_settings().assets_dir
    rows = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
    drifted = 0
    missing = 0
    keep = []
    for r in rows:
        p = assets_dir / r["rel_path"]
        r["path"] = str(p)  # downstream code and result JSON use an absolute path
        if not p.exists():
            missing += 1
            continue
        # sha256 is a tripwire, not a gate: yt-dlp format selection drifts
        # between runs and would otherwise make this unrunnable later.
        import hashlib

        if hashlib.sha256(p.read_bytes()).hexdigest() != r.get("sha256"):
            drifted += 1
        keep.append(r)
    if drifted:
        print(f"WARNING: {drifted}/{len(keep)} frames differ from the manifest sha256")
    if missing:
        print(
            f"WARNING: {missing}/{len(rows)} frames absent under {assets_dir} — "
            "re-ingest the manifest's asset_source URLs to regenerate them"
        )
    return keep[:limit] if limit else keep


# --------------------------------------------------------------------------
# model adapters
# --------------------------------------------------------------------------


def load_model(cand: Candidate, dtype_name: str = "float32"):
    """fp32 is the comparable default. fp16 is well-supported on MPS and roughly
    halves resident memory — use it for a footnote row on the large candidates,
    which nobody would actually ship in fp32. NEVER bfloat16 (see module docstring).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if dtype_name == "bfloat16":
        raise ValueError("bfloat16 is a trap on MPS — see the module docstring")
    dtype = getattr(torch, dtype_name)
    attn = "sdpa"
    kwargs = dict(dtype=dtype, attn_implementation=attn)
    try:
        model = AutoModelForImageTextToText.from_pretrained(cand.repo, **kwargs)
    except (ValueError, TypeError) as exc:
        # A model whose vision tower lacks the attention interface would
        # otherwise die here; fall back but record it, since a row benched on
        # eager next to rows on sdpa is not a fair comparison.
        print(f"  sdpa rejected ({type(exc).__name__}), retrying default attn")
        attn = "eager-or-default"
        model = AutoModelForImageTextToText.from_pretrained(cand.repo, dtype=dtype)
    model = model.to("mps")
    model.train(False)
    processor = AutoProcessor.from_pretrained(cand.repo)
    return model, processor, attn


def assert_on_mps(model) -> None:
    bad = {p.device.type for p in model.parameters()} - {"mps"}
    if bad:
        raise RuntimeError(f"parameters not on mps: {bad}")


def _open_images(rows: list[dict], square_pad: bool = False):
    from PIL import Image

    imgs = [Image.open(r["path"]).convert("RGB") for r in rows]
    if not square_pad:
        return imgs
    out = []
    for im in imgs:  # letterbox, don't crop — cropping would drop content
        side = max(im.size)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        out.append(canvas)
    return out


def run_batch(
    model, processor, cand: Candidate, images, max_new_tokens: int, rows: list[dict] | None = None
):
    """Returns (generate_seconds, captions, total_new_tokens, preprocess_seconds).

    ``rows`` is only consulted when the candidate's prompt carries a
    ``{transcript}`` placeholder — each frame then gets its own prompt filled
    from the manifest's ``transcript`` field (empty string when absent).
    """
    import torch

    n = len(images)
    conditioned = "{transcript}" in cand.prompt

    def _prompt_for(i: int) -> str:
        if not conditioned:
            return cand.prompt
        transcript = (rows[i].get("transcript") or "") if rows else ""
        return cand.prompt.format(transcript=transcript)

    t_pre = time.perf_counter()
    if cand.adapter == "florence":
        inputs = processor(text=[cand.prompt] * n, images=images, return_tensors="pt")
    else:
        texts = []
        for i in range(n):
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _prompt_for(i)},
                    ],
                }
            ]
            text = processor.apply_chat_template(msgs, add_generation_prompt=True)
            if not isinstance(text, str):  # some processors return token ids
                text = processor.decode(text)
            texts.append(text)
        processor.tokenizer.padding_side = "left"
        try:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        except ValueError:
            # Some processors (Gemma-4) read a flat image list as ONE batch of
            # n images rather than n batches of one, and need explicit nesting.
            # Flat stays primary so already-measured candidates keep their path.
            inputs = processor(
                text=texts,
                images=[[img] for img in images],
                return_tensors="pt",
                padding=True,
            )
    inputs = {k: (v.to("mps") if hasattr(v, "to") else v) for k, v in inputs.items()}
    pre_s = time.perf_counter() - t_pre

    in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
        )
    torch.mps.synchronize()
    gen_s = time.perf_counter() - t0

    if out.device.type != "mps":
        raise RuntimeError(f"generate() output on {out.device.type}, expected mps")

    # Encoder-decoder models (Florence-2/BART) return decoder tokens only;
    # decoder-only chat models return prompt + completion. Subtracting the
    # prompt length from the former yields a negative count.
    enc_dec = bool(getattr(model.config, "is_encoder_decoder", False))
    gen_len = out.shape[1] if enc_dec else max(0, out.shape[1] - in_len)
    new_tokens = int(gen_len * out.shape[0])
    if cand.adapter == "florence":
        raw = processor.batch_decode(out, skip_special_tokens=False)
        caps = [
            str(
                processor.post_process_generation(t, task=cand.prompt, image_size=img.size).get(
                    cand.prompt, ""
                )
            )
            for t, img in zip(raw, images, strict=True)
        ]
    else:
        trimmed = out[:, in_len:]
        caps = [c.strip() for c in processor.batch_decode(trimmed, skip_special_tokens=True)]
    return gen_s, caps, new_tokens, pre_s


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


@dataclass
class BatchResult:
    batch: int
    p50_ms: float
    p95_ms: float
    frames_per_s: float
    preprocess_ms: float
    peak_mb: float


@dataclass
class ModelResult:
    key: str
    repo: str
    license: str
    params: str
    ok: bool = True
    error: str = ""
    attn: str = ""
    dtype: str = "float32"
    disk_mb: float = float("nan")
    load_s: float = float("nan")
    baseline_mb: float = float("nan")
    post_load_mb: float = float("nan")
    post_release_mb: float = float("nan")
    mps_alloc_post_load_mb: float = float("nan")
    mean_output_tokens: float = float("nan")
    distinct_word_ratio: float = float("nan")
    duplicate_captions: int = 0
    mps_fallbacks: list[str] = field(default_factory=list)
    batches: list[BatchResult] = field(default_factory=list)
    captions: list[dict] = field(default_factory=list)


_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "is",
    "are",
    "and",
    "or",
    "with",
    "this",
    "that",
    "it",
    "its",
    "there",
    "image",
    "picture",
    "photo",
}


def _quality_metrics(caps: list[str]) -> tuple[float, int]:
    """Mode-collapse detector: a model emitting the same generic sentence for
    every frame scores near zero. Free, needs no human."""
    toks = [
        w
        for c in caps
        for w in "".join(ch.lower() if ch.isalnum() else " " for ch in c).split()
        if w not in _STOPWORDS
    ]
    ratio = len(set(toks)) / len(toks) if toks else 0.0
    dupes = len(caps) - len(set(c.strip() for c in caps))
    return ratio, dupes


def bench_one(
    cand: Candidate,
    rows: list[dict],
    batches: list[int],
    max_new_tokens: int,
    dtype_name: str = "float32",
) -> ModelResult:
    res = ModelResult(key=cand.key, repo=cand.repo, license=cand.license, params=cand.params)
    res.baseline_mb = phys_footprint_mb()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            t0 = time.perf_counter()
            model, processor, attn = load_model(cand, dtype_name)
            res.load_s = time.perf_counter() - t0
            res.attn = attn
            res.dtype = dtype_name
            assert_on_mps(model)
            res.post_load_mb = phys_footprint_mb()
            res.mps_alloc_post_load_mb = mps_allocated_mb()
            res.disk_mb = repo_disk_mb(cand.repo)

            images = _open_images(rows, cand.square_pad)
            all_caps: list[str] = []
            tok_total, tok_batches = 0, 0

            for b in batches:
                if b > len(images):
                    continue
                warm = images[: min(b, len(images))]
                for _ in range(2):  # MPS compiles kernels per tensor shape
                    run_batch(model, processor, cand, warm, max_new_tokens, rows[: len(warm)])

                per_frame_ms: list[float] = []
                pre_ms: list[float] = []
                peak = res.post_load_mb
                caps_this_pass: list[str] = []
                for i in range(0, len(images), b):
                    chunk = images[i : i + b]
                    gen_s, caps, ntok, pre_s = run_batch(
                        model, processor, cand, chunk, max_new_tokens, rows[i : i + b]
                    )
                    per_frame_ms.extend([gen_s * 1000 / len(chunk)] * len(chunk))
                    pre_ms.append(pre_s * 1000 / len(chunk))
                    caps_this_pass.extend(caps)
                    tok_total += ntok
                    tok_batches += len(chunk)
                    peak = max(peak, phys_footprint_mb())

                ordered = sorted(per_frame_ms)
                res.batches.append(
                    BatchResult(
                        batch=b,
                        p50_ms=statistics.median(ordered),
                        p95_ms=ordered[max(0, int(len(ordered) * 0.95) - 1)],
                        frames_per_s=1000.0 / statistics.median(ordered),
                        preprocess_ms=statistics.median(pre_ms),
                        peak_mb=peak,
                    )
                )
                if b == batches[0]:
                    all_caps = caps_this_pass

            res.mean_output_tokens = tok_total / tok_batches if tok_batches else float("nan")
            res.distinct_word_ratio, res.duplicate_captions = _quality_metrics(all_caps)
            res.captions = [
                {"path": r["path"], "t_s": r["t_s"], "caption": c}
                for r, c in zip(rows, all_caps, strict=False)
            ]

            del model, processor
            gc.collect()
            import torch

            torch.mps.empty_cache()
            res.post_release_mb = phys_footprint_mb()
        except Exception as exc:  # one bad candidate must not kill the sweep
            res.ok = False
            res.error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {res.error}", file=sys.stderr)

        res.mps_fallbacks = sorted(
            {m for rec in caught if "fall back to run on the CPU" in (m := str(rec.message))}
        )
    return res


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _fmt(x: float, nd: int = 1) -> str:
    return "—" if x != x else f"{x:.{nd}f}"


def format_table(results: list[ModelResult]) -> str:
    hdr = (
        "| Model | Params | Disk MB | License | Resident MB | Load s | "
        "p50 ms/frame @B=1 | p50 @B=8 | frames/s best | mean out tok | "
        "distinct-word | dupes | MPS fallbacks |"
    )
    sep = "|" + "---|" * 13
    lines = [hdr, sep]
    for r in results:
        if not r.ok:
            lines.append(
                f"| {r.key} | {r.params} | — | {r.license} | FAILED: {r.error} |" + " — |" * 8
            )
            continue
        b1 = next((b for b in r.batches if b.batch == 1), None)
        b8 = next((b for b in r.batches if b.batch == 8), None)
        best = max((b.frames_per_s for b in r.batches), default=float("nan"))
        lines.append(
            f"| {r.key} | {r.params} | {_fmt(r.disk_mb, 0)} | {r.license} | "
            f"{_fmt(r.post_load_mb - r.baseline_mb, 0)} | {_fmt(r.load_s)} | "
            f"{_fmt(b1.p50_ms) if b1 else '—'} | {_fmt(b8.p50_ms) if b8 else '—'} | "
            f"{_fmt(best, 2)} | {_fmt(r.mean_output_tokens, 1)} | "
            f"{_fmt(r.distinct_word_ratio, 3)} | {r.duplicate_captions} | "
            f"{', '.join(r.mps_fallbacks) if r.mps_fallbacks else 'none'} |"
        )
    return "\n".join(lines)


def format_lifecycle(results: list[ModelResult]) -> str:
    """Feeds MM-RAG-ei7 — does footprint return to baseline after release?"""
    lines = [
        "| Model | baseline MB | post-load MB | peak MB | post-release MB | reclaimed? |",
        "|" + "---|" * 6,
    ]
    for r in results:
        if not r.ok:
            continue
        peak = max((b.peak_mb for b in r.batches), default=float("nan"))
        reclaimed = (
            "yes"
            if r.post_release_mb == r.post_release_mb
            and r.post_release_mb - r.baseline_mb < 0.2 * (r.post_load_mb - r.baseline_mb)
            else "NO"
        )
        lines.append(
            f"| {r.key} | {_fmt(r.baseline_mb, 0)} | {_fmt(r.post_load_mb, 0)} | "
            f"{_fmt(peak, 0)} | {_fmt(r.post_release_mb, 0)} | {reclaimed} |"
        )
    return "\n".join(lines)


def dump_html(results: list[ModelResult], out: Path, n: int) -> None:
    ok = [r for r in results if r.ok and r.captions]
    if not ok:
        sys.exit("no successful results with captions to dump")
    frames = [c["path"] for c in ok[0].captions][:n]
    parts = [
        "<html><head><meta charset='utf-8'><title>VLM caption comparison</title>",
        "<style>body{font:14px system-ui;margin:2rem;max-width:1100px}"
        "img{max-width:560px;border:1px solid #ccc}"
        "td{padding:6px 10px;vertical-align:top;border-bottom:1px solid #eee}"
        "th{text-align:left;padding:6px 10px}</style></head><body>",
        f"<h1>VLM caption comparison — {len(frames)} frames × {len(ok)} models</h1>",
    ]
    for i, path in enumerate(frames):
        parts.append(f"<hr><h3>frame {i + 1}</h3><img src='file://{path}'><table>")
        for r in ok:
            cap = next((c["caption"] for c in r.captions if c["path"] == path), "")
            parts.append(f"<tr><th>{r.key}</th><td>{cap}</td></tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-corpus", type=Path, metavar="OUT")
    ap.add_argument(
        "--with-transcript",
        action="store_true",
        help="corpus from scenes WITH speech (round-2 conditioning)",
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--model", help="bench a single candidate key")
    ap.add_argument("--all", action="store_true", help="subprocess per candidate")
    ap.add_argument("--only", help="comma-separated candidate keys for --all")
    ap.add_argument("--out", type=Path, default=Path("/tmp/mmrag-vlm-bench"))
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--limit", type=int, default=40, help="frames to bench")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--dump-html", type=Path, metavar="OUT")
    ap.add_argument("--report", action="store_true", help="re-print tables from --out")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for c in CANDIDATES:
            print(f"{c.key:24s} {c.repo:45s} {c.adapter:9s} {c.license}")
        return

    if args.build_corpus:
        build_corpus(args.build_corpus, args.limit, args.with_transcript)
        return

    args.out.mkdir(parents=True, exist_ok=True)

    if args.report or args.dump_html:
        results = [
            ModelResult(
                **{
                    **json.loads(p.read_text()),
                    "batches": [
                        BatchResult(**b) for b in json.loads(p.read_text()).get("batches", [])
                    ],
                }
            )
            for p in sorted(args.out.glob("*.json"))
        ]
        if args.dump_html:
            dump_html(results, args.dump_html, 20)
        else:
            print(format_table(results))
            print()
            print(format_lifecycle(results))
        return

    batches = [int(b) for b in args.batches.split(",")]

    if args.model:
        cand = BY_KEY.get(args.model) or sys.exit(f"unknown candidate {args.model}")
        rows = load_corpus(args.manifest, args.limit)
        print(f"== {cand.key} ({cand.repo}) — {len(rows)} frames")
        res = bench_one(cand, rows, batches, args.max_new_tokens, args.dtype)
        suffix = "" if args.dtype == "float32" else f".{args.dtype}"
        (args.out / f"{cand.key}{suffix}.json").write_text(json.dumps(asdict(res), indent=2))
        print(format_table([res]))
        return

    if args.all:
        keys = args.only.split(",") if args.only else [c.key for c in CANDIDATES]
        for key in keys:
            print(f"\n=== {key} ===", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--model",
                    key,
                    "--out",
                    str(args.out),
                    "--manifest",
                    str(args.manifest),
                    "--batches",
                    args.batches,
                    "--limit",
                    str(args.limit),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ],
                check=False,
            )
        subprocess.run([sys.executable, __file__, "--report", "--out", str(args.out)])
        return

    ap.print_help()


if __name__ == "__main__":
    main()
