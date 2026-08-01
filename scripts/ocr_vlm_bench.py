"""Bench Florence-2's OCR head against Tesseract psm 3 (MM-RAG-9wq).

Florence-2-base is already resident in the ingest child for captioning, and it
ships an ``<OCR>`` task head MM-RAG never uses. Tesseract's failure mode on
video keyframes is documented in ``stages/ocr.py``: psm 3 suppresses most
hallucination but still manufactures text on some frames, and it gives up on
short isolated tokens.

MM-RAG-7rm changed what this question means. Source video was silently capped
at 360p, where on-screen text is not in the pixels at all; with that fixed the
interesting comparison is not just Tesseract vs Florence but **both extractors
at both resolutions**, paired on the same timestamps of the same asset.

Scoring avoids hand-labelling by splitting tokens three ways against the
asset's own ASR transcript and the system wordlist:

  redundant  — token appears in the transcript. Real, but OCR adds nothing:
               these are burned-in subtitles ASR already captured.
  novel      — a dictionary word absent from the transcript. This is the
               content OCR exists for (code, UI chrome, slides).
  noise      — neither. Hallucinated texture.

    uv run --extra dev --extra m3-visual python scripts/ocr_vlm_bench.py \
        --mezzanine-1080 /path/to/1080p/mezzanine.mp4 --content-hash 13b96dc3
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

PSM = "3"
FLORENCE_REPO = "florence-community/Florence-2-base"
FLORENCE_TASK = "<OCR>"
BATCH = 8
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_WORDLIST = Path("/usr/share/dict/words")


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower().strip("'-") for m in _TOKEN_RE.finditer(text or "")}


def _frames(db: Path, content_hash_prefix: str) -> list[dict]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT f.id, f.path, f.t_s
          FROM frames f JOIN assets a ON a.id = f.asset_id
         WHERE a.content_hash LIKE ? || '%'
         ORDER BY f.t_s
        """,
        (content_hash_prefix,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows if r["path"] and Path(r["path"]).exists()]


def _transcript_vocab(db: Path, content_hash_prefix: str) -> set[str]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT ts.text FROM transcript_segments ts
          JOIN assets a ON a.id = ts.asset_id
         WHERE a.content_hash LIKE ? || '%'
        """,
        (content_hash_prefix,),
    ).fetchall()
    con.close()
    vocab: set[str] = set()
    for r in rows:
        vocab |= _tokens(r["text"])
    return vocab


def _extract_hi_res(mezzanine: Path, times: list[float], out_dir: Path) -> list[Path]:
    """Pull the same timestamps out of a higher-resolution mezzanine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t_s in enumerate(times):
        out = out_dir / f"hi_{i:04d}.jpg"
        if not out.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t_s:.3f}",
                 "-i", str(mezzanine), "-frames:v", "1", "-q:v", "3", str(out)],
                check=False,
            )
        paths.append(out)
    return [p for p in paths if p.exists()]


def _tesseract(paths: list[Path]) -> tuple[list[str], float]:
    start = time.perf_counter()
    out = []
    for p in paths:
        r = subprocess.run(
            ["tesseract", str(p), "stdout", "--psm", PSM],
            capture_output=True, text=True, timeout=60, check=False,
        )
        out.append(r.stdout.strip())
    return out, (time.perf_counter() - start) / max(len(paths), 1)


def _florence(paths: list[Path]) -> tuple[list[str], float]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # float32 + sdpa: MPS has no optimised bf16 conv kernels and no
    # flash_attention_2, and Florence's vision tower is conv-heavy.
    model = AutoModelForImageTextToText.from_pretrained(
        FLORENCE_REPO, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.train(False)
    processor = AutoProcessor.from_pretrained(FLORENCE_REPO)

    out: list[str] = []
    start = time.perf_counter()
    for i in range(0, len(paths), BATCH):
        chunk = paths[i : i + BATCH]
        images = [Image.open(p).convert("RGB") for p in chunk]
        inputs = processor(text=[FLORENCE_TASK] * len(images), images=images, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=256, num_beams=1, do_sample=False, use_cache=True
            )
        for img, raw in zip(images, processor.batch_decode(gen, skip_special_tokens=False),
                            strict=True):
            parsed = processor.post_process_generation(
                raw, task=FLORENCE_TASK, image_size=img.size
            )
            out.append(" ".join(str(parsed.get(FLORENCE_TASK, "")).split()))
            img.close()
    return out, (time.perf_counter() - start) / max(len(paths), 1)


def _score(texts: list[str], vocab: set[str], words: set[str]) -> dict:
    redundant = novel = noise = 0
    novel_tokens: set[str] = set()
    for t in texts:
        for tok in _tokens(t):
            if tok in vocab:
                redundant += 1
            elif tok in words:
                novel += 1
                novel_tokens.add(tok)
            else:
                noise += 1
    total = redundant + novel + noise
    return {
        "frames_with_text": sum(1 for t in texts if t.strip()),
        "tokens": total,
        "redundant": redundant,
        "novel": novel,
        "noise": noise,
        "noise_share": round(noise / total, 3) if total else 0.0,
        "novel_examples": sorted(novel_tokens)[:30],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / ".local/share/mmrag/mmrag.db"))
    ap.add_argument("--content-hash", required=True, help="asset content_hash prefix")
    ap.add_argument("--mezzanine-1080", default=None, help="higher-res mezzanine of the same asset")
    ap.add_argument("--work-dir", default="/tmp/ocr-vlm-bench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    db = Path(args.db)
    frames = _frames(db, args.content_hash)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print("no frames found", file=sys.stderr)
        return 1
    vocab = _transcript_vocab(db, args.content_hash)
    words = {w.strip().lower() for w in _WORDLIST.open()} if _WORDLIST.exists() else set()
    print(f"{len(frames)} frames | transcript vocab {len(vocab)} | wordlist {len(words)}\n")

    conditions: dict[str, list[Path]] = {"360p": [Path(f["path"]) for f in frames]}
    if args.mezzanine_1080:
        hi = _extract_hi_res(
            Path(args.mezzanine_1080), [f["t_s"] for f in frames], Path(args.work_dir) / "hi"
        )
        if len(hi) == len(frames):
            conditions["1080p"] = hi
        else:
            print(f"warning: extracted {len(hi)}/{len(frames)} hi-res frames; skipping", file=sys.stderr)

    report: dict = {"n_frames": len(frames), "results": {}}
    rows = []
    for res, paths in conditions.items():
        for engine, fn in (("tesseract", _tesseract), ("florence", _florence)):
            texts, per_frame_s = fn(paths)
            scored = _score(texts, vocab, words)
            scored["seconds_per_frame"] = round(per_frame_s, 3)
            report["results"][f"{engine}@{res}"] = scored
            report["results"][f"{engine}@{res}"]["texts"] = texts
            rows.append((f"{engine}@{res}", scored))

    print(f"{'condition':>18} {'frames':>7} {'tokens':>7} {'redund':>7} {'novel':>6} "
          f"{'noise':>6} {'noise%':>7} {'s/frame':>8}")
    for name, s in rows:
        print(f"{name:>18} {s['frames_with_text']:>7} {s['tokens']:>7} {s['redundant']:>7} "
              f"{s['novel']:>6} {s['noise']:>6} {100*s['noise_share']:>6.1f}% "
              f"{s['seconds_per_frame']:>8.2f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\njson: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
