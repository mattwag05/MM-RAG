"""Measure Tesseract OCR output against input frame resolution (MM-RAG-7rm).

MM-RAG never rescales: normalize.py transcodes without a scale filter and
frame_sample.py extracts at the mezzanine's native size, so whatever the
source was is what Tesseract sees. On the reference assets that is 640x360 —
below the 1024px width other tools bump to for on-screen text — which makes
"does upscaling recover text?" a real question rather than a theoretical one.

Runs the OCR stage's exact invocation (`tesseract <path> stdout --psm 3`) at
several scale factors over every frame on disk for the given assets, and
reports where the conditions disagree so the disagreements can be eyeballed.

    uv run --extra dev --extra m3-visual python scripts/ocr_resolution_bench.py \
        --scales 1,2,3 --json /tmp/ocr-res.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PSM = "3"
TIMEOUT_S = 30.0
_TOKEN_MIN_LEN = 2


def _frames(db_path: Path, limit: int | None) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT f.id, f.path, f.ocr_text, f.width, f.height, a.content_hash
          FROM frames f JOIN assets a ON a.id = f.asset_id
         ORDER BY a.content_hash, f.id
        """
    ).fetchall()
    con.close()
    out = [dict(r) for r in rows if r["path"] and Path(r["path"]).exists()]
    return out[:limit] if limit else out


def _ocr_at_scale(args: tuple[str, int]) -> str:
    """OCR one frame at an integer scale factor. Scale 1 runs on the file as-is."""
    path, scale = args
    if scale == 1:
        target = path
        tmp_path = None
    else:
        from PIL import Image

        fd, tmp_name = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        tmp_path = Path(tmp_name)
        with Image.open(path) as img:
            # LANCZOS: the OCR stage would pay this cost once per frame, and a
            # cheap resampler would confound "resolution helps" with "the
            # resampler smeared the glyphs".
            img.resize((img.width * scale, img.height * scale), Image.LANCZOS).save(tmp_path)
        target = str(tmp_path)
    try:
        proc = subprocess.run(
            ["tesseract", target, "stdout", "--psm", PSM],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
        return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in text.split() if len(t) >= _TOKEN_MIN_LEN}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / ".local/share/mmrag/mmrag.db"))
    ap.add_argument("--scales", default="1,2", help="comma-separated integer scale factors")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    frames = _frames(Path(args.db), args.limit)
    if not frames:
        print("no frames on disk", file=sys.stderr)
        return 1
    sizes = {f"{f['width']}x{f['height']}" for f in frames}
    print(f"{len(frames)} frames, native sizes: {sorted(sizes)}")

    results: dict[int, list[str]] = {}
    with ProcessPoolExecutor() as pool:
        for scale in scales:
            texts = list(pool.map(_ocr_at_scale, [(f["path"], scale) for f in frames]))
            results[scale] = texts
            n_text = sum(1 for t in texts if t)
            n_chars = sum(len(t) for t in texts)
            n_tokens = sum(len(_tokens(t)) for t in texts)
            print(
                f"scale {scale}x  frames_with_text={n_text:4d}/{len(frames)}  "
                f"chars={n_chars:7d}  tokens={n_tokens:6d}"
            )

    base = scales[0]
    report: dict = {"n_frames": len(frames), "scales": scales, "per_scale": {}, "disagreements": {}}
    for scale in scales:
        texts = results[scale]
        report["per_scale"][scale] = {
            "frames_with_text": sum(1 for t in texts if t),
            "chars": sum(len(t) for t in texts),
            "tokens": sum(len(_tokens(t)) for t in texts),
        }
    # Full per-frame text, so a follow-up pass can score token *value* (real
    # vs redundant-with-transcript vs noise) without paying for OCR again.
    report["texts"] = {
        str(scale): [
            {"id": f["id"], "path": f["path"], "text": t}
            for f, t in zip(frames, results[scale], strict=True)
        ]
        for scale in scales
    }

    for scale in scales[1:]:
        gained, lost, shared_tokens, only_hi, only_lo = [], [], 0, 0, 0
        for frame, lo, hi in zip(frames, results[base], results[scale], strict=True):
            if hi and not lo:
                gained.append({"id": frame["id"], "path": frame["path"], "text": hi[:300]})
            if lo and not hi:
                lost.append({"id": frame["id"], "path": frame["path"], "text": lo[:300]})
            tl, th = _tokens(lo), _tokens(hi)
            shared_tokens += len(tl & th)
            only_lo += len(tl - th)
            only_hi += len(th - tl)
        report["disagreements"][scale] = {
            "frames_gained_text": len(gained),
            "frames_lost_text": len(lost),
            "tokens_shared": shared_tokens,
            f"tokens_only_at_{base}x": only_lo,
            f"tokens_only_at_{scale}x": only_hi,
            "gained_samples": gained[:25],
            "lost_samples": lost[:25],
        }
        print(
            f"\n{base}x -> {scale}x: frames that gained text={len(gained)}, "
            f"lost text={len(lost)}\n"
            f"  tokens shared={shared_tokens}, only@{base}x={only_lo}, only@{scale}x={only_hi}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\njson: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
