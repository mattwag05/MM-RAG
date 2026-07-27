"""Stage 6: OCR via the Tesseract CLI.

Runs sequentially across frames with a per-frame 10s hard timeout through
``subprocess_util.run``. On timeout the subprocess wrapper sends SIGTERM
and escalates to SIGKILL, so a slow frame cannot leave a runaway
Tesseract process behind.

PSM 3 (fully automatic page segmentation — Tesseract's own default) is
used instead of PSM 6. PSM 6 asserts "there IS a single uniform block of
text here", which on an arbitrary video keyframe forces Tesseract to
manufacture one out of texture and compression artifacts: "ah id santa ae
Beguine ae ©". That junk was indexed into fts_scenes and served in
evidence packs as "Visible text: …".

Measured on 120 real keyframes from two ingested videos (MM-RAG-xvg):

    mode   noise frames emptied   real-text frames kept   real chars kept
    psm 6         39% (17/44)          100% (35/35)            100%
    psm 11         9% ( 4/44)          100% (35/35)            148%
    psm 12         7% ( 3/44)          100% (35/35)            143%
    psm 3         77% (34/44)           71% (25/35)             81%

The sparse-text modes (11/12) are worse, not better — they emit *more*
text on noise frames. PSM 3 is the only mode that meaningfully suppresses
hallucination.

Its 29% loss on "real text" frames is acceptable *for this pipeline*
specifically: what it drops is mostly burned-in subtitles, which duplicate
what faster-whisper already transcribed. What it keeps is on-screen code,
IDE chrome, and slides — content that is never spoken and that OCR is the
only source for.

Per-word confidence filtering was measured and rejected: a conf>=40 floor
cut real-text retention from 100% to 89% (and conf>=60 to 77%), so
Tesseract's confidence is not calibrated well enough on video frames to
gate on. A word-shape ratio filter was also rejected — it scores on-screen
code at 0.22 and IDE breadcrumbs at 0.31, i.e. it penalises exactly the
content PSM 3 is being kept for.

Per-frame OCR failures set ``ocr_text = ""`` and log a structured
warning — they do not fail the stage. A missing Tesseract binary is a
hard error and raises ``OCRError(kind='binary_missing')`` before any
frame runs.
"""

from __future__ import annotations

import shutil

from mmrag.logging import get_logger
from mmrag.pipeline.m3_errors import OCRError
from mmrag.pipeline.subprocess_util import SubprocessFailed, SubprocessTimeout, run

log = get_logger("stage.ocr")

_PSM = "3"
_PER_FRAME_TIMEOUT_S = 10.0
_TESSERACT_AVAILABLE = False


def _ensure_tesseract_available() -> None:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE:
        return
    if shutil.which("tesseract") is None:
        raise OCRError(
            kind="binary_missing",
            message=(
                "tesseract binary not found. Install with: "
                "'brew install tesseract' (macOS) or "
                "'apt install tesseract-ocr' (Debian/Pi)."
            ),
        )
    _TESSERACT_AVAILABLE = True


async def _run_one(path: str) -> str:
    try:
        result = await run(
            ["tesseract", path, "stdout", "--psm", _PSM],
            timeout_s=_PER_FRAME_TIMEOUT_S,
        )
        return result.stdout.strip()
    except SubprocessTimeout:
        log.warning("ocr.timeout", path=path)
        return ""
    except OCRError:
        # Typed hard errors must propagate — never downgrade them to "".
        raise
    except FileNotFoundError:
        log.warning("ocr.file_missing", path=path)
        return ""
    except SubprocessFailed as e:
        log.warning("ocr.failed", path=path, error=str(e))
        return ""
    except Exception as e:  # noqa: BLE001
        log.warning("ocr.failed", path=path, error=str(e))
        return ""


async def ocr(*, frames: list[dict]) -> dict:
    if not frames:
        return {"frames": []}
    _ensure_tesseract_available()

    out: list[dict] = []
    for frame in frames:
        text = await _run_one(frame["path"])
        out.append({**frame, "ocr_text": text})

    log.info("ocr.done", n_frames=len(out))
    return {"frames": out}
