"""Stage 6: OCR via the Tesseract CLI.

Runs sequentially across frames with a per-frame 10s hard timeout through
``subprocess_util.run``. On timeout the subprocess wrapper sends SIGTERM
and escalates to SIGKILL, so a slow frame cannot leave a runaway
Tesseract process behind.

PSM 6 ("assume a single uniform block of text") is a reasonable default
for burned-in captions, slides, title cards, and on-screen UI.

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

_PSM = "6"
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
