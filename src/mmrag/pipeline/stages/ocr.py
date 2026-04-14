"""Stage 6: OCR via Tesseract / pytesseract.

Runs sequentially across frames with a per-frame 10s timeout via a shared
ThreadPoolExecutor (pytesseract is in-process, spawning Tesseract itself
as a subprocess). PSM 6 ("assume a single uniform block of text") is a
reasonable default for burned-in captions, slides, title cards, and
on-screen UI.

Per-frame OCR failures set ``ocr_text = ""`` and log a warning — they do
not fail the stage. A missing Tesseract binary is a hard error and raises
``OCRError(kind='binary_missing')``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from mmrag.logging import get_logger
from mmrag.pipeline.m3_errors import OCRError

log = get_logger("stage.ocr")

_PSM = "--psm 6"
_PER_FRAME_TIMEOUT_S = 10.0
_OCR_POOL: ThreadPoolExecutor | None = None
_TESSERACT_CHECKED = False


def _ensure_tesseract_available() -> None:
    global _TESSERACT_CHECKED
    if _TESSERACT_CHECKED:
        return
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:  # noqa: BLE001
        raise OCRError(
            kind="binary_missing",
            message=(
                "tesseract binary not found. Install with: "
                "'brew install tesseract' (macOS) or "
                "'apt install tesseract-ocr' (Debian/Pi). "
                f"Original error: {e}"
            ),
        ) from e
    _TESSERACT_CHECKED = True


def _pool() -> ThreadPoolExecutor:
    global _OCR_POOL
    if _OCR_POOL is None:
        _OCR_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
    return _OCR_POOL


def _run_one_sync(path: str) -> str:
    import pytesseract
    from PIL import Image

    with Image.open(path) as img:
        return pytesseract.image_to_string(img, config=_PSM).strip()


async def _run_one(path: str) -> str:
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_pool(), _run_one_sync, path)
    try:
        return await asyncio.wait_for(fut, timeout=_PER_FRAME_TIMEOUT_S)
    except (FuturesTimeout, asyncio.TimeoutError):
        log.warning("ocr.timeout", path=path)
        return ""
    except FileNotFoundError:
        log.warning("ocr.file_missing", path=path)
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
