from __future__ import annotations

import hashlib
import html.parser
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from mmrag.models.content_item import ContentItem

_DOCUMENT_EXTS = {
    ".pdf",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".txt",
    ".docx",
}


class DocumentIngestError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def is_document_source(path: str) -> bool:
    return Path(path).suffix.lower() in _DOCUMENT_EXTS


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _chunks(text: str, *, max_chars: int = 1200) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    buf = ""
    for sentence in sentences:
        if len(buf) + len(sentence) + 1 > max_chars and buf:
            out.append(buf.strip())
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        out.append(buf.strip())
    return out


def _html_text(path: Path) -> str:
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return " ".join(parser.parts)


def _docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as e:
        raise DocumentIngestError("document_parse_failed", f"invalid docx: {path}") from e
    root = ElementTree.fromstring(xml)
    parts = [node.text for node in root.iter() if node.text]
    return " ".join(parts)


def _pdf_items(path: Path, asset_id: str) -> list[ContentItem]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise DocumentIngestError(
            "document_dependency_missing",
            "PDF ingestion requires pypdf. Install the mmrag runtime dependencies.",
        ) from e

    items: list[ContentItem] = []
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        for local_idx, chunk in enumerate(_chunks(text)):
            idx = len(items)
            items.append(
                ContentItem(
                    id=f"doc:{asset_id}:p{page_idx}:t{local_idx}",
                    type="text",
                    source_id=asset_id,
                    chunk_idx=idx,
                    asset_id=asset_id,
                    page_idx=page_idx,
                    text=chunk,
                    file_path=str(path),
                    metadata={"document_type": "pdf"},
                )
            )
    return items


async def ingest_document(*, raw_path: str, asset_id: str) -> dict:
    path = Path(raw_path)
    suffix = path.suffix.lower()
    if not path.exists():
        raise DocumentIngestError("document_missing", f"missing document: {path}")
    if suffix == ".pdf":
        items = _pdf_items(path, asset_id)
    elif suffix in {".html", ".htm"}:
        items = _text_items(_html_text(path), path, asset_id, "html")
    elif suffix in {".md", ".markdown"}:
        items = _text_items(
            path.read_text(encoding="utf-8", errors="ignore"), path, asset_id, "markdown"
        )
    elif suffix == ".docx":
        items = _text_items(_docx_text(path), path, asset_id, "docx")
    else:
        items = _text_items(
            path.read_text(encoding="utf-8", errors="ignore"), path, asset_id, "text"
        )
    return {
        "document_items": [item.__dict__ for item in items],
        "document_item_count": len(items),
        "document_type": suffix.lstrip(".") or "generic",
    }


def _text_items(text: str, path: Path, asset_id: str, document_type: str) -> list[ContentItem]:
    out: list[ContentItem] = []
    for idx, chunk in enumerate(_chunks(text)):
        item_type = "table" if "|" in chunk and "---" in chunk else "text"
        out.append(
            ContentItem(
                id=f"doc:{asset_id}:t{idx}",
                type=item_type,
                source_id=asset_id,
                chunk_idx=idx,
                asset_id=asset_id,
                text=chunk,
                file_path=str(path),
                metadata={"document_type": document_type},
            )
        )
    if not out:
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        out.append(
            ContentItem(
                id=f"doc:{asset_id}:empty:{digest}",
                type="generic",
                source_id=asset_id,
                chunk_idx=0,
                asset_id=asset_id,
                file_path=str(path),
                metadata={"document_type": document_type},
            )
        )
    return out
