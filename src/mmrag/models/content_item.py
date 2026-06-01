from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ContentItemType = Literal[
    "text",
    "image",
    "table",
    "equation",
    "video_segment",
    "audio_segment",
    "generic",
]


@dataclass(frozen=True)
class ContentItem:
    id: str
    type: ContentItemType
    source_id: str
    chunk_idx: int
    asset_id: str
    page_idx: int | None = None
    scene_id: int | None = None
    frame_id: int | None = None
    segment_id: int | None = None
    start_s: float | None = None
    end_s: float | None = None
    text: str | None = None
    caption: str | None = None
    file_path: str | None = None
    metadata: dict = field(default_factory=dict)
