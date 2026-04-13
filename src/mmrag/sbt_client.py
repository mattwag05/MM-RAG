"""Social Bookmarks Triage REST client. Stub for M1; wired in M5.

In M5 this will POST to {SBT_URL}/api/ingest/multimodal with:
    {
        "url": <source_url>,
        "platform": "threads"|"instagram"|"unknown",
        "mmrag_asset_id": <uuid>,
        "summary": <gemma4 summary>,
        "topTags": [...],
        "transcriptText": <flattened transcript>,
    }
and SBT will upsert a Bookmark + MediaItem keyed on its existing
`postId` hash (base64url of URL path segments)."""

from __future__ import annotations


async def push_to_sbt(payload: dict) -> dict:  # pragma: no cover — M5
    raise NotImplementedError("SBT push lands in M5")
