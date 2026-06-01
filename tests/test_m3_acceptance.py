"""M3 bead acceptance: an apple/table query lands on the matching visual scene.

Generates a 5-second natural-image clip from a fixture, runs the full
ingest pipeline end-to-end (fetch → normalize → scene_detect → transcribe
→ frame_sample → ocr → embed → summarize), then issues a cross-modal
vector query that should match the visual content via SigLIP.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.migrations import apply_migrations
from mmrag.handlers.ingest import handle_ingest
from mmrag.handlers.search import handle_search
from mmrag.models.mcp_io import IngestInput, SearchInput

pytestmark = pytest.mark.m3_visual

FIXTURES_DIR = Path(__file__).parent / "fixtures"
APPLE_STILL = FIXTURES_DIR / "red_apple_table.jpg"


def _make_apple_clip(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(APPLE_STILL),
            "-t",
            "5",
            "-r",
            "1",
            "-vf",
            "format=yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.asyncio
async def test_natural_image_cross_modal_query(tmp_path):
    try:
        reset_settings_for_tests(Settings(data_dir=tmp_path))
        apply_migrations()

        video_path = tmp_path / "red_apple_table.mp4"
        _make_apple_clip(video_path)

        ingest_result = await handle_ingest(IngestInput(source=str(video_path), wait_ms=120000))
        assert ingest_result.status == "done", f"ingest failed: {ingest_result.error}"
        assert ingest_result.asset_id is not None

        hits_out = await handle_search(
            SearchInput(
                query="a red apple sitting on a wood table",
                asset_id=ingest_result.asset_id,
                top_k=3,
                mode="vector",
            )
        )
        assert len(hits_out.hits) >= 1, "vector query returned no hits"
        top = hits_out.hits[0]
        assert top.asset_id == ingest_result.asset_id
        assert top.source_stream == "vec_frames"
        assert top.score > 0.14, (
            f"SigLIP cosine too low: {top.score}. "
            f"Expected > 0.14 confirming the natural-image visual fixture "
            f"was indexed and retrieved as a meaningful cross-modal match."
        )
    finally:
        reset_settings_for_tests(Settings())
