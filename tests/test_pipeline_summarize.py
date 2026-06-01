from __future__ import annotations

import pytest

from mmrag.pipeline.stages.summarize import summarize


@pytest.mark.asyncio
async def test_summarize_distills_transcript_and_ocr_by_scene() -> None:
    out = await summarize(
        scenes=[
            {"scene_idx": 0, "start_s": 0.0, "end_s": 4.0},
            {"scene_idx": 1, "start_s": 4.0, "end_s": 8.0},
        ],
        segments=[
            {
                "scene_idx": 0,
                "seg_idx": 0,
                "start_s": 0.5,
                "end_s": 2.0,
                "text": "The host introduces the recipe.",
            },
            {
                "scene_idx": 1,
                "seg_idx": 1,
                "start_s": 4.5,
                "end_s": 6.0,
                "text": "The timer is set for ten minutes.",
            },
        ],
        frames=[
            {
                "scene_idx": 0,
                "frame_idx": 0,
                "t_s": 1.0,
                "ocr_text": "WELCOME",
            },
            {
                "scene_idx": 1,
                "frame_idx": 0,
                "t_s": 5.0,
                "ocr_text": "10:00",
            },
        ],
    )

    assert out == {
        "summaries": [
            {
                "scene_idx": 0,
                "summary": "Spoken: The host introduces the recipe. Visible text: WELCOME",
            },
            {
                "scene_idx": 1,
                "summary": "Spoken: The timer is set for ten minutes. Visible text: 10:00",
            },
        ]
    }


@pytest.mark.asyncio
async def test_summarize_marks_empty_scenes_without_inference() -> None:
    out = await summarize(
        scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 4.0}],
        segments=[],
        frames=[],
    )

    assert out["summaries"] == [{"scene_idx": 0, "summary": "No transcript or OCR text detected."}]
