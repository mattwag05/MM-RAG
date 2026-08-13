"""Stage 4: transcribe — onnx-asr → transcript segments."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.pipeline.stages import transcribe as transcribe_mod
from mmrag.pipeline.stages.transcribe import transcribe
from tests.conftest import SAMPLE_WAV


@pytest.mark.asyncio
async def test_transcribe_maps_raw_segments_to_scene_indices(monkeypatch) -> None:
    """Stage logic: real shape mapping, speech-to-text primitive faked."""
    fake_raw = [
        {"start": 0.0, "end": 1.2, "text": "  hello world  "},
        {"start": 1.2, "end": 2.5, "text": " testing one two "},
        {"start": 2.6, "end": 3.8, "text": "three four"},
        {"start": 3.9, "end": 4.0, "text": "   "},  # empty after strip — dropped
    ]

    def fake_stt(audio_path: str) -> list[dict]:
        return fake_raw

    monkeypatch.setattr(transcribe_mod, "_run_speech_to_text", fake_stt)

    scenes = [
        {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
        {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
    ]
    result = await transcribe(audio_path=str(SAMPLE_WAV), scenes=scenes)
    segments = result["segments"]

    assert len(segments) == 3
    assert [s["text"] for s in segments] == [
        "hello world",
        "testing one two",
        "three four",
    ]
    assert [s["start_s"] for s in segments] == [0.0, 1.2, 2.6]
    assert [s["seg_idx"] for s in segments] == [0, 1, 2]
    # First two segments start inside scene 0 (< 2.0s), third inside scene 1.
    assert [s["scene_idx"] for s in segments] == [0, 0, 1]


@pytest.mark.asyncio
async def test_transcribe_no_audio_returns_empty() -> None:
    result = await transcribe(audio_path=None, scenes=[])
    assert result["segments"] == []


@pytest.mark.asyncio
async def test_transcribe_audio_path_missing_returns_empty() -> None:
    result = await transcribe(audio_path="/tmp/definitely_not_a_file.wav", scenes=[])
    assert result["segments"] == []


@pytest.mark.asyncio
async def test_transcribe_with_no_scenes_leaves_scene_idx_none(monkeypatch) -> None:
    """Audio-only assets have no scenes — segments should still emit but
    scene_idx stays None so the DB FK can NULL-out cleanly."""

    def fake_stt(audio_path: str) -> list[dict]:
        return [{"start": 0.0, "end": 1.0, "text": "hi"}]

    monkeypatch.setattr(transcribe_mod, "_run_speech_to_text", fake_stt)

    result = await transcribe(audio_path=str(SAMPLE_WAV), scenes=[])
    assert len(result["segments"]) == 1
    assert result["segments"][0]["scene_idx"] is None


@pytest.mark.asyncio
async def test_transcribe_real_asr_on_speech_fixture(speech_wav: Path) -> None:
    """Integration test: real onnx-asr model on a TTS-generated clip.

    Skipped automatically when no TTS tool is available to produce the
    fixture (see conftest.speech_wav fixture). Downloads ~640 MB of
    Parakeet TDT int8 weights plus the 2 MB Silero VAD on first run;
    subsequent runs read them from the shared HF hub cache, alongside
    SigLIP and Florence-2.
    """
    result = await transcribe(audio_path=str(speech_wav), scenes=[])
    segments = result["segments"]
    assert len(segments) >= 1
    joined = " ".join(s["text"].lower() for s in segments)
    # The TTS phrase is "multimodal retrieval augmented generation test
    # fixture". Parakeet returns it verbatim, but stay fuzzy so a TTS voice
    # change on another machine doesn't turn this into a brittle string check.
    assert "test" in joined or "fixture" in joined or "generation" in joined
    # Every segment should carry a positive-length interval.
    for seg in segments:
        assert seg["end_s"] > seg["start_s"]
        assert seg["text"].strip() != ""


_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:01.500 align:start position:0%
hello <c>world</c>

00:00:01.500 --> 00:00:02.500
testing one
two

00:00:02.600 --> 00:00:03.800
three four
"""


@pytest.mark.asyncio
async def test_transcribe_uses_subtitles_when_present(monkeypatch, tmp_path) -> None:
    """A fetched caption track replaces ASR entirely (MM-RAG-8vj)."""

    def fail_stt(audio_path: str) -> list[dict]:
        raise AssertionError("ASR must not run when a subtitle track is present")

    monkeypatch.setattr(transcribe_mod, "_run_speech_to_text", fail_stt)

    vtt = tmp_path / "subs.en.vtt"
    vtt.write_text(_VTT, encoding="utf-8")
    scenes = [
        {"scene_idx": 0, "start_s": 0.0, "end_s": 2.0},
        {"scene_idx": 1, "start_s": 2.0, "end_s": 4.0},
    ]
    result = await transcribe(audio_path=str(SAMPLE_WAV), scenes=scenes, subtitle_path=str(vtt))
    segments = result["segments"]

    assert result["transcript_source"] == "captions"
    assert [s["text"] for s in segments] == ["hello world", "testing one two", "three four"]
    assert [s["start_s"] for s in segments] == [0.0, 1.5, 2.6]
    assert [s["scene_idx"] for s in segments] == [0, 0, 1]


@pytest.mark.asyncio
async def test_transcribe_missing_subtitle_file_falls_back_to_asr(monkeypatch) -> None:
    def fake_stt(audio_path: str) -> list[dict]:
        return [{"start": 0.0, "end": 1.0, "text": "from asr"}]

    monkeypatch.setattr(transcribe_mod, "_run_speech_to_text", fake_stt)
    result = await transcribe(audio_path=str(SAMPLE_WAV), scenes=[], subtitle_path="/tmp/nope.vtt")
    assert result["transcript_source"] == "asr"
    assert [s["text"] for s in result["segments"]] == ["from asr"]
