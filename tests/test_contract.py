"""Pydantic schema contract tests for every MCP tool's input/output."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mmrag.models.mcp_io import (
    AskInput,
    AskOutput,
    Evidence,
    IngestInput,
    IngestOutput,
    SearchHit,
    SearchInput,
    SearchOutput,
    StatusInput,
    StatusOutput,
)


class TestIngest:
    def test_minimal_input(self) -> None:
        inp = IngestInput(source="tests/fixtures/sample.mp4")
        assert inp.mode == "standard"
        assert inp.wait_ms == 30000
        assert inp.push_to_sbt is False

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestInput(source="x", surprise="boom")

    def test_wait_ms_bounds(self) -> None:
        with pytest.raises(ValidationError):
            IngestInput(source="x", wait_ms=-1)
        with pytest.raises(ValidationError):
            IngestInput(source="x", wait_ms=10**9)

    def test_output_shapes(self) -> None:
        IngestOutput(status="done", asset_id="a", job_id="j", summary=None)
        IngestOutput(status="in_progress", job_id="j")
        IngestOutput(status="error", error="boom")
        with pytest.raises(ValidationError):
            IngestOutput(status="weird")  # not in Literal


class TestAsk:
    def test_minimal_input(self) -> None:
        inp = AskInput(question="what?")
        assert inp.top_k == 5
        assert inp.model == "gemma4:e4b"
        assert inp.synthesize is False

    def test_evidence_shape(self) -> None:
        out = AskOutput(
            answer=None,
            evidence=[
                Evidence(
                    asset_id="a",
                    start_s=1.0,
                    end_s=2.0,
                    transcript_snippet="hi",
                    source_stream="fts_transcript",
                    snippet="hi",
                    score=0.42,
                )
            ],
            confidence="medium",
        )
        assert out.answer is None
        assert out.evidence[0].start_s == 1.0
        assert out.evidence[0].source_stream == "fts_transcript"


class TestSearch:
    def test_input(self) -> None:
        inp = SearchInput(query="cats")
        assert inp.mode == "hybrid"
        assert inp.top_k == 10

    def test_hit_shape(self) -> None:
        out = SearchOutput(hits=[SearchHit(asset_id="a", start_s=0, end_s=1, score=0.9)])
        assert out.hits[0].score == 0.9


class TestStatus:
    def test_input(self) -> None:
        StatusInput(job_id="abc")

    def test_output(self) -> None:
        out = StatusOutput(status="running", stage="normalize", progress=0.25)
        assert 0 <= out.progress <= 1
        with pytest.raises(ValidationError):
            StatusOutput(status="running", stage="x", progress=2.0)
