"""Pydantic schema contract tests for every MCP tool's input/output."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mmrag.models.mcp_io import (
    AskInput,
    AskOutput,
    DensifyInput,
    DensifyOutput,
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
        assert inp.wait_ms == 30000

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestInput(source="x", surprise="boom")

    def test_wait_ms_bounds(self) -> None:
        with pytest.raises(ValidationError):
            IngestInput(source="x", wait_ms=-1)
        with pytest.raises(ValidationError):
            IngestInput(source="x", wait_ms=10**9)

    def test_profile_defaults_to_full_and_rejects_unknown(self) -> None:
        assert IngestInput(source="x").profile == "full"
        IngestInput(source="x", profile="transcript_only")
        with pytest.raises(ValidationError):
            IngestInput(source="x", profile="token_burner")

    def test_output_shapes(self) -> None:
        IngestOutput(status="done", asset_id="a", job_id="j", summary=None)
        IngestOutput(status="in_progress", job_id="j")
        IngestOutput(status="error", error="boom")
        with pytest.raises(ValidationError):
            IngestOutput(status="weird")  # not in Literal


class TestDensify:
    def test_minimal_input(self) -> None:
        inp = DensifyInput(asset_id="a", time_range=(1.0, 4.0))
        assert inp.interval_s == 0.5
        assert inp.wait_ms == 60000

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DensifyInput(asset_id="a", time_range=(1.0, 4.0), surprise="boom")

    def test_empty_or_reversed_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DensifyInput(asset_id="a", time_range=(4.0, 1.0))
        with pytest.raises(ValidationError):
            DensifyInput(asset_id="a", time_range=(2.0, 2.0))

    def test_interval_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DensifyInput(asset_id="a", time_range=(1.0, 4.0), interval_s=0.0)
        with pytest.raises(ValidationError):
            DensifyInput(asset_id="a", time_range=(1.0, 4.0), interval_s=60.0)

    def test_output_shapes(self) -> None:
        DensifyOutput(status="done", asset_id="a", job_id="j", frames_added=12)
        DensifyOutput(status="error", error="boom")
        with pytest.raises(ValidationError):
            DensifyOutput(status="weird")


class TestAsk:
    def test_minimal_input(self) -> None:
        inp = AskInput(question="what?")
        assert inp.top_k == 5
        assert inp.model is None
        assert inp.synthesize is False

    def test_model_is_free_form_not_a_gemma_literal(self) -> None:
        # The schema must not advertise a fixed set of Ollama tags: the backend
        # is chosen by MMRAG_SYNTHESIZE_PROVIDER, so any provider's model id has
        # to be expressible here (docs/pmf-rethink.md).
        assert AskInput(question="what?", model="openbmb/MiniCPM-V-4_6").model == (
            "openbmb/MiniCPM-V-4_6"
        )

    def test_reversed_time_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AskInput(question="what?", time_range=(5.0, 1.0))

    def test_evidence_shape(self) -> None:
        out = AskOutput(
            answer=None,
            evidence=[
                Evidence(
                    asset_id="a",
                    content_item_id="doc:a:t0",
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
        assert out.evidence[0].content_item_id == "doc:a:t0"
        assert out.evidence[0].start_s == 1.0
        assert out.evidence[0].source_stream == "fts_transcript"


class TestSearch:
    def test_input(self) -> None:
        inp = SearchInput(query="cats")
        assert inp.mode == "hybrid"
        assert inp.top_k == 10
        graph = SearchInput(query="cats", mode="hybrid_graph")
        assert graph.mode == "hybrid_graph"

    def test_reversed_time_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchInput(query="cats", time_range=(5.0, 1.0))

    def test_hit_shape(self) -> None:
        out = SearchOutput(
            hits=[
                SearchHit(asset_id="a", content_item_id="doc:a:t0", start_s=0, end_s=1, score=0.9)
            ]
        )
        assert out.hits[0].score == 0.9
        assert out.hits[0].content_item_id == "doc:a:t0"


class TestStatus:
    def test_input(self) -> None:
        StatusInput(job_id="abc")

    def test_output(self) -> None:
        out = StatusOutput(status="running", stage="normalize", progress=0.25)
        assert 0 <= out.progress <= 1
        with pytest.raises(ValidationError):
            StatusOutput(status="running", stage="x", progress=2.0)
