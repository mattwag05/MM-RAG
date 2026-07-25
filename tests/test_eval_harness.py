"""Eval harness: metric math + end-to-end scoring against a seeded store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmrag.db.connection import connect
from mmrag.models.mcp_io import SearchHit
from mmrag.ops.evaluate import (
    EvalConfig,
    GoldRef,
    load_dataset,
    resolve_gold_sources,
    run_eval,
    score_hits,
)
from mmrag.pipeline.runner import _persist_scenes, _persist_segments


def _hit(asset_id: str, start_s: float = 0.0, end_s: float = 1.0, stream: str = "fts") -> SearchHit:
    return SearchHit(
        asset_id=asset_id, start_s=start_s, end_s=end_s, score=1.0, source_stream=stream
    )


class TestScoreHits:
    RESOLVED = {"a.mp4": {"a1"}, "b.mp4": {"b1"}}

    def test_gold_at_rank_two(self) -> None:
        hits = [_hit("x"), _hit("a1"), _hit("x"), _hit("a1")]
        relevant, first_rank, stream = score_hits(hits, (GoldRef("a.mp4"),), self.RESOLVED)
        assert relevant == 2
        assert first_rank == 2
        assert stream == "fts"

    def test_no_gold_hit(self) -> None:
        relevant, first_rank, stream = score_hits(
            [_hit("x"), _hit("y")], (GoldRef("a.mp4"),), self.RESOLVED
        )
        assert (relevant, first_rank, stream) == (0, None, None)

    def test_time_range_must_overlap(self) -> None:
        gold = (GoldRef("a.mp4", time_range=(10.0, 20.0)),)
        outside = [_hit("a1", start_s=0.0, end_s=5.0)]
        touching = [_hit("a1", start_s=18.0, end_s=25.0)]
        assert score_hits(outside, gold, self.RESOLVED)[1] is None
        assert score_hits(touching, gold, self.RESOLVED)[1] == 1

    def test_any_gold_ref_counts(self) -> None:
        gold = (GoldRef("a.mp4"), GoldRef("b.mp4"))
        relevant, first_rank, _ = score_hits([_hit("b1")], gold, self.RESOLVED)
        assert (relevant, first_rank) == (1, 1)


class TestLoadDataset:
    def test_roundtrip_and_comments(self, tmp_path: Path) -> None:
        path = tmp_path / "ds.jsonl"
        path.write_text(
            "# comment\n"
            + json.dumps(
                {
                    "id": "q1",
                    "question": "what?",
                    "media": [{"source": "x.mp4", "kind": "file"}],
                    "gold": [{"source": "x.mp4", "time_range": [1.0, 2.0]}],
                    "tags": ["t"],
                }
            )
            + "\n"
        )
        (q,) = load_dataset(path)
        assert q.id == "q1"
        assert q.media == ("x.mp4",)
        assert q.gold == (GoldRef("x.mp4", time_range=(1.0, 2.0)),)

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("# nothing\n")
        with pytest.raises(ValueError, match="empty"):
            load_dataset(path)

    def test_missing_gold_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"id": "q1", "question": "what?", "gold": []}) + "\n")
        with pytest.raises(ValueError, match="no gold"):
            load_dataset(path)


def _seed_asset(asset_id: str, content_hash: str, source_url: str, text: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind, source_url) "
            "VALUES (?, ?, 'file', ?)",
            (asset_id, content_hash, source_url),
        )
    _persist_scenes(asset_id=asset_id, scenes=[{"scene_idx": 0, "start_s": 0.0, "end_s": 4.0}])
    _persist_segments(
        asset_id=asset_id,
        segments=[{"seg_idx": 0, "start_s": 0.0, "end_s": 2.0, "text": text, "scene_idx": 0}],
    )


@pytest.mark.asyncio
async def test_run_eval_end_to_end(isolated_data_dir: Path, tmp_path: Path) -> None:
    _seed_asset("a1", "h1", "tests/fixtures/clip-alpha.mp4", "ordinary weather report today")
    _seed_asset("a2", "h2", "tests/fixtures/clip-beta.mp4", "zebras gallop across the savanna")

    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "zebras gallop",
                "gold": [{"source": "clip-beta.mp4"}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "q2",
                "question": "weather report",
                "gold": [{"source": "clip-alpha.mp4"}],
            }
        )
        + "\n"
    )

    report = await run_eval(EvalConfig(dataset=dataset, top_k=5, mode="fts"))

    assert report.n_questions == 2
    assert report.n_gold_unresolved == 0
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0
    assert report.context_precision_at_k == 1.0
    assert report.latency_p50_ms >= 0
    assert report.latency_p95_ms >= report.latency_p50_ms >= 0
    assert report.stream_attribution == {"fts_transcript": 2}
    assert report.ingest_media_hours_per_min is None


@pytest.mark.asyncio
async def test_run_eval_unresolved_gold_scores_zero(
    isolated_data_dir: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "question": "anything", "gold": [{"source": "missing.mp4"}]})
        + "\n"
    )
    report = await run_eval(EvalConfig(dataset=dataset, top_k=5, mode="fts"))
    assert report.n_gold_unresolved == 1
    assert report.recall_at_k == 0.0


def test_resolve_gold_sources_suffix_match(isolated_data_dir: Path) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind, source_url, mezzanine_path) "
            "VALUES ('a1', 'h1', 'url', 'https://example.org/v/clip.mp4', '/data/mezz/a1.mp4')",
        )
    # File ingests keep only title = path stem (source_url NULL, hashed mezzanine).
    with connect() as conn:
        conn.execute(
            "INSERT INTO assets (id, content_hash, source_kind, title) "
            "VALUES ('a2', 'h2', 'file', 'speech')",
        )
    resolved = resolve_gold_sources({"clip.mp4", "a1.mp4", "speech.mp4", "nope.mp4"})
    assert resolved["clip.mp4"] == {"a1"}
    assert resolved["a1.mp4"] == {"a1"}
    assert resolved["speech.mp4"] == {"a2"}
    assert resolved["nope.mp4"] == set()
