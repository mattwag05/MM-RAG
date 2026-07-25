"""Deterministic retrieval evaluation harness (Phase 6).

Scores `handle_search` against a JSONL dataset of questions with gold source
citations. All metrics are computed from hit/gold overlap — no LLM judging,
consistent with the evidence-first design (`synthesize=false` everywhere).

Dataset schema (one JSON object per line):

    {
      "id": "lec01-q03",
      "question": "What loss function does the lecture derive?",
      "media": [{"source": "https://...", "kind": "url"}],
      "gold": [{"source": "https://...", "time_range": [812.0, 905.0]}],
      "tags": ["lecture"]
    }

`gold[].source` is matched against `assets.source_url` / `assets.mezzanine_path`
by suffix, so fixture basenames ("speech.mp4") and full URLs both resolve.
A hit is gold-relevant iff its asset matches AND (no time_range, or the hit's
[start_s, end_s] overlaps it — same predicate as ask's time filtering).
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmrag.logging import get_logger

log = get_logger("ops.evaluate")


@dataclass(frozen=True)
class GoldRef:
    source: str
    time_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    gold: tuple[GoldRef, ...]
    media: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalConfig:
    dataset: Path
    top_k: int = 10
    mode: str = "hybrid"
    ingest: bool = False
    ingest_wait_ms: int = 600000


@dataclass
class QuestionResult:
    id: str
    hits: int
    relevant: int
    first_relevant_rank: int | None
    first_relevant_stream: str | None
    latency_ms: float
    gold_resolved: bool


@dataclass
class EvalReport:
    dataset: str
    mode: str
    top_k: int
    n_questions: int
    n_gold_unresolved: int
    recall_at_k: float
    context_precision_at_k: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    stream_attribution: dict[str, int] = field(default_factory=dict)
    ingest_media_hours_per_min: float | None = None
    questions: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "questions"}
        d["questions"] = [q.__dict__ for q in self.questions]
        return d


def load_dataset(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        gold = tuple(
            GoldRef(
                source=g["source"],
                time_range=tuple(g["time_range"]) if g.get("time_range") else None,
            )
            for g in raw["gold"]
        )
        if not gold:
            raise ValueError(f"{path}:{lineno}: question {raw.get('id')!r} has no gold refs")
        questions.append(
            EvalQuestion(
                id=raw["id"],
                question=raw["question"],
                gold=gold,
                media=tuple(m["source"] for m in raw.get("media", [])),
                tags=tuple(raw.get("tags", [])),
            )
        )
    if not questions:
        raise ValueError(f"{path}: dataset is empty")
    return questions


def resolve_gold_sources(sources: set[str]) -> dict[str, set[str]]:
    """Map each gold source string to the set of asset ids it matches.

    Suffix match against source_url and mezzanine_path covers full URLs (a
    string is a suffix of itself). File ingests store neither (source_url is
    NULL, mezzanine_path is a content-hash name), only title = path stem — so
    a gold source's stem matching the title also counts.
    """
    from mmrag.db.connection import connect

    with connect() as conn:
        rows = conn.execute("SELECT id, source_url, mezzanine_path, title FROM assets").fetchall()
    resolved: dict[str, set[str]] = {s: set() for s in sources}
    for row in rows:
        for source in sources:
            matched = any(
                col and col.endswith(source) for col in (row["source_url"], row["mezzanine_path"])
            ) or (row["title"] is not None and row["title"] == Path(source).stem)
            if matched:
                resolved[source].add(row["id"])
    return resolved


def _overlaps(hit: Any, time_range: tuple[float, float] | None) -> bool:
    if time_range is None:
        return True
    return hit.end_s >= time_range[0] and hit.start_s <= time_range[1]


def _is_relevant(hit: Any, gold: tuple[GoldRef, ...], resolved: dict[str, set[str]]) -> bool:
    return any(
        hit.asset_id in resolved.get(g.source, ()) and _overlaps(hit, g.time_range) for g in gold
    )


def score_hits(
    hits: list[Any], gold: tuple[GoldRef, ...], resolved: dict[str, set[str]]
) -> tuple[int, int | None, str | None]:
    """Return (relevant_count, first_relevant_rank (1-based), first_relevant_stream)."""
    relevant = 0
    first_rank: int | None = None
    first_stream: str | None = None
    for rank, hit in enumerate(hits, start=1):
        if _is_relevant(hit, gold, resolved):
            relevant += 1
            if first_rank is None:
                first_rank = rank
                first_stream = hit.source_stream
    return relevant, first_rank, first_stream


def _percentiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return qs[49], qs[94]


async def _ingest_media(questions: list[EvalQuestion], wait_ms: int) -> float | None:
    """Ingest all media sources; return media-hours per wall-minute, or None."""
    from mmrag.db.connection import connect
    from mmrag.handlers.ingest import handle_ingest
    from mmrag.models.mcp_io import IngestInput

    sources = list(dict.fromkeys(s for q in questions for s in q.media))
    if not sources:
        return None
    t0 = time.perf_counter()
    asset_ids: list[str] = []
    for source in sources:
        out = await handle_ingest(IngestInput(source=source, wait_ms=wait_ms))
        log.info("eval.ingest", source=source, status=out.status, asset_id=out.asset_id)
        if out.status != "done":
            log.warning("eval.ingest_incomplete", source=source, status=out.status, error=out.error)
        elif out.asset_id:
            asset_ids.append(out.asset_id)
    wall_min = (time.perf_counter() - t0) / 60.0
    if not asset_ids or wall_min == 0:
        return None
    placeholders = ",".join("?" for _ in asset_ids)
    with connect() as conn:
        row = conn.execute(
            f"SELECT COALESCE(SUM(duration_s), 0) AS total FROM assets WHERE id IN ({placeholders})",
            asset_ids,
        ).fetchone()
    return (float(row["total"]) / 3600.0) / wall_min


async def run_eval(config: EvalConfig) -> EvalReport:
    from mmrag.handlers.search import handle_search
    from mmrag.models.mcp_io import SearchInput

    questions = load_dataset(config.dataset)

    throughput = None
    if config.ingest:
        throughput = await _ingest_media(questions, config.ingest_wait_ms)

    all_sources = {g.source for q in questions for g in q.gold}
    resolved = resolve_gold_sources(all_sources)
    for source, ids in resolved.items():
        if not ids:
            log.warning("eval.gold_unresolved", source=source)

    # Untimed warm-up so model/extension load doesn't pollute latency stats.
    await handle_search(
        SearchInput(query=questions[0].question, top_k=config.top_k, mode=config.mode)
    )

    results: list[QuestionResult] = []
    for q in questions:
        t0 = time.perf_counter()
        out = await handle_search(
            SearchInput(query=q.question, top_k=config.top_k, mode=config.mode)
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        relevant, first_rank, first_stream = score_hits(out.hits, q.gold, resolved)
        results.append(
            QuestionResult(
                id=q.id,
                hits=len(out.hits),
                relevant=relevant,
                first_relevant_rank=first_rank,
                first_relevant_stream=first_stream,
                latency_ms=latency_ms,
                gold_resolved=any(resolved.get(g.source) for g in q.gold),
            )
        )

    n = len(results)
    p50, p95 = _percentiles([r.latency_ms for r in results])
    attribution: dict[str, int] = {}
    for r in results:
        if r.first_relevant_stream is not None:
            attribution[r.first_relevant_stream] = attribution.get(r.first_relevant_stream, 0) + 1
    return EvalReport(
        dataset=str(config.dataset),
        mode=config.mode,
        top_k=config.top_k,
        n_questions=n,
        n_gold_unresolved=sum(1 for r in results if not r.gold_resolved),
        recall_at_k=sum(1 for r in results if r.first_relevant_rank is not None) / n,
        context_precision_at_k=sum(r.relevant / r.hits for r in results if r.hits) / n,
        mrr=sum(1.0 / r.first_relevant_rank for r in results if r.first_relevant_rank) / n,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        stream_attribution=attribution,
        ingest_media_hours_per_min=throughput,
        questions=results,
    )


def run_eval_sync(config: EvalConfig) -> EvalReport:
    return asyncio.run(run_eval(config))


def format_report(report: EvalReport) -> str:
    rows = [
        ("dataset", report.dataset),
        ("mode", report.mode),
        ("top_k", report.top_k),
        ("questions", report.n_questions),
        ("gold unresolved", report.n_gold_unresolved),
        (f"recall@{report.top_k}", f"{report.recall_at_k:.3f}"),
        (f"context precision@{report.top_k}", f"{report.context_precision_at_k:.3f}"),
        ("MRR", f"{report.mrr:.3f}"),
        ("latency p50 (ms)", f"{report.latency_p50_ms:.1f}"),
        ("latency p95 (ms)", f"{report.latency_p95_ms:.1f}"),
        (
            "first-hit streams",
            ", ".join(f"{k}={v}" for k, v in sorted(report.stream_attribution.items())) or "-",
        ),
    ]
    if report.ingest_media_hours_per_min is not None:
        rows.append(("ingest (media-hr/min)", f"{report.ingest_media_hours_per_min:.2f}"))
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


def format_sweep_row(report: EvalReport, key: str) -> str:
    value = report.top_k if key == "top_k" else report.mode
    return (
        f"{key}={value!s:<14} recall={report.recall_at_k:.3f}  "
        f"precision={report.context_precision_at_k:.3f}  mrr={report.mrr:.3f}  "
        f"p95={report.latency_p95_ms:.1f}ms"
    )
