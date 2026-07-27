from __future__ import annotations

from mmrag.config import get_settings
from mmrag.db.connection import connect
from mmrag.logging import get_logger
from mmrag.models.mcp_io import AskInput, AskOutput, Evidence, SearchInput, SearchOutput
from mmrag.providers.base import GenerateConfig, Message

log = get_logger("handler.ask")


def _scene_summaries(scene_ids: list[str | None]) -> dict[str, str]:
    ids = sorted({int(scene_id) for scene_id in scene_ids if scene_id is not None})
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    with connect() as conn:
        rows = conn.execute(
            f"SELECT id, summary FROM scenes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    return {
        str(r["id"]): str(r["summary"])
        for r in rows
        if r["summary"] is not None and str(r["summary"]).strip()
    }


def _scene_captions(scene_ids: list[str | None]) -> dict[str, str]:
    """Ingest-time VLM captions per scene, joined across the scene's frames.

    Only silent scenes with no on-screen text are captioned, so this is
    empty for most hits.
    """
    ids = sorted({int(scene_id) for scene_id in scene_ids if scene_id is not None})
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    with connect() as conn:
        rows = conn.execute(
            f"SELECT scene_id, caption FROM frames "  # noqa: S608 — placeholders only
            f"WHERE scene_id IN ({placeholders}) AND COALESCE(caption,'') <> '' "
            f"ORDER BY frame_idx",
            ids,
        ).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        key = str(r["scene_id"])
        text = str(r["caption"]).strip()
        out[key] = f"{out[key]} {text}".strip() if key in out else text
    return out


def _hit_to_evidence(hit, summaries: dict[str, str], captions: dict[str, str]) -> Evidence:
    snippet = hit.snippet if hit.snippet != "[visual match]" else None
    transcript_snippet = (
        snippet if hit.source_stream in {"fts_transcript", "vec_transcript"} else None
    )
    ocr_snippet = snippet if hit.source_stream in {"fts_scenes", "vec_frames"} else None
    return Evidence(
        caption=captions.get(hit.scene_id),
        asset_id=hit.asset_id,
        content_item_id=hit.content_item_id,
        scene_id=hit.scene_id,
        frame_id=hit.frame_id,
        start_s=hit.start_s,
        end_s=hit.end_s,
        source_stream=hit.source_stream,
        snippet=snippet,
        score=hit.score,
        summary=summaries.get(hit.scene_id),
        ocr_snippet=ocr_snippet,
        transcript_snippet=transcript_snippet,
    )


def _filter_time_range(
    evidence: list[Evidence], time_range: tuple[float, float] | None
) -> list[Evidence]:
    if time_range is None:
        return evidence
    start, end = time_range
    return [ev for ev in evidence if ev.end_s >= start and ev.start_s <= end]


def _evidence_prompt(question: str, evidence: list[Evidence]) -> list[Message]:
    lines = []
    for idx, ev in enumerate(evidence, start=1):
        snippet = ev.snippet or ev.transcript_snippet or ev.ocr_snippet or ev.summary or ""
        lines.append(
            "\n".join(
                [
                    f"[{idx}] asset={ev.asset_id} scene={ev.scene_id} "
                    f"time={ev.start_s:.2f}-{ev.end_s:.2f}s source={ev.source_stream} "
                    f"score={ev.score if ev.score is not None else 'n/a'}",
                    snippet,
                ]
            )
        )
    context = "\n\n".join(lines) if lines else "(no retrieved evidence)"
    return [
        Message(
            role="system",
            content=(
                "Answer only from the retrieved MM-RAG evidence. "
                "If the evidence is insufficient, say what is missing. "
                "Cite evidence by bracket number."
            ),
        ),
        Message(role="user", content=f"Question: {question}\n\nEvidence:\n{context}"),
    ]


async def _generate_answer(inp: AskInput, evidence: list[Evidence]) -> str:
    from mmrag.providers.ollama import OllamaProvider

    settings = get_settings()
    provider = OllamaProvider(settings.ollama_url)
    chunks: list[str] = []
    async for chunk in provider.generate(
        _evidence_prompt(inp.question, evidence),
        GenerateConfig(model=inp.model),
    ):
        chunks.append(chunk.delta)
    return "".join(chunks).strip()


async def handle_ask(inp: AskInput) -> AskOutput:
    from mmrag.handlers.search import handle_search

    search_out: SearchOutput = await handle_search(
        SearchInput(
            query=inp.question,
            asset_id=inp.asset_id,
            top_k=inp.top_k,
            mode="hybrid",
            time_range=inp.time_range,
        )
    )
    scene_ids = [hit.scene_id for hit in search_out.hits]
    summaries = _scene_summaries(scene_ids)
    captions = _scene_captions(scene_ids)
    evidence = _filter_time_range(
        [_hit_to_evidence(hit, summaries, captions) for hit in search_out.hits],
        inp.time_range,
    )
    if not inp.synthesize:
        return AskOutput(answer=None, evidence=evidence, confidence="low")

    try:
        answer = await _generate_answer(inp, evidence)
    except Exception as e:  # noqa: BLE001
        log.warning("ask.synthesis_failed", error=str(e))
        return AskOutput(answer=None, evidence=evidence, confidence="low")

    return AskOutput(
        answer=answer or None,
        evidence=evidence,
        confidence="medium" if answer and evidence else "low",
    )
