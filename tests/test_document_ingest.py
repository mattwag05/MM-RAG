from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.db.connection import connect
from mmrag.handlers.ingest import handle_ingest
from mmrag.handlers.search import handle_search
from mmrag.models.mcp_io import IngestInput, SearchInput


@pytest.mark.asyncio
async def test_markdown_ingest_projects_content_items_and_searches(
    isolated_data_dir: Path,
) -> None:
    doc = isolated_data_dir / "notes.md"
    doc.write_text(
        "# Launch Notes\n\nThe gamma-ray telescope found blazars.\n\n| name | value |\n| --- | --- |\n| jets | bright |\n",
        encoding="utf-8",
    )

    ingest = await handle_ingest(IngestInput(source=str(doc), wait_ms=120000))

    assert ingest.status == "done"
    assert ingest.asset_id is not None
    assert "gamma-ray telescope" in (ingest.summary or "")

    with connect() as conn:
        rows = conn.execute(
            "SELECT item_type, text FROM content_items WHERE asset_id = ? ORDER BY chunk_idx",
            (ingest.asset_id,),
        ).fetchall()
    assert rows
    assert rows[0]["item_type"] in {"text", "table"}

    out = await handle_search(
        SearchInput(query="blazars", mode="fts", asset_id=ingest.asset_id, top_k=5)
    )
    assert out.hits
    assert out.hits[0].content_item_id is not None
    assert out.hits[0].source_stream == "content_items"


@pytest.mark.asyncio
async def test_hybrid_graph_expands_from_content_item_seed(isolated_data_dir: Path) -> None:
    doc = isolated_data_dir / "graph.md"
    doc.write_text(("alpha seed sentence. " * 90) + "\n\nneighbor context about jets.", encoding="utf-8")

    ingest = await handle_ingest(IngestInput(source=str(doc), wait_ms=120000))
    assert ingest.status == "done"
    assert ingest.asset_id is not None

    out = await handle_search(
        SearchInput(query="neighbor", mode="hybrid_graph", asset_id=ingest.asset_id, top_k=3)
    )

    assert out.hits
    assert any(hit.source_stream == "graph" for hit in out.hits)
