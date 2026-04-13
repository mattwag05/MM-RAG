from __future__ import annotations

from mmrag.models.mcp_io import AskInput, AskOutput

# M1 stub: returns a valid-shape placeholder. Real evidence-pack assembly
# and gemma4 inference land in M4.


async def handle_ask(inp: AskInput) -> AskOutput:
    return AskOutput(
        answer=(
            "ask() is scaffolded but not yet wired to the retrieval + reasoning "
            "pipeline. Real answers land in milestone M4."
        ),
        evidence=[],
        confidence="low",
    )
