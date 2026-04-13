from __future__ import annotations

from collections.abc import AsyncIterator

from mmrag.providers.base import GenerateConfig, Message, ModelProvider, StreamChunk


class OllamaProvider(ModelProvider):
    """Talks to a local Ollama HTTP endpoint. M1 ships the shell only;
    the real httpx-streaming implementation lands in M4 alongside the
    `ask` evidence-pack assembly."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def generate(
        self, messages: list[Message], config: GenerateConfig
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError("OllamaProvider.generate lands in M4")
        # Make this a generator so the type checker accepts the signature.
        if False:  # pragma: no cover
            yield StreamChunk(delta="", done=True)
