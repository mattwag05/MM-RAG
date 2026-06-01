from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from mmrag.providers.base import GenerateConfig, Message, ModelProvider, StreamChunk


class OllamaProvider(ModelProvider):
    """Talks to a local Ollama HTTP endpoint for opt-in answer synthesis."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def generate(
        self, messages: list[Message], config: GenerateConfig
    ) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=config.timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = (data.get("message") or {}).get("content") or data.get("response") or ""
        yield StreamChunk(delta=content, done=True)
