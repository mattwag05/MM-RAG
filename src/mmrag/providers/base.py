from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    images: list[bytes] = field(default_factory=list)  # raw bytes; provider encodes


@dataclass
class GenerateConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: float = 120.0


@dataclass
class StreamChunk:
    delta: str
    done: bool = False


class ModelProvider(ABC):
    """Abstraction over multimodal text-generation backends.

    M1 ships only the OllamaProvider shell. The interface is the slot that
    a future LLaVA-Video or `gemma4:video` provider will plug into without
    touching the call sites.
    """

    @abstractmethod
    async def generate(
        self, messages: list[Message], config: GenerateConfig
    ) -> AsyncIterator[StreamChunk]:
        ...
