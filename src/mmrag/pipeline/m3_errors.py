"""Typed errors for M3 visual pipeline stages."""

from __future__ import annotations


class OCRError(Exception):
    def __init__(self, *, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


class M3ExtraMissingError(Exception):
    """Raised when an M3 stage runs without the m3-visual extra installed."""

    def __init__(self, *, stage: str) -> None:
        super().__init__(
            f"Stage {stage!r} requires the m3-visual extra. Install with: make sync-m3"
        )
        self.stage = stage
