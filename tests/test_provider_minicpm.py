"""MiniCPM synthesize provider wiring (MM-RAG-thx).

Deliberately does NOT load the model — it is ~5.9 GB resident and the suite has
to stay in the tens of seconds. What is covered here is everything that can be
wrong without the weights: backend selection, the evidence-first default, the
letterbox workaround, and the chat flattening that carries the system prompt.
The model itself was verified by running it end to end on a real keyframe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.handlers.ask import _frame_bytes, _provider
from mmrag.models.mcp_io import Evidence
from mmrag.providers.base import Message
from mmrag.providers.minicpm import _to_chat, letterbox_to_square
from mmrag.providers.ollama import OllamaProvider


def test_default_provider_is_ollama(isolated_data_dir: Path) -> None:
    """Evidence-first stays the default; nobody gets a 5.9 GB model by accident."""
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir))
    assert isinstance(_provider(), OllamaProvider)


def test_minicpm_selected_only_when_configured(isolated_data_dir: Path) -> None:
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir, synthesize_provider="minicpm"))
    from mmrag.providers.minicpm import MiniCPMProvider

    assert isinstance(_provider(), MiniCPMProvider)


@pytest.mark.m3_visual  # constructs a PIL image, so it needs the extra
def test_letterbox_pads_a_wide_frame_to_square_without_cropping() -> None:
    """MiniCPM raises on 16:9 input (its slicing miscounts visual tokens), and
    every MM-RAG keyframe is 16:9. Cropping to square would throw away a third
    of the picture, so the whole frame has to survive the pad."""
    from PIL import Image

    img = Image.new("RGB", (640, 360), (255, 0, 0))
    out = letterbox_to_square(img)

    assert out.size == (640, 640)
    # The original pixels are centred and intact...
    assert out.getpixel((320, 320)) == (255, 0, 0)
    # ...and the padding is black, not stretched content.
    assert out.getpixel((320, 5)) == (0, 0, 0)
    assert out.getpixel((320, 634)) == (0, 0, 0)


@pytest.mark.m3_visual
def test_letterbox_leaves_a_square_image_alone() -> None:
    from PIL import Image

    img = Image.new("RGB", (256, 256), (0, 255, 0))
    assert letterbox_to_square(img) is img


def test_chat_flattening_keeps_the_system_prompt() -> None:
    """MiniCPM's template takes no system role. Dropping it would discard the
    answer-only-from-evidence instruction that makes synthesis safe to ship."""
    chat = _to_chat(
        [
            Message(role="system", content="Answer only from the evidence."),
            Message(role="user", content="Question: what happens?"),
        ],
        n_images=2,
    )

    assert len(chat) == 1 and chat[0]["role"] == "user"
    text_parts = [c["text"] for c in chat[0]["content"] if c["type"] == "text"]
    assert len(text_parts) == 1
    assert "Answer only from the evidence." in text_parts[0]
    assert "Question: what happens?" in text_parts[0]
    # One image placeholder per attached frame, before the text.
    assert sum(1 for c in chat[0]["content"] if c["type"] == "image") == 2


def _evidence(frame_path: str | None) -> Evidence:
    return Evidence(asset_id="a", start_s=0.0, end_s=1.0, frame_path=frame_path)


def test_frame_bytes_dedups_and_respects_the_limit(tmp_path: Path) -> None:
    """Peak memory scales with image count, so the cap is a memory guard."""
    paths = []
    for i in range(4):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(bytes([i]) * 8)
        paths.append(str(p))
    # Same frame twice: two evidence items can share a scene's representative.
    evidence = [_evidence(paths[0]), _evidence(paths[0]), *(_evidence(p) for p in paths[1:])]

    assert len(_frame_bytes(evidence, limit=2)) == 2
    assert len(_frame_bytes(evidence, limit=10)) == 4  # 4 distinct, not 5


def test_frame_bytes_is_empty_without_include_frames() -> None:
    """frame_path is None unless the caller opted in, which is the text-only
    path every provider already handled."""
    assert _frame_bytes([_evidence(None), _evidence(None)], limit=4) == []


def test_frame_bytes_survives_a_missing_file(tmp_path: Path) -> None:
    good = tmp_path / "good.bin"
    good.write_bytes(b"x")
    out = _frame_bytes([_evidence(str(tmp_path / "gone.jpg")), _evidence(str(good))], limit=4)
    assert out == [b"x"]


@pytest.mark.asyncio
async def test_ollama_path_never_attaches_images(isolated_data_dir: Path, monkeypatch) -> None:
    """Ollama's chat endpoint ignores images; attaching them would be silent
    waste and would change the prompt the default backend receives."""
    reset_settings_for_tests(Settings(data_dir=isolated_data_dir))
    from mmrag.handlers import ask as ask_mod

    seen: dict = {}

    class _Recorder:
        async def generate(self, messages, config):
            seen["images"] = [len(m.images) for m in messages]
            yield type("C", (), {"delta": "ok", "done": True})()

    monkeypatch.setattr(ask_mod, "_provider", lambda: _Recorder())
    from mmrag.models.mcp_io import AskInput

    await ask_mod._generate_answer(
        AskInput(question="q", synthesize=True), [_evidence("/tmp/whatever.jpg")]
    )
    assert seen["images"] == [0, 0]
