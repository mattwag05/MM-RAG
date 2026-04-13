from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mmrag.config import Settings, reset_settings_for_tests
from mmrag.db.migrations import apply_migrations

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_MP4 = FIXTURES_DIR / "sample.mp4"
SAMPLE_WAV = FIXTURES_DIR / "sample.wav"
SAMPLE_PNG = FIXTURES_DIR / "sample.png"
# Multi-shot: red for 2s then blue for 2s — PySceneDetect's ContentDetector
# should catch the hard cut at ~2s.
MULTISHOT_MP4 = FIXTURES_DIR / "multishot.mp4"
# Speech: synthetic TTS clip produced via `say` (macOS) or `espeak-ng` (Linux).
# Tests that need this fixture skip themselves if neither tool is available.
SPEECH_WAV = FIXTURES_DIR / "speech.wav"
# Speech packaged into an MP4 container with a black video track, so the
# normalize stage has a video stream to transcode and the transcribe stage
# has real speech audio to segment.
SPEECH_MP4 = FIXTURES_DIR / "speech.mp4"
SPEECH_PHRASE = "multimodal retrieval augmented generation test fixture"


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _generate_fixtures() -> None:
    """Generate small deterministic media fixtures via ffmpeg lavfi sources."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLE_MP4.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=3:size=320x240:rate=30",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=3:sample_rate=44100",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(SAMPLE_MP4),
            ],
            check=True,
        )

    if not SAMPLE_WAV.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=3:sample_rate=16000",
                "-ac",
                "1",
                str(SAMPLE_WAV),
            ],
            check=True,
        )

    if not SAMPLE_PNG.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=0.1:size=320x240:rate=10",
                "-frames:v",
                "1",
                str(SAMPLE_PNG),
            ],
            check=True,
        )

    if not MULTISHOT_MP4.exists():
        # Two 2-second solid-color clips concatenated: red → blue.
        # PySceneDetect's ContentDetector (default threshold 27) catches this
        # cleanly because the full-frame color change is about as extreme a
        # content delta as you can produce.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=320x240:r=30:d=2",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=320x240:r=30:d=2",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(MULTISHOT_MP4),
            ],
            check=True,
        )


def _tts_available() -> str | None:
    """Return the name of an available TTS binary, or None."""
    if shutil.which("say") is not None:
        return "say"
    if shutil.which("espeak-ng") is not None:
        return "espeak-ng"
    if shutil.which("espeak") is not None:
        return "espeak"
    return None


def _generate_speech_fixture() -> bool:
    """Try to produce SPEECH_WAV. Returns True on success, False if no TTS tool."""
    if SPEECH_WAV.exists():
        return True
    tool = _tts_available()
    if tool is None:
        return False
    tmp_dir = FIXTURES_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if tool == "say":
        raw = tmp_dir / "_speech_raw.aiff"
        subprocess.run(
            ["say", "-o", str(raw), SPEECH_PHRASE],
            check=True,
        )
    else:  # espeak / espeak-ng
        raw = tmp_dir / "_speech_raw.wav"
        subprocess.run(
            [tool, "-w", str(raw), "-s", "140", SPEECH_PHRASE],
            check=True,
        )

    # Re-encode to 16 kHz mono pcm_s16le to match what the normalize stage
    # would produce, so the transcribe stage sees its native input shape.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(SPEECH_WAV),
        ],
        check=True,
    )
    raw.unlink(missing_ok=True)
    return True


def _generate_speech_mp4() -> bool:
    """Wrap SPEECH_WAV in an MP4 container with a black video track."""
    if SPEECH_MP4.exists():
        return True
    if not _generate_speech_fixture():
        return False
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=30",
            "-i",
            str(SPEECH_WAV),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(SPEECH_MP4),
        ],
        check=True,
    )
    return True


@pytest.fixture(scope="session", autouse=True)
def _media_fixtures() -> None:
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not on PATH; cannot generate media fixtures", allow_module_level=True)
    _generate_fixtures()


@pytest.fixture()
def speech_wav() -> Path:
    """Skip the test if no TTS tool is available to generate speech audio."""
    if not _generate_speech_fixture():
        pytest.skip("no TTS tool (say/espeak-ng/espeak) on PATH to generate speech fixture")
    return SPEECH_WAV


@pytest.fixture()
def speech_mp4() -> Path:
    """Skip the test if no TTS tool is available to wrap speech into an MP4."""
    if not _generate_speech_mp4():
        pytest.skip("no TTS tool available to build speech.mp4 fixture")
    return SPEECH_MP4


@pytest.fixture()
def isolated_data_dir(tmp_path: Path) -> Path:
    """Point the global Settings at a per-test data dir and run migrations."""
    data_dir = tmp_path / "mmrag-data"
    data_dir.mkdir()
    settings = Settings(data_dir=data_dir)
    settings.ensure_dirs()
    reset_settings_for_tests(settings)
    apply_migrations()
    yield data_dir
    reset_settings_for_tests(Settings())  # reset to env defaults
