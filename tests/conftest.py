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


@pytest.fixture(scope="session", autouse=True)
def _media_fixtures() -> None:
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not on PATH; cannot generate media fixtures", allow_module_level=True)
    _generate_fixtures()


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
