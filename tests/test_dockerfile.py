from pathlib import Path


def test_dockerfile_syncs_from_lockfile_with_cpu_torch() -> None:
    dockerfile = Path("Dockerfile").read_text()

    assert "UV_TORCH_BACKEND=cpu" in dockerfile
    assert "uv sync --frozen --extra m3-visual --no-dev --no-cache" in dockerfile
    assert "uv pip install" not in dockerfile
