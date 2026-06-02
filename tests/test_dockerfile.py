from pathlib import Path


def test_dockerfile_syncs_from_lockfile_with_cpu_torch() -> None:
    dockerfile = Path("Dockerfile").read_text()

    assert "uv export --quiet --frozen --extra m3-visual --no-dev" in dockerfile
    assert "--constraints /tmp/constraints.txt" in dockerfile
    assert "--torch-backend cpu" in dockerfile
