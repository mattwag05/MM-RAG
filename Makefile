# mmrag — convenience wrapper around `uv` that pins the venv to
# `.venv.nosync/`. The `.nosync` suffix tells macOS iCloud Drive to skip the
# directory entirely, which is required when this repo lives under
# ~/Desktop or ~/Documents (both of which iCloud syncs by default).
#
# Without `.nosync`, iCloud sets the macOS `UF_HIDDEN` flag on .pth files,
# which Python 3.13 then silently skips in `site.py` — and your editable
# install becomes invisible. See CLAUDE.md "Gotchas" for the full story.
#
# Always invoke uv via `make` (or export UV_PROJECT_ENVIRONMENT=.venv.nosync
# in your shell). Plain `uv sync` will work but bypass the .nosync trick.

export UV_PROJECT_ENVIRONMENT := .venv.nosync

.PHONY: help sync sync-dev test lint format clean init-db serve-api serve-mcp worker docker-build docker-up

help:
	@echo "Targets:"
	@echo "  make sync        # uv sync (runtime deps only)"
	@echo "  make sync-dev    # uv sync --extra dev (runtime + test deps)"
	@echo "  make test        # uv run pytest -q"
	@echo "  make lint        # uv run ruff check src tests"
	@echo "  make format      # uv run ruff format src tests"
	@echo "  make init-db     # create the SQLite store at MMRAG_DATA_DIR"
	@echo "  make serve-api   # FastAPI on :8765"
	@echo "  make serve-mcp   # FastMCP over stdio"
	@echo "  make worker      # drain the job queue"
	@echo "  make clean       # remove venv + caches"
	@echo "  make docker-build"
	@echo "  make docker-up"

sync:
	uv sync

sync-dev:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

init-db:
	uv run mmrag init-db

serve-api:
	uv run mmrag serve-api

serve-mcp:
	uv run mmrag serve-mcp

worker:
	uv run mmrag worker

clean:
	rm -rf .venv.nosync .pytest_cache .ruff_cache

docker-build:
	docker build -t mmrag:0.1.0 .

docker-up:
	docker compose up
