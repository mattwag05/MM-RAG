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

# Every `uv run` must request the extras explicitly. uv >=0.5 auto-syncs the
# environment to the project's DEFAULT deps on each `uv run`, silently
# UNINSTALLING whatever `make sync-m3` added (torch, open-clip, Pillow, ...).
# The default schema needs sqlite-vec, so sqlite-vec is a core dependency; the
# visual pipeline still needs the m3-visual extra. Keep all run targets on the
# same full extra set so `uv run` does not thrash the venv by repeatedly
# stripping and reinstalling visual dependencies. (uv 0.11 has no `[tool.uv]
# default-extras`, so this can't move into pyproject.toml without demoting the
# extras to non-distributable dependency-groups.)
UV_RUN := uv run --extra dev --extra m3-visual

.PHONY: help sync sync-dev sync-m3 test lint format clean init-db serve-api serve-mcp serve-mcp-http check-mcp worker docker-build docker-up docker-pi-config docker-pi-up docker-pi-down docker-pi-logs

help:
	@echo "Targets:"
	@echo "  make sync        # uv sync (runtime deps only)"
	@echo "  make sync-dev    # uv sync --extra dev (runtime + test deps)"
	@echo "  make sync-m3     # uv sync --extra dev --extra m3-visual (runtime + M3 deps)"
	@echo "  make test        # uv run pytest -q"
	@echo "  make lint        # uv run ruff check src tests"
	@echo "  make format      # uv run ruff format src tests"
	@echo "  make init-db     # create the SQLite store at MMRAG_DATA_DIR"
	@echo "  make serve-api   # FastAPI on :8765"
	@echo "  make serve-mcp   # FastMCP over stdio"
	@echo "  make serve-mcp-http # FastMCP Streamable HTTP on :8766"
	@echo "  make check-mcp   # validate the deployed MCP service"
	@echo "  make worker      # drain the job queue"
	@echo "  make clean       # remove venv + caches"
	@echo "  make docker-build"
	@echo "  make docker-up"
	@echo "  make docker-pi-config # validate Pi/tailnet compose config"
	@echo "  make docker-pi-up     # start MCP + worker Pi/tailnet stack"
	@echo "  make docker-pi-down"
	@echo "  make docker-pi-logs"

sync:
	uv sync

sync-dev:
	uv sync --extra dev

sync-m3:
	uv sync --extra dev --extra m3-visual

test:
	$(UV_RUN) pytest -q

lint:
	$(UV_RUN) ruff check src tests

format:
	$(UV_RUN) ruff format src tests

init-db:
	$(UV_RUN) mmrag init-db

serve-api:
	$(UV_RUN) mmrag serve-api

serve-mcp:
	$(UV_RUN) mmrag serve-mcp

serve-mcp-http:
	$(UV_RUN) mmrag serve-mcp-http

check-mcp:
	$(UV_RUN) mmrag check-mcp-health

worker:
	$(UV_RUN) mmrag worker

clean:
	rm -rf .venv.nosync .pytest_cache .ruff_cache

docker-build:
	docker build -t mmrag:0.1.0 .

docker-up:
	docker compose up

docker-pi-config:
	docker compose -f docker-compose.pi.yml config

docker-pi-up:
	docker compose -f docker-compose.pi.yml up -d

docker-pi-down:
	docker compose -f docker-compose.pi.yml down

docker-pi-logs:
	docker compose -f docker-compose.pi.yml logs -f
