from __future__ import annotations

import typer

from mmrag import __version__
from mmrag.config import get_settings
from mmrag.logging import get_logger

app = typer.Typer(
    name="mmrag",
    help="Edge-optimized multimodal ingestion tool with an MCP server interface.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("init-db")
def init_db() -> None:
    """Create or migrate the SQLite store at the configured data dir."""
    from mmrag.db.migrations import apply_migrations

    settings = get_settings()
    settings.ensure_dirs()
    log = get_logger("init-db")
    log.info("initializing", db_path=str(settings.db_path))
    apply_migrations()
    log.info("done", db_path=str(settings.db_path))


@app.command("serve-api")
def serve_api(
    host: str | None = typer.Option(None, help="Override MMRAG_API_HOST"),
    port: int | None = typer.Option(None, help="Override MMRAG_API_PORT"),
) -> None:
    """Run the FastAPI REST server."""
    import uvicorn

    settings = get_settings()
    settings.ensure_dirs()
    uvicorn.run(
        "mmrag.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_config=None,
    )


@app.command("serve-mcp")
def serve_mcp() -> None:
    """Run the MCP server over stdio."""
    from mmrag.mcp_server import run_stdio

    settings = get_settings()
    settings.ensure_dirs()
    run_stdio()


@app.command("worker")
def worker() -> None:
    """Drain the job queue continuously."""
    import asyncio

    from mmrag.worker import run_worker

    settings = get_settings()
    settings.ensure_dirs()
    asyncio.run(run_worker())


@app.command("version")
def version() -> None:
    """Print the mmrag version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
