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


@app.command("serve-mcp-http")
def serve_mcp_http(
    host: str | None = typer.Option(None, help="Override MMRAG_MCP_HOST"),
    port: int | None = typer.Option(None, help="Override MMRAG_MCP_PORT"),
    path: str | None = typer.Option(None, help="Override MMRAG_MCP_PATH"),
    public_url: str | None = typer.Option(None, help="Override MMRAG_MCP_PUBLIC_URL"),
) -> None:
    """Run the MCP server over Streamable HTTP."""
    from mmrag.mcp_server import run_streamable_http

    settings = get_settings()
    settings.ensure_dirs()
    try:
        run_streamable_http(host=host, port=port, path=path, public_url=public_url)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@app.command("worker")
def worker() -> None:
    """Drain the job queue continuously."""
    import asyncio

    from mmrag.worker import run_worker_until_signalled

    settings = get_settings()
    settings.ensure_dirs()
    asyncio.run(run_worker_until_signalled())


@app.command("run-job")
def run_job(job_id: str = typer.Argument(..., help="Job id from the jobs table.")) -> None:
    """Run one queued job in this process, then exit.

    Normally spawned by the worker or the MCP server so pipeline models die
    with the child; also the supported way to drive one job by hand.
    """
    import asyncio

    from mmrag.pipeline.spawn import run_job_in_process

    settings = get_settings()
    settings.ensure_dirs()
    asyncio.run(run_job_in_process(job_id))


@app.command("check-mcp-health")
def check_mcp_health(
    public_url: str = typer.Option(
        "http://127.0.0.1:8766",
        help="Base URL for the deployed MCP HTTP service.",
    ),
    mcp_path: str = typer.Option("/mcp", help="Streamable HTTP MCP path."),
    token_env: str = typer.Option(
        "MMRAG_MCP_TOKEN,MCP_MMRAG_API_KEY",
        help="Comma-separated bearer-token env var names to try, in order.",
    ),
    job_id: str = typer.Option(
        "da7c953e-a6db-45e9-bb1e-57237f144ebe",
        help="Known completed burn-in job id for status validation.",
    ),
    asset_id: str = typer.Option(
        "b30d0b6f-a449-4837-a9ad-a9f19b6fde38",
        help="Known burn-in asset id for scoped search/ask validation.",
    ),
    search_query: str = typer.Option("Hestia", help="Known query for scoped search."),
    ask_question: str = typer.Option(
        "What is this video about?",
        help="Question for evidence-first ask validation.",
    ),
    top_k: int = typer.Option(3, min=1, help="Search result count for validation."),
    timeout_s: float = typer.Option(15.0, min=1.0, help="Network timeout in seconds."),
) -> None:
    """Run the standard post-restart MCP health check."""
    from mmrag.ops.mcp_health import (
        HealthCheckConfig,
        HealthCheckError,
        parse_token_envs,
        run_health_check_sync,
    )

    config = HealthCheckConfig(
        public_url=public_url,
        mcp_path=mcp_path,
        token_envs=parse_token_envs(token_env),
        job_id=job_id,
        asset_id=asset_id,
        search_query=search_query,
        ask_question=ask_question,
        top_k=top_k,
        timeout_s=timeout_s,
    )

    try:
        summary = run_health_check_sync(config)
    except HealthCheckError as e:
        typer.echo(f"MM-RAG MCP health check failed: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo("MM-RAG MCP health check passed")
    typer.echo(f"discovery: {summary.discovery_url}")
    typer.echo(f"mcp: {summary.mcp_url}")
    typer.echo(f"token_env: {summary.token_env}")
    typer.echo(f"tools: {', '.join(summary.tools)}")
    typer.echo(f"status: {summary.job_status}")
    typer.echo(f"search_hits: {summary.search_hits}")
    typer.echo(f"ask_evidence: {summary.ask_evidence}")


@app.command("eval")
def eval_cmd(
    dataset: str = typer.Option("eval/smoke.jsonl", help="Path to a JSONL eval dataset."),
    top_k: int = typer.Option(10, min=1, max=100, help="Hits requested per question."),
    mode: str = typer.Option("hybrid", help="Search mode: hybrid|vector|fts."),
    ingest: bool = typer.Option(
        False, help="Ingest dataset media sources first (slow; local runs only)."
    ),
    sweep: str = typer.Option(
        None,
        help="Sweep one lever, e.g. 'top_k=3,5,10' or 'mode=hybrid,fts,vector'.",
    ),
    json_out: str = typer.Option(None, "--json", help="Also write the report(s) as JSON."),
    fail_under_recall: float = typer.Option(
        None, help="Exit 1 if recall@k falls below this (CI gate)."
    ),
) -> None:
    """Run the deterministic retrieval eval against the configured store."""
    import json
    from pathlib import Path

    from mmrag.ops.evaluate import (
        EvalConfig,
        format_report,
        format_sweep_row,
        run_eval_sync,
    )

    settings = get_settings()
    settings.ensure_dirs()

    base = EvalConfig(dataset=Path(dataset), top_k=top_k, mode=mode, ingest=ingest)
    if sweep is None:
        configs = [base]
        sweep_key = None
    else:
        # ponytail: only top_k/mode are sweepable — extend when graph knobs land in Settings.
        key, _, raw_values = sweep.partition("=")
        sweep_key = key.strip()
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if sweep_key not in ("top_k", "mode") or not values:
            raise typer.BadParameter("--sweep expects 'top_k=3,5,10' or 'mode=hybrid,fts,...'")
        configs = []
        for i, value in enumerate(values):
            override = {sweep_key: int(value) if sweep_key == "top_k" else value}
            # Ingest at most once, on the first run.
            configs.append(
                EvalConfig(
                    dataset=base.dataset,
                    ingest=ingest and i == 0,
                    **{
                        "top_k": base.top_k,
                        "mode": base.mode,
                        **override,
                    },
                )
            )

    reports = []
    failed = False
    for config in configs:
        report = run_eval_sync(config)
        reports.append(report)
        if sweep_key is None:
            typer.echo(format_report(report))
        else:
            typer.echo(format_sweep_row(report, sweep_key))
        if fail_under_recall is not None and report.recall_at_k < fail_under_recall:
            failed = True

    if json_out is not None:
        payload = [r.to_dict() for r in reports]
        Path(json_out).write_text(
            json.dumps(payload[0] if len(payload) == 1 else payload, indent=2)
        )
        typer.echo(f"json report: {json_out}")

    if failed:
        typer.echo(f"recall below threshold {fail_under_recall}", err=True)
        raise typer.Exit(1)


@app.command("version")
def version() -> None:
    """Print the mmrag version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
