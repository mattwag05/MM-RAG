from __future__ import annotations

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from mmrag import __version__
from mmrag.config import get_settings
from mmrag.handlers.ask import handle_ask
from mmrag.handlers.ingest import handle_ingest
from mmrag.handlers.search import handle_search
from mmrag.handlers.status import JobNotFound, handle_status
from mmrag.logging import configure_logging
from mmrag.models.mcp_io import (
    AskInput,
    AskOutput,
    IngestInput,
    IngestOutput,
    SearchInput,
    SearchOutput,
    StatusInput,
    StatusOutput,
)

configure_logging()

_MCP_SCOPE = "mmrag:mcp"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class SharedBearerTokenVerifier(TokenVerifier):
    def __init__(self, expected_token: str):
        self.expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != self.expected_token:
            return None
        return AccessToken(token=token, client_id="mmrag-shared-token", scopes=[_MCP_SCOPE])


def is_loopback_host(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def validate_http_bind(host: str, token: str | None) -> None:
    if not is_loopback_host(host) and not token:
        raise ValueError("MMRAG_MCP_TOKEN is required when serving MCP HTTP on a non-loopback host")


def _base_url(*, host: str, port: int, public_url: str | None) -> str:
    if public_url:
        return public_url.rstrip("/")
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}"


def _mcp_url(*, host: str, port: int, path: str, public_url: str | None) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{_base_url(host=host, port=port, public_url=public_url)}{normalized_path}"


def _auth_settings(resource_url: str) -> AuthSettings:
    return AuthSettings(
        issuer_url=resource_url,
        resource_server_url=resource_url,
        required_scopes=[_MCP_SCOPE],
    )


def _register_tools(server: FastMCP) -> None:
    @server.tool()
    async def ingest(
        source: str,
        wait_ms: int = 30000,
        push_to_sbt: bool = False,
    ) -> dict:
        """Ingest a public URL or local file. Sync-if-fast (within wait_ms),
        async-if-slow. Returns a job_id you can poll with `status`."""
        inp = IngestInput(source=source, wait_ms=wait_ms, push_to_sbt=push_to_sbt)
        out: IngestOutput = await handle_ingest(inp)
        return out.model_dump()

    @server.tool()
    async def ask(
        question: str,
        asset_id: str | None = None,
        time_range: list[float] | None = None,
        top_k: int = 5,
        model: str = "gemma4:e4b",
        synthesize: bool = False,
        include_frames: bool = False,
    ) -> dict:
        """Answer a natural-language question about an ingested asset (or the
        whole library). By default returns retrieved evidence only; set
        synthesize=true to ask the configured reasoning backend for an answer.
        Set include_frames=true to get local frame JPEG paths on each evidence
        item so you can look at the retrieved moments yourself."""
        inp = AskInput(
            question=question,
            asset_id=asset_id,
            time_range=tuple(time_range) if time_range is not None else None,
            top_k=top_k,
            model=model,
            synthesize=synthesize,
            include_frames=include_frames,
        )
        out: AskOutput = await handle_ask(inp)
        return out.model_dump()

    @server.tool()
    async def search(
        query: str,
        asset_id: str | None = None,
        time_range: list[float] | None = None,
        top_k: int = 10,
        mode: str = "hybrid",
        include_frames: bool = False,
    ) -> dict:
        """Search transcripts, OCR, document content, and optional graph context.
        Set include_frames=true to get local frame JPEG paths on each hit."""
        inp = SearchInput(
            query=query,
            asset_id=asset_id,
            time_range=tuple(time_range) if time_range is not None else None,
            top_k=top_k,
            mode=mode,
            include_frames=include_frames,
        )
        out: SearchOutput = await handle_search(inp)
        return out.model_dump()

    @server.tool()
    async def status(job_id: str) -> dict:
        """Poll the status of an ingest job by id."""
        try:
            out: StatusOutput = await handle_status(StatusInput(job_id=job_id))
        except JobNotFound:
            return {
                "status": "error",
                "stage": "queued",
                "progress": 0.0,
                "asset_id": None,
                "error": f"job not found: {job_id}",
            }
        return out.model_dump()


def _register_discovery_route(
    server: FastMCP,
    *,
    host: str,
    port: int,
    path: str,
    public_url: str | None,
    token_required: bool,
) -> None:
    @server.custom_route("/.well-known/mcp-resource", methods=["GET"], include_in_schema=False)
    async def mcp_resource(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "name": "mmrag",
                "version": __version__,
                "transport": "streamable-http",
                "mcp_url": _mcp_url(host=host, port=port, path=path, public_url=public_url),
                "auth": (
                    {
                        "type": "bearer",
                        "header": "Authorization",
                        "scheme": "Bearer",
                        "scope": _MCP_SCOPE,
                    }
                    if token_required
                    else {"type": "none"}
                ),
                "tools": ["ingest", "ask", "search", "status"],
            }
        )


def create_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    path: str = "/mcp",
    token: str | None = None,
    public_url: str | None = None,
) -> FastMCP:
    validate_http_bind(host, token)
    normalized_path = path if path.startswith("/") else f"/{path}"
    resource_url = _base_url(host=host, port=port, public_url=public_url)
    server = FastMCP(
        "mmrag",
        host=host,
        port=port,
        streamable_http_path=normalized_path,
        auth=_auth_settings(resource_url) if token else None,
        token_verifier=SharedBearerTokenVerifier(token) if token else None,
    )
    _register_tools(server)
    _register_discovery_route(
        server,
        host=host,
        port=port,
        path=normalized_path,
        public_url=public_url,
        token_required=token is not None,
    )
    return server


mcp = create_mcp_server()


def run_stdio() -> None:
    mcp.run()


def run_streamable_http(
    *,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    token: str | None = None,
    public_url: str | None = None,
) -> None:
    settings = get_settings()
    server = create_mcp_server(
        host=host or settings.mcp_host,
        port=port or settings.mcp_port,
        path=path or settings.mcp_path,
        token=token if token is not None else settings.mcp_token,
        public_url=public_url if public_url is not None else settings.mcp_public_url,
    )
    server.run(transport="streamable-http")
