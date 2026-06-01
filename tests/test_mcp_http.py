from __future__ import annotations

import json

import pytest

from mmrag.config import Settings
from mmrag.mcp_server import (
    SharedBearerTokenVerifier,
    create_mcp_server,
    validate_http_bind,
)


async def test_shared_bearer_token_verifier_accepts_only_expected_token():
    verifier = SharedBearerTokenVerifier("secret-token")

    accepted = await verifier.verify_token("secret-token")
    rejected = await verifier.verify_token("wrong-token")

    assert accepted is not None
    assert accepted.client_id == "mmrag-shared-token"
    assert accepted.scopes == ["mmrag:mcp"]
    assert rejected is None


def test_loopback_http_bind_does_not_require_token():
    validate_http_bind("127.0.0.1", None)
    server = create_mcp_server(host="127.0.0.1", port=8766, path="/mcp")
    assert server.settings.host == "127.0.0.1"
    assert server.settings.streamable_http_path == "/mcp"


def test_non_loopback_http_bind_requires_token():
    with pytest.raises(ValueError, match="MMRAG_MCP_TOKEN"):
        validate_http_bind("0.0.0.0", None)

    server = create_mcp_server(host="0.0.0.0", port=8766, path="/mcp", token="secret")
    assert server.settings.host == "0.0.0.0"


def test_mcp_http_settings_map_from_env(monkeypatch):
    monkeypatch.setenv("MMRAG_MCP_HOST", "100.64.0.10")
    monkeypatch.setenv("MMRAG_MCP_PORT", "9876")
    monkeypatch.setenv("MMRAG_MCP_PATH", "/tailnet-mcp")
    monkeypatch.setenv("MMRAG_MCP_TOKEN", "shared-secret")
    monkeypatch.setenv("MMRAG_MCP_PUBLIC_URL", "http://mmrag.tailnet:9876")

    settings = Settings()

    assert settings.mcp_host == "100.64.0.10"
    assert settings.mcp_port == 9876
    assert settings.mcp_path == "/tailnet-mcp"
    assert settings.mcp_token == "shared-secret"
    assert settings.mcp_public_url == "http://mmrag.tailnet:9876"


def test_streamable_http_app_mounts_mcp_and_discovery_routes():
    server = create_mcp_server(
        host="127.0.0.1",
        port=8766,
        path="/mcp",
        token="secret",
        public_url="http://mmrag.tailnet:8766",
    )
    app = server.streamable_http_app()
    paths = {route.path for route in app.routes}

    assert "/mcp" in paths
    assert "/.well-known/mcp-resource" in paths


@pytest.mark.asyncio
async def test_discovery_route_returns_mcp_resource_metadata():
    server = create_mcp_server(
        host="127.0.0.1",
        port=8766,
        path="/mcp",
        token="secret",
        public_url="http://mmrag.tailnet:8766",
    )
    app = server.streamable_http_app()
    route = next(route for route in app.routes if route.path == "/.well-known/mcp-resource")

    response = await route.endpoint(None)
    payload = json.loads(response.body)

    assert payload["name"] == "mmrag"
    assert payload["transport"] == "streamable-http"
    assert payload["mcp_url"] == "http://mmrag.tailnet:8766/mcp"
    assert payload["auth"]["type"] == "bearer"
    assert payload["auth"]["scope"] == "mmrag:mcp"
    assert payload["tools"] == ["ingest", "ask", "search", "status"]


def test_mcp_tool_registration_stays_four_tool_surface():
    server = create_mcp_server()
    assert set(server._tool_manager._tools) == {"ingest", "ask", "search", "status"}
