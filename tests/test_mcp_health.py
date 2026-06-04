from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmrag.ops.mcp_health import (
    EXPECTED_TOOLS,
    HealthCheckError,
    build_urls,
    parse_token_envs,
    select_token,
    tool_payload,
    validate_discovery,
)


def test_build_urls_uses_base_url_and_mcp_path():
    discovery_url, mcp_url = build_urls("http://mmrag.tailnet:8766/", "/mcp")

    assert discovery_url == "http://mmrag.tailnet:8766/.well-known/mcp-resource"
    assert mcp_url == "http://mmrag.tailnet:8766/mcp"


def test_parse_token_envs_rejects_empty_list():
    assert parse_token_envs(" MMRAG_MCP_TOKEN, MCP_MMRAG_API_KEY ") == (
        "MMRAG_MCP_TOKEN",
        "MCP_MMRAG_API_KEY",
    )

    with pytest.raises(HealthCheckError, match="At least one"):
        parse_token_envs(" , ")


def test_select_token_prefers_first_present_env():
    env = {"MCP_MMRAG_API_KEY": "local", "MMRAG_MCP_TOKEN": "remote"}

    token_env, token = select_token(("MMRAG_MCP_TOKEN", "MCP_MMRAG_API_KEY"), env)

    assert token_env == "MMRAG_MCP_TOKEN"
    assert token == "remote"


def test_select_token_rejects_missing_token():
    with pytest.raises(HealthCheckError, match="No MCP bearer token"):
        select_token(("MMRAG_MCP_TOKEN",), {})


def test_validate_discovery_accepts_expected_metadata():
    tools = validate_discovery(
        {
            "transport": "streamable-http",
            "mcp_url": "http://mmrag.tailnet:8766/mcp",
            "auth": {"type": "bearer", "scheme": "Bearer"},
            "tools": list(EXPECTED_TOOLS),
        },
        "http://mmrag.tailnet:8766/mcp",
    )

    assert tools == EXPECTED_TOOLS


def test_validate_discovery_rejects_wrong_tool_surface():
    with pytest.raises(HealthCheckError, match="tool surface"):
        validate_discovery(
            {
                "transport": "streamable-http",
                "mcp_url": "http://mmrag.tailnet:8766/mcp",
                "auth": {"type": "bearer", "scheme": "Bearer"},
                "tools": ["ask", "search"],
            },
            "http://mmrag.tailnet:8766/mcp",
        )


def test_tool_payload_prefers_structured_content():
    result = SimpleNamespace(structuredContent={"status": "done"}, content=[])

    assert tool_payload(result) == {"status": "done"}


def test_tool_payload_parses_json_text_content():
    result = SimpleNamespace(
        structuredContent=None,
        content=[SimpleNamespace(text='{"answer": null, "evidence": [1, 2]}')],
    )

    assert tool_payload(result) == {"answer": None, "evidence": [1, 2]}


def test_tool_payload_rejects_non_json_text():
    result = SimpleNamespace(structuredContent=None, content=[SimpleNamespace(text="not json")])

    with pytest.raises(HealthCheckError, match="non-JSON"):
        tool_payload(result)

