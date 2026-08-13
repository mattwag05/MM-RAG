from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

EXPECTED_TOOLS = ("ingest", "ask", "search", "densify", "status")
DEFAULT_PUBLIC_URL = "http://127.0.0.1:8766"
DEFAULT_MCP_PATH = "/mcp"
DEFAULT_JOB_ID = "da7c953e-a6db-45e9-bb1e-57237f144ebe"
DEFAULT_ASSET_ID = "b30d0b6f-a449-4837-a9ad-a9f19b6fde38"
DEFAULT_SEARCH_QUERY = "Hestia"
DEFAULT_ASK_QUESTION = "What is this video about?"
DEFAULT_TOKEN_ENVS = ("MMRAG_MCP_TOKEN", "MCP_MMRAG_API_KEY")


class HealthCheckError(RuntimeError):
    """Raised when a deployment health check fails."""


@dataclass(frozen=True)
class HealthCheckConfig:
    public_url: str = DEFAULT_PUBLIC_URL
    mcp_path: str = DEFAULT_MCP_PATH
    token_envs: tuple[str, ...] = DEFAULT_TOKEN_ENVS
    job_id: str = DEFAULT_JOB_ID
    asset_id: str = DEFAULT_ASSET_ID
    search_query: str = DEFAULT_SEARCH_QUERY
    ask_question: str = DEFAULT_ASK_QUESTION
    top_k: int = 3
    timeout_s: float = 15.0


@dataclass(frozen=True)
class HealthCheckSummary:
    discovery_url: str
    mcp_url: str
    token_env: str
    tools: tuple[str, ...]
    job_status: str
    search_hits: int
    ask_evidence: int


def build_urls(public_url: str, mcp_path: str) -> tuple[str, str]:
    base = public_url.rstrip("/")
    path = "/" + mcp_path.strip("/")
    return f"{base}/.well-known/mcp-resource", f"{base}{path}"


def parse_token_envs(value: str) -> tuple[str, ...]:
    envs = tuple(part.strip() for part in value.split(",") if part.strip())
    if not envs:
        raise HealthCheckError("At least one token env var name is required.")
    return envs


def select_token(
    env_names: tuple[str, ...], environ: dict[str, str] | None = None
) -> tuple[str, str]:
    source = environ if environ is not None else os.environ
    for name in env_names:
        value = source.get(name)
        if value:
            return name, value
    names = ", ".join(env_names)
    raise HealthCheckError(f"No MCP bearer token found. Set one of: {names}.")


def validate_discovery(payload: dict[str, Any], expected_mcp_url: str) -> tuple[str, ...]:
    if payload.get("transport") != "streamable-http":
        raise HealthCheckError("Discovery transport is not streamable-http.")
    if payload.get("mcp_url") != expected_mcp_url:
        raise HealthCheckError(
            f"Discovery MCP URL mismatch: expected {expected_mcp_url!r}, "
            f"got {payload.get('mcp_url')!r}."
        )
    auth = payload.get("auth") or {}
    if auth.get("type") != "bearer" or auth.get("scheme") != "Bearer":
        raise HealthCheckError("Discovery auth metadata is not bearer/Bearer.")
    tools = tuple(payload.get("tools") or ())
    if tools != EXPECTED_TOOLS:
        raise HealthCheckError(f"Discovery tool surface mismatch: {tools!r}.")
    return tools


def tool_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if structured:
        return dict(structured)
    for item in getattr(result, "content", ()) or ():
        text = getattr(item, "text", None)
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise HealthCheckError("MCP tool returned non-JSON text content.") from e
            if not isinstance(parsed, dict):
                raise HealthCheckError("MCP tool returned JSON content that was not an object.")
            return parsed
    raise HealthCheckError("MCP tool response did not include structured or JSON text content.")


async def fetch_discovery(discovery_url: str, timeout_s: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.get(discovery_url)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise HealthCheckError("Discovery response was not a JSON object.")
    return payload


async def run_mcp_probe(config: HealthCheckConfig, mcp_url: str, token: str) -> dict[str, Any]:
    async with (
        httpx.AsyncClient(
            headers={"Authorization": "Bearer " + token},
            timeout=config.timeout_s,
        ) as http_client,
        streamable_http_client(mcp_url, http_client=http_client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools_result = await session.list_tools()
        tools = tuple(tool.name for tool in tools_result.tools)
        if set(tools) != set(EXPECTED_TOOLS) or len(tools) != len(EXPECTED_TOOLS):
            raise HealthCheckError(f"MCP tool surface mismatch: {tools!r}.")

        status = tool_payload(await session.call_tool("status", {"job_id": config.job_id}))
        if status.get("status") != "done":
            raise HealthCheckError(f"Expected burn-in job status 'done', got {status!r}.")

        search = tool_payload(
            await session.call_tool(
                "search",
                {
                    "query": config.search_query,
                    "mode": "fts",
                    "asset_id": config.asset_id,
                    "top_k": config.top_k,
                },
            )
        )
        hits = search.get("hits") or []
        if not hits:
            raise HealthCheckError("Scoped search returned no hits.")
        if any(hit.get("asset_id") != config.asset_id for hit in hits):
            raise HealthCheckError("Scoped search returned a hit for a different asset.")

        ask = tool_payload(
            await session.call_tool(
                "ask",
                {
                    "question": config.ask_question,
                    "asset_id": config.asset_id,
                    "synthesize": False,
                },
            )
        )
        evidence = ask.get("evidence") or []
        if ask.get("answer") is not None:
            raise HealthCheckError("Evidence-first ask returned a synthesized answer.")
        if not evidence:
            raise HealthCheckError("Evidence-first ask returned no evidence.")

    return {
        "tools": tools,
        "job_status": status["status"],
        "search_hits": len(hits),
        "ask_evidence": len(evidence),
    }


async def run_health_check(config: HealthCheckConfig) -> HealthCheckSummary:
    discovery_url, mcp_url = build_urls(config.public_url, config.mcp_path)
    token_env, token = select_token(config.token_envs)

    discovery_payload = await fetch_discovery(discovery_url, config.timeout_s)
    discovery_tools = validate_discovery(discovery_payload, mcp_url)

    probe = await run_mcp_probe(config, mcp_url, token)

    return HealthCheckSummary(
        discovery_url=discovery_url,
        mcp_url=mcp_url,
        token_env=token_env,
        tools=discovery_tools,
        job_status=probe["job_status"],
        search_hits=probe["search_hits"],
        ask_evidence=probe["ask_evidence"],
    )


def run_health_check_sync(config: HealthCheckConfig) -> HealthCheckSummary:
    return asyncio.run(run_health_check(config))
