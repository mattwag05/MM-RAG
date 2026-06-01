from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request


class SBTClientError(RuntimeError):
    pass


async def push_to_sbt(base_url: str, payload: dict) -> dict:
    return await asyncio.to_thread(_post_json, base_url.rstrip("/") + "/api/ingest/multimodal", payload)


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise SBTClientError(str(e)) from e
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
