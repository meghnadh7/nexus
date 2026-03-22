"""
Datadog API tool implementations for the mcp-datadog MCP server.

Uses the Datadog REST API v1/v2 with DD-API-KEY + DD-APPLICATION-KEY headers.
All functions are async, retried with tenacity.
"""

import time as _time
from typing import Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_DD_API = "https://api.datadoghq.com/api"
_TIMEOUT = httpx.Timeout(30.0)


def _headers(api_key: str, app_key: str) -> dict[str, str]:
    """Return Datadog API request headers."""
    return {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
        "Content-Type": "application/json",
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def get_alert(api_key: str, app_key: str, alert_id: str) -> dict:
    """
    Fetch full details of a Datadog monitor by ID.

    Args:
        api_key:  Datadog API key.
        app_key:  Datadog application key.
        alert_id: Monitor ID (numeric string).

    Returns:
        Dict containing monitor name, type, query, overall_state, message, tags.
    """
    url = f"{_DD_API}/v1/monitor/{alert_id}"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, headers=_headers(api_key, app_key))
        resp.raise_for_status()
    data = resp.json()
    log.info("datadog_alert_fetched", alert_id=alert_id, name=data.get("name"))
    return {
        "id": str(data["id"]),
        "name": data.get("name", ""),
        "type": data.get("type", ""),
        "query": data.get("query", ""),
        "overall_state": data.get("overall_state", ""),
        "message": data.get("message", ""),
        "tags": data.get("tags", []),
        "created": data.get("created", ""),
        "modified": data.get("modified", ""),
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def query_metrics(
    api_key: str,
    app_key: str,
    metric_query: str,
    from_ts: int,
    to_ts: int,
) -> dict:
    """
    Run a Datadog metrics query and return the series data.

    Args:
        api_key:       Datadog API key.
        app_key:       Datadog application key.
        metric_query:  Datadog query string (e.g. ``avg:system.cpu.user{*}``).
        from_ts:       Start of the query window as a Unix timestamp.
        to_ts:         End of the query window as a Unix timestamp.

    Returns:
        Dict with keys ``query``, ``from_date``, ``to_date``, ``series``
        (list of dicts with ``metric``, ``pointlist``, ``scope``).
    """
    url = f"{_DD_API}/v1/query"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            url,
            headers=_headers(api_key, app_key),
            params={"from": from_ts, "to": to_ts, "query": metric_query},
        )
        resp.raise_for_status()
    data = resp.json()
    return {
        "query": metric_query,
        "from_date": data.get("from_date"),
        "to_date": data.get("to_date"),
        "series": [
            {
                "metric": s.get("metric", ""),
                "scope": s.get("scope", ""),
                "pointlist": s.get("pointlist", []),
            }
            for s in data.get("series", [])
        ],
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def list_active_alerts(api_key: str, app_key: str) -> list[dict]:
    """
    List all monitors currently in Alert or Warn state with env:production tag.

    Args:
        api_key: Datadog API key.
        app_key: Datadog application key.

    Returns:
        List of dicts with keys ``id``, ``name``, ``overall_state``, ``tags``.
    """
    url = f"{_DD_API}/v1/monitor"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            url,
            headers=_headers(api_key, app_key),
            params={
                "monitor_tags": "env:production",
                "with_downtimes": False,
            },
        )
        resp.raise_for_status()
    monitors = resp.json()
    active = [
        {
            "id": str(m["id"]),
            "name": m.get("name", ""),
            "overall_state": m.get("overall_state", ""),
            "tags": m.get("tags", []),
        }
        for m in monitors
        if m.get("overall_state") in ("Alert", "Warn")
    ]
    log.info("datadog_active_alerts", count=len(active))
    return active


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def mute_monitor(
    api_key: str,
    app_key: str,
    monitor_id: str,
    duration_seconds: int,
) -> dict:
    """
    Mute a Datadog monitor for a given number of seconds.

    Args:
        api_key:          Datadog API key.
        app_key:          Datadog application key.
        monitor_id:       Monitor ID to mute.
        duration_seconds: How long (in seconds) to mute the monitor.

    Returns:
        Dict with keys ``id`` and ``end`` (Unix timestamp when mute expires).
    """
    end_ts = int(_time.time()) + duration_seconds
    url = f"{_DD_API}/v1/monitor/{monitor_id}/mute"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            url,
            headers=_headers(api_key, app_key),
            json={"end": end_ts},
        )
        resp.raise_for_status()
    data = resp.json()
    log.info("datadog_monitor_muted", monitor_id=monitor_id, end_ts=end_ts)
    return {"id": str(data.get("id", monitor_id)), "end": end_ts}
