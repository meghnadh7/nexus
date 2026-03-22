"""
MCP server for Datadog operations.

Exposes ``POST /invoke`` and ``GET /health`` endpoints.
"""

import os
import time
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mcp_datadog import tools

_env = os.getenv("ENV", "production")

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if _env == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)

app = FastAPI(title="Nexus MCP — Datadog", version="1.0.0")

_DD_API_KEY = os.getenv("DATADOG_API_KEY", "")
_DD_APP_KEY = os.getenv("DATADOG_APP_KEY", "")


class InvokeRequest(BaseModel):
    """Payload for the /invoke endpoint."""

    tool: str
    arguments: dict[str, Any] = {}


class InvokeResponse(BaseModel):
    """Successful response from /invoke."""

    result: Any
    duration_ms: float


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "mcp-datadog"}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse:
    """Dispatch a tool call to the Datadog API wrapper."""
    start = time.monotonic()
    tool_name = request.tool
    args = request.arguments

    log.info("tool_invoked", tool=tool_name)

    try:
        if tool_name == "get_alert":
            result = await tools.get_alert(
                api_key=_DD_API_KEY,
                app_key=_DD_APP_KEY,
                alert_id=str(args["alert_id"]),
            )
        elif tool_name == "query_metrics":
            result = await tools.query_metrics(
                api_key=_DD_API_KEY,
                app_key=_DD_APP_KEY,
                metric_query=args["metric_query"],
                from_ts=int(args["from_ts"]),
                to_ts=int(args["to_ts"]),
            )
        elif tool_name == "list_active_alerts":
            result = await tools.list_active_alerts(
                api_key=_DD_API_KEY,
                app_key=_DD_APP_KEY,
            )
        elif tool_name == "mute_monitor":
            result = await tools.mute_monitor(
                api_key=_DD_API_KEY,
                app_key=_DD_APP_KEY,
                monitor_id=str(args["monitor_id"]),
                duration_seconds=int(args["duration_seconds"]),
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name!r}")

        duration = (time.monotonic() - start) * 1000
        log.info("tool_succeeded", tool=tool_name, duration_ms=round(duration, 2))
        return InvokeResponse(result=result, duration_ms=round(duration, 2))

    except HTTPException:
        raise
    except Exception as exc:
        duration = (time.monotonic() - start) * 1000
        log.error("tool_failed", tool=tool_name, error=str(exc), duration_ms=round(duration, 2))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("mcp_datadog.server:app", host="0.0.0.0", port=8004, reload=False)
