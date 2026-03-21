"""
MCP server for Slack operations.

Exposes ``POST /invoke`` and ``GET /health`` endpoints.
"""

import os
import time
from typing import Any, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mcp_slack import tools

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

app = FastAPI(title="Nexus MCP — Slack", version="1.0.0")

_SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")


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
    return {"status": "ok", "service": "mcp-slack"}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse:
    """Dispatch a tool call to the Slack API wrapper."""
    start = time.monotonic()
    tool_name = request.tool
    args = request.arguments

    log.info("tool_invoked", tool=tool_name)

    try:
        if tool_name == "post_message":
            result = await tools.post_message(
                token=_SLACK_TOKEN,
                channel=args["channel"],
                text=args["text"],
                thread_ts=args.get("thread_ts"),
            )
        elif tool_name == "get_channel_history":
            result = await tools.get_channel_history(
                token=_SLACK_TOKEN,
                channel=args["channel"],
                limit=int(args.get("limit", 20)),
            )
        elif tool_name == "lookup_user":
            result = await tools.lookup_user(
                token=_SLACK_TOKEN,
                user_id=args["user_id"],
            )
        elif tool_name == "update_message":
            result = await tools.update_message(
                token=_SLACK_TOKEN,
                channel=args["channel"],
                ts=args["ts"],
                text=args["text"],
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
    uvicorn.run("mcp_slack.server:app", host="0.0.0.0", port=8002, reload=False)
