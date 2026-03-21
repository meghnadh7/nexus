"""
MCP server for GitHub operations.

Exposes a single ``POST /invoke`` endpoint that dispatches to the appropriate
tool function based on the ``tool`` field in the request body.  Also exposes
``GET /health`` for container health-checks.
"""

import os
import time
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mcp_github import tools

# ─── Logging ──────────────────────────────────────────────────────────────────

_env = os.getenv("ENV", "production")

import structlog.processors

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

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Nexus MCP — GitHub", version="1.0.0")

_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


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
    """Liveness probe — returns 200 OK if the server is running."""
    return {"status": "ok", "service": "mcp-github"}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse:
    """
    Dispatch a tool call to the underlying GitHub API wrapper.

    The ``tool`` field must match one of the registered tool names.
    The ``arguments`` dict is passed as keyword arguments to the tool function.
    """
    start = time.monotonic()
    tool_name = request.tool
    args = request.arguments

    log.info("tool_invoked", tool=tool_name)

    try:
        if tool_name == "get_pr_diff":
            result = await tools.get_pr_diff(
                token=_GITHUB_TOKEN,
                repo=args["repo"],
                pr_number=int(args["pr_number"]),
            )
        elif tool_name == "post_pr_comment":
            result = await tools.post_pr_comment(
                token=_GITHUB_TOKEN,
                repo=args["repo"],
                pr_number=int(args["pr_number"]),
                body=args["body"],
            )
        elif tool_name == "trigger_workflow":
            result = await tools.trigger_workflow(
                token=_GITHUB_TOKEN,
                repo=args["repo"],
                workflow_id=args["workflow_id"],
                ref=args.get("ref", "main"),
                inputs=args.get("inputs", {}),
            )
        elif tool_name == "get_workflow_run":
            result = await tools.get_workflow_run(
                token=_GITHUB_TOKEN,
                repo=args["repo"],
                run_id=str(args["run_id"]),
            )
        elif tool_name == "list_open_prs":
            result = await tools.list_open_prs(
                token=_GITHUB_TOKEN,
                repo=args["repo"],
            )
        elif tool_name == "get_pr_files":
            result = await tools.get_pr_files(
                token=_GITHUB_TOKEN,
                repo=args["repo"],
                pr_number=int(args["pr_number"]),
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
    uvicorn.run("mcp_github.server:app", host="0.0.0.0", port=8001, reload=False)
