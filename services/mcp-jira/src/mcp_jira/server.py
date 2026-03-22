"""
MCP server for Jira operations.

Exposes ``POST /invoke`` and ``GET /health`` endpoints.
"""

import os
import time
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mcp_jira import tools

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

app = FastAPI(title="Nexus MCP — Jira", version="1.0.0")

_JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
_JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
_JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
_JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "ENG")


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
    return {"status": "ok", "service": "mcp-jira"}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse:
    """Dispatch a tool call to the Jira API wrapper."""
    start = time.monotonic()
    tool_name = request.tool
    args = request.arguments

    log.info("tool_invoked", tool=tool_name)

    try:
        if tool_name == "create_issue":
            result = await tools.create_issue(
                base_url=_JIRA_BASE_URL,
                email=_JIRA_EMAIL,
                api_token=_JIRA_API_TOKEN,
                project_key=args.get("project_key", _JIRA_PROJECT_KEY),
                summary=args["summary"],
                description=args["description"],
                priority=args.get("priority", "High"),
                labels=args.get("labels", []),
                issue_type=args.get("issue_type", "Bug"),
            )
        elif tool_name == "update_issue_status":
            result = await tools.update_issue_status(
                base_url=_JIRA_BASE_URL,
                email=_JIRA_EMAIL,
                api_token=_JIRA_API_TOKEN,
                issue_key=args["issue_key"],
                status=args["status"],
            )
        elif tool_name == "add_comment":
            result = await tools.add_comment(
                base_url=_JIRA_BASE_URL,
                email=_JIRA_EMAIL,
                api_token=_JIRA_API_TOKEN,
                issue_key=args["issue_key"],
                body=args["body"],
            )
        elif tool_name == "search_issues":
            result = await tools.search_issues(
                base_url=_JIRA_BASE_URL,
                email=_JIRA_EMAIL,
                api_token=_JIRA_API_TOKEN,
                jql=args["jql"],
                max_results=int(args.get("max_results", 50)),
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
    uvicorn.run("mcp_jira.server:app", host="0.0.0.0", port=8003, reload=False)
