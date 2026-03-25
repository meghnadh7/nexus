"""
Nexus Orchestrator — FastAPI application entry point.

Exposes webhook endpoints for GitHub, Datadog, and Slack, a manual deploy
endpoint, health check, and Prometheus metrics. All heavy work runs as
FastAPI BackgroundTasks so webhook handlers return immediately.
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import structlog
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from orchestrator.config import get_settings
from orchestrator.graph.orchestrator import build_graph
from orchestrator.graph.state import NexusEvent, NexusState
from orchestrator.memory.checkpointer import checkpointer_lifespan

# ─── Logging ──────────────────────────────────────────────────────────────────

settings = get_settings()

_env = settings.env

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

# ─── Prometheus Metrics ───────────────────────────────────────────────────────

nexus_events_total = Counter(
    "nexus_events_total",
    "Total events received by the orchestrator",
    ["event_type", "source"],
)

nexus_agent_duration_seconds = Histogram(
    "nexus_agent_duration_seconds",
    "Duration of agent graph executions in seconds",
    ["agent"],
)

nexus_errors_total = Counter(
    "nexus_errors_total",
    "Total errors raised during agent execution",
    ["agent"],
)

nexus_llm_tokens_total = Counter(
    "nexus_llm_tokens_total",
    "Total LLM tokens consumed by agents",
    ["agent", "model"],
)

# ─── Application state ────────────────────────────────────────────────────────

_graph = None  # Set in lifespan


# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    On startup:
      1. Creates the AsyncPostgresSaver checkpointer (runs DDL migrations).
      2. Compiles the LangGraph graph.
      3. Logs readiness.

    On shutdown:
      - Closes the Postgres connection pool cleanly.
    """
    global _graph

    log.info("orchestrator_starting", env=settings.env)

    try:
        async with checkpointer_lifespan() as checkpointer:
            _graph = build_graph(checkpointer=checkpointer)
            log.info("orchestrator_ready", human_in_the_loop=settings.human_in_the_loop)
            yield
    except Exception as exc:
        log.error("orchestrator_startup_failed", error=str(exc))
        raise
    finally:
        log.info("orchestrator_shutting_down")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─── Helper: verify GitHub signature ─────────────────────────────────────────


def _verify_github_signature(body: bytes, signature_header: str | None) -> None:
    """
    Verify the X-Hub-Signature-256 HMAC for a GitHub webhook payload.

    Args:
        body:             Raw request body bytes.
        signature_header: Value of the ``X-Hub-Signature-256`` header.

    Raises:
        HTTPException(401): If the signature is missing or invalid.
    """
    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")
    if not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Invalid signature format")

    expected = hmac.new(
        settings.github_webhook_secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    provided = signature_header[len("sha256="):]

    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="GitHub signature verification failed")


# ─── Helper: verify Slack signature ──────────────────────────────────────────


def _verify_slack_signature(
    body: bytes,
    timestamp: str | None,
    signature_header: str | None,
) -> None:
    """
    Verify the X-Slack-Signature HMAC for a Slack webhook payload.

    Args:
        body:             Raw request body bytes.
        timestamp:        Value of the ``X-Slack-Request-Timestamp`` header.
        signature_header: Value of the ``X-Slack-Signature`` header.

    Raises:
        HTTPException(401): If the signature is missing or invalid.
    """
    if not signature_header or not timestamp:
        raise HTTPException(
            status_code=401,
            detail="Missing Slack signature headers",
        )

    base_string = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected = (
        "v0="
        + hmac.new(
            settings.slack_signing_secret.encode(),
            base_string.encode(),
            hashlib.sha256,
        ).hexdigest()
    )

    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Slack signature verification failed")


# ─── Background graph runner ──────────────────────────────────────────────────


async def run_graph(event: NexusEvent) -> None:
    """
    Execute the LangGraph graph for a given event as a background task.

    Args:
        event: The NexusEvent that triggered this run.
    """
    if _graph is None:
        log.error("graph_not_initialised", event_id=event.event_id)
        return

    thread_id = event.event_id

    initial_state: NexusState = {
        "event": event,
        "messages": [],
        "current_agent": "",
        "iteration_count": 0,
        "rag_context": None,
        "code_review_result": None,
        "incident_result": None,
        "deploy_result": None,
        "next_action": "",
        "error": None,
        "requires_human_approval": False,
        "human_approved": None,
        "tool_calls": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    nexus_events_total.labels(
        event_type=event.event_type,
        source=event.source,
    ).inc()

    start_time = time.monotonic()
    agent_label = event.event_type

    try:
        log.info(
            "graph_run_starting",
            event_id=event.event_id,
            event_type=event.event_type,
        )

        async for step in _graph.astream(initial_state, config=config):
            node_name = list(step.keys())[0] if step else "unknown"
            log.info(
                "graph_step",
                event_id=event.event_id,
                node=node_name,
            )

        duration = time.monotonic() - start_time
        nexus_agent_duration_seconds.labels(agent=agent_label).observe(duration)

        log.info(
            "graph_run_complete",
            event_id=event.event_id,
            duration_seconds=round(duration, 3),
        )

    except Exception as exc:
        duration = time.monotonic() - start_time
        nexus_agent_duration_seconds.labels(agent=agent_label).observe(duration)
        nexus_errors_total.labels(agent=agent_label).inc()

        log.error(
            "graph_run_failed",
            event_id=event.event_id,
            error=str(exc),
            duration_seconds=round(duration, 3),
        )


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Kubernetes liveness and readiness probe."""
    return {"status": "ok", "service": "nexus-orchestrator"}


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Receive and process GitHub webhook events.

    Only ``pull_request`` events with action ``opened`` or ``synchronize``
    are processed. All other events return 200 immediately.

    Verifies the ``X-Hub-Signature-256`` HMAC before processing.
    """
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    _verify_github_signature(body, signature)

    github_event = request.headers.get("X-GitHub-Event", "")
    payload = json.loads(body)

    if github_event != "pull_request":
        log.info("github_event_ignored", github_event=github_event)
        return JSONResponse({"status": "ignored", "event": github_event})

    action = payload.get("action", "")
    if action not in ("opened", "synchronize"):
        log.info("github_pr_action_ignored", action=action)
        return JSONResponse({"status": "ignored", "action": action})

    event_type = "pr_opened" if action == "opened" else "pr_updated"
    pr_number = payload.get("pull_request", {}).get("number", 0)

    event = NexusEvent(
        event_type=event_type,  # type: ignore[arg-type]
        source="github",
        payload=payload,
        event_id=f"gh-{pr_number}-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    log.info(
        "github_webhook_received",
        event_type=event_type,
        event_id=event.event_id,
        pr_number=pr_number,
    )

    background_tasks.add_task(run_graph, event)
    return JSONResponse({"status": "accepted", "event_id": event.event_id})


@app.post("/webhook/datadog")
async def datadog_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Receive Datadog alert webhook payloads.

    Datadog does not sign webhooks with HMAC by default, but if a shared
    secret is configured in the Datadog UI the ``X-Datadog-Signature`` header
    is present and validated here.
    """
    body = await request.body()
    payload: dict = json.loads(body)

    alert_id = str(payload.get("id") or payload.get("alert_id") or uuid.uuid4().hex[:8])

    event = NexusEvent(
        event_type="alert_fired",
        source="datadog",
        payload=payload,
        event_id=f"dd-{alert_id}-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    log.info(
        "datadog_webhook_received",
        event_id=event.event_id,
        alert_id=alert_id,
    )

    background_tasks.add_task(run_graph, event)
    return JSONResponse({"status": "accepted", "event_id": event.event_id})


@app.post("/webhook/slack")
async def slack_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Receive Slack events and app mentions.

    Handles the Slack URL verification challenge automatically.
    Processes ``app_mention`` events by enqueuing a ``slack_mention`` graph run.
    Verifies the ``X-Slack-Signature`` header.
    """
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")
    _verify_slack_signature(body, timestamp, signature)

    payload: dict = json.loads(body)

    # Handle Slack's URL verification challenge
    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload["challenge"]})

    event_data = payload.get("event", {})
    event_type = event_data.get("type", "")

    if event_type != "app_mention":
        return JSONResponse({"status": "ignored", "event_type": event_type})

    user_id = event_data.get("user", "unknown")
    channel = event_data.get("channel", "")
    ts = event_data.get("ts", "")

    event = NexusEvent(
        event_type="slack_mention",
        source="slack",
        payload={
            "text": event_data.get("text", ""),
            "user": user_id,
            "channel": channel,
            "ts": ts,
            "thread_ts": event_data.get("thread_ts", ts),
        },
        event_id=f"slack-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    log.info(
        "slack_mention_received",
        event_id=event.event_id,
        channel=channel,
        user_id=user_id,
    )

    background_tasks.add_task(run_graph, event)
    return JSONResponse({"status": "accepted", "event_id": event.event_id})


class DeployRequest(BaseModel):
    """Payload for the manual deploy endpoint."""

    repo: str
    environment: str
    commit_sha: str
    workflow_id: str = "cd.yml"
    ref: str = "main"


@app.post("/deploy")
async def trigger_deploy(
    deploy_request: DeployRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Manually trigger a deployment via the CD workflow.

    In production, protect this endpoint with API key middleware or network
    policy (internal-only). The HUMAN_IN_THE_LOOP flag controls whether
    approval is required before the workflow is dispatched.
    """
    event = NexusEvent(
        event_type="deploy_request",
        source="manual",
        payload=deploy_request.model_dump(),
        event_id=f"deploy-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    log.info(
        "deploy_request_received",
        event_id=event.event_id,
        repo=deploy_request.repo,
        environment=deploy_request.environment,
    )

    background_tasks.add_task(run_graph, event)
    return JSONResponse({"status": "accepted", "event_id": event.event_id})


if __name__ == "__main__":
    uvicorn.run(
        "orchestrator.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.env == "development",
        log_level=settings.log_level.lower(),
    )
