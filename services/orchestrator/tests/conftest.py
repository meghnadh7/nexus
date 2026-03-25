"""
Pytest configuration and shared fixtures for the Nexus orchestrator tests.

Fixtures:
  - pr_event        : fully-populated NexusEvent for a PR opened event
  - alert_event     : NexusEvent for a Datadog alert fired event
  - slack_event     : NexusEvent for a Slack app_mention event
  - mock_mcp        : AsyncMock patch for MCPClient.call
  - mock_llm        : patch for ChatOpenAI that returns pre-configured responses
  - base_state      : minimal NexusState dict for graph testing
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.graph.state import (
    CodeReviewResult,
    IncidentResult,
    NexusEvent,
    NexusState,
    RAGContext,
    RetrievedChunk,
)


# ─── Event Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def pr_event() -> NexusEvent:
    """NexusEvent simulating a GitHub pull_request opened webhook."""
    return NexusEvent(
        event_type="pr_opened",
        source="github",
        payload={
            "action": "opened",
            "pull_request": {
                "number": 42,
                "title": "Add user authentication endpoint",
                "body": "Adds JWT-based authentication",
                "head": {"sha": "abc1234"},
                "user": {"login": "developer"},
            },
            "repository": {"full_name": "myorg/myrepo"},
        },
        event_id="gh-42-abc123",
        timestamp="2026-03-25T10:00:00Z",
    )


@pytest.fixture
def alert_event() -> NexusEvent:
    """NexusEvent simulating a Datadog alert fired webhook."""
    return NexusEvent(
        event_type="alert_fired",
        source="datadog",
        payload={
            "alert_id": "12345",
            "id": "12345",
            "alert_title": "High memory usage on api-service",
            "title": "High memory usage on api-service",
            "alert_metric": "system.mem.used",
            "alert_status": "Alert",
            "alert_query": "avg(last_5m):avg:system.mem.used{service:api} > 90",
        },
        event_id="dd-12345-xyz789",
        timestamp="2026-03-25T02:15:00Z",
    )


@pytest.fixture
def slack_event() -> NexusEvent:
    """NexusEvent simulating a Slack app_mention event."""
    return NexusEvent(
        event_type="slack_mention",
        source="slack",
        payload={
            "text": "<@U_BOT_ID> where does JWT validation happen?",
            "user": "U_DEVELOPER_001",
            "channel": "C_DEV_CHANNEL",
            "ts": "1711360000.000000",
            "thread_ts": "1711360000.000000",
        },
        event_id="slack-aabbccdd",
        timestamp="2026-03-25T11:30:00Z",
    )


@pytest.fixture
def deploy_event() -> NexusEvent:
    """NexusEvent simulating a manual deploy request."""
    return NexusEvent(
        event_type="deploy_request",
        source="manual",
        payload={
            "repo": "myorg/myrepo",
            "environment": "staging",
            "commit_sha": "abc1234def5678",
            "workflow_id": "cd.yml",
            "ref": "main",
        },
        event_id="deploy-deadbeef",
        timestamp="2026-03-25T14:00:00Z",
    )


# ─── State Fixture ────────────────────────────────────────────────────────────


@pytest.fixture
def base_state(pr_event: NexusEvent) -> NexusState:
    """Minimal NexusState dict suitable for graph node testing."""
    return {
        "event": pr_event,
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


@pytest.fixture
def base_state_with_rag(base_state: NexusState) -> NexusState:
    """NexusState with pre-populated RAG context for agent tests."""
    state = dict(base_state)
    state["rag_context"] = RAGContext(
        query="JWT validation",
        retrieved_chunks=[
            RetrievedChunk(
                content="def validate_jwt(token: str) -> dict:\n    '''Validate and decode a JWT token.'''\n    return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])",
                source_file="src/auth/jwt.py",
                chunk_index=0,
                score=0.95,
                language="python",
            ),
            RetrievedChunk(
                content="# Authentication Middleware\n\nJWT tokens are validated using the `validate_jwt` function in `src/auth/jwt.py`.",
                source_file="docs/auth.md",
                chunk_index=0,
                score=0.88,
                language="markdown",
            ),
        ],
        synthesized_answer=None,
    )
    return state  # type: ignore[return-value]


# ─── Mock MCP Client Fixture ──────────────────────────────────────────────────


@pytest.fixture
def mock_mcp():
    """
    Patch MCPClient.call with an AsyncMock.

    Returns the mock so tests can configure return_value or assert calls.
    The default return value is a generic dict to prevent attribute errors.
    """
    with patch(
        "orchestrator.tools.mcp_client.MCPClient.call",
        new_callable=AsyncMock,
    ) as mock:
        # Default responses for each tool
        async def side_effect(server: str, tool: str, arguments: dict) -> Any:
            if tool == "get_pr_diff":
                return "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1,5 +1,6 @@\n+import os\n+SECRET = os.environ.get('JWT_SECRET', 'hardcoded-secret')\n def login():\n     pass"
            if tool == "post_pr_comment":
                return {"id": 999, "html_url": "https://github.com/pr/42#comment-999"}
            if tool == "get_alert":
                return {
                    "id": "12345",
                    "name": "High Memory Usage",
                    "type": "metric alert",
                    "query": "avg(last_5m):avg:system.mem.used{*} > 90",
                    "overall_state": "Alert",
                    "message": "Memory usage is critically high",
                    "tags": ["env:production", "service:api"],
                }
            if tool == "post_message":
                return {"ts": "1711360001.000000", "channel": "C_CHANNEL"}
            if tool == "create_issue":
                return {"id": "10001", "key": "ENG-42", "url": "https://jira.example.com/browse/ENG-42"}
            if tool == "trigger_workflow":
                return "987654321"
            if tool == "get_workflow_run":
                return {"id": "987654321", "status": "completed", "conclusion": "success", "html_url": "https://github.com/actions/runs/987654321"}
            return {}

        mock.side_effect = side_effect
        yield mock


# ─── Mock LLM Fixture ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_code_review():
    """
    Patch ChatOpenAI.ainvoke for the CodeReviewAgent.

    Returns a response containing a JSON code review with one critical issue
    (hardcoded secret).
    """
    review_json = json.dumps({
        "issues": [
            {
                "severity": "critical",
                "file": "src/auth.py",
                "line": 2,
                "message": "Hardcoded secret detected — 'hardcoded-secret' should never be a fallback for JWT_SECRET",
                "suggestion": "SECRET = os.environ['JWT_SECRET']  # Fail fast if not set",
            }
        ],
        "summary": "Critical security issue: hardcoded JWT secret fallback found.",
        "approved": False,
    })

    mock_message = MagicMock()
    mock_message.content = review_json

    with patch(
        "langchain_openai.ChatOpenAI.ainvoke",
        new_callable=AsyncMock,
        return_value=mock_message,
    ) as mock:
        yield mock


@pytest.fixture
def mock_llm_incident():
    """
    Patch ChatOpenAI.ainvoke for the IncidentResponseAgent.

    Returns a response with a realistic RCA JSON.
    """
    rca_json = json.dumps({
        "root_cause": "Memory leak in the connection pool manager causing OOM after 6 hours of sustained load.",
        "suggested_fix": "1. Restart api-service pods immediately.\n2. Deploy hotfix from PR #101 (fix connection pool cleanup).\n3. Monitor memory for 30 minutes post-restart.",
        "runbook_references": ["runbooks/memory-leak.md", "runbooks/oom-recovery.md"],
        "severity": "p2",
    })

    mock_message = MagicMock()
    mock_message.content = rca_json

    with patch(
        "langchain_openai.ChatOpenAI.ainvoke",
        new_callable=AsyncMock,
        return_value=mock_message,
    ) as mock:
        yield mock


@pytest.fixture
def mock_llm_rag():
    """
    Patch ChatOpenAI.ainvoke for the RAGAgent (HyDE + synthesis).

    First call returns a hypothetical document; second call returns a
    synthesized answer.
    """
    hyde_message = MagicMock()
    hyde_message.content = (
        "JWT validation in this codebase happens in src/auth/jwt.py "
        "using the validate_jwt function which calls jwt.decode() with HS256 algorithm."
    )

    synthesis_message = MagicMock()
    synthesis_message.content = (
        "JWT validation happens in `src/auth/jwt.py` in the `validate_jwt` function. "
        "It decodes tokens using HS256 and the `SECRET_KEY` environment variable. "
        "See also `docs/auth.md` for the authentication flow overview."
    )

    with patch(
        "langchain_openai.ChatOpenAI.ainvoke",
        new_callable=AsyncMock,
        side_effect=[hyde_message, synthesis_message],
    ) as mock:
        yield mock
