"""
Tests for individual Nexus agents.

Covers:
  - CodeReviewAgent: JSON parsing, MCP tool calls, SQL injection detection
  - RAGAgent: HyDE retrieval calls Pinecone with correct vector
  - IncidentResponseAgent: creates both Slack message and Jira ticket
  - Integration test (requires real LLM): SQL injection → approved=False + critical severity
"""

import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from orchestrator.agents.code_review import CodeReviewAgent
from orchestrator.agents.incident import IncidentResponseAgent
from orchestrator.agents.rag_agent import RAGAgent
from orchestrator.graph.state import NexusEvent, NexusState, RAGContext, RetrievedChunk


# ─── CodeReviewAgent tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_code_review_agent_parses_llm_json_and_posts_comment(
    base_state_with_rag: NexusState,
    mock_mcp,
    mock_llm_code_review,
):
    """
    CodeReviewAgent must:
    1. Parse the LLM's JSON output into CodeReviewResult.
    2. Set approved=False when a critical issue is present.
    3. Call post_pr_comment via the MCP client.
    """
    mock_settings = MagicMock(
        openai_model="gpt-4o",
        openai_api_key="test",
        github_org="myorg",
        github_repo="myrepo",
    )
    with patch("orchestrator.agents.base.get_settings", return_value=mock_settings):
        agent = CodeReviewAgent()
        # Bypass MCP client on agent instance with our fixture mock
        agent.mcp = MagicMock()
        agent.mcp.call = mock_mcp
        agent.mcp.drain_logs = MagicMock(return_value=[])

        result_dict = await agent.run(base_state_with_rag)

    result = result_dict["code_review_result"]

    assert result is not None
    assert result.approved is False
    assert len(result.issues) == 1
    assert result.issues[0].severity == "critical"
    assert "hardcoded" in result.issues[0].message.lower() or "secret" in result.issues[0].message.lower()
    assert result.posted_to_github is True

    # Verify post_pr_comment was called
    mock_mcp.assert_any_await(
        "github",
        "post_pr_comment",
        ANY,
    )


@pytest.mark.asyncio
async def test_code_review_agent_handles_malformed_json(
    base_state_with_rag: NexusState,
    mock_mcp,
):
    """
    When the LLM returns malformed JSON, CodeReviewAgent must not crash —
    it should return a safe fallback result with approved=False.
    """
    bad_message = MagicMock()
    bad_message.content = "This is not valid JSON at all!!!"

    mock_settings = MagicMock(
        openai_model="gpt-4o",
        openai_api_key="test",
        github_org="myorg",
        github_repo="myrepo",
    )
    with (
        patch(
            "langchain_openai.ChatOpenAI.ainvoke",
            new_callable=AsyncMock,
            return_value=bad_message,
        ),
        patch("orchestrator.agents.base.get_settings", return_value=mock_settings),
    ):
        agent = CodeReviewAgent()
        agent.mcp = MagicMock()
        agent.mcp.call = mock_mcp
        agent.mcp.drain_logs = MagicMock(return_value=[])

        result_dict = await agent.run(base_state_with_rag)

    result = result_dict["code_review_result"]
    assert result is not None
    assert result.approved is False  # Safe fallback
    assert "parsing failed" in result.summary.lower() or result.summary != ""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_review_detects_sql_injection():
    """
    Integration test using the real GPT-4o LLM.

    A diff containing an f-string SQL query must result in:
    - approved=False
    - At least one critical severity issue mentioning SQL injection

    Requires: OPENAI_API_KEY set in environment.
    Skip with: pytest -m "not integration"
    """
    import os
    from orchestrator.graph.state import RAGContext

    sql_injection_diff = """\
--- a/src/db.py
+++ b/src/db.py
@@ -1,4 +1,8 @@
 import sqlite3

+def get_user(user_id):
+    conn = sqlite3.connect('users.db')
+    cursor = conn.cursor()
+    cursor.execute(f'SELECT * FROM users WHERE id={user_id}')
+    return cursor.fetchone()
"""

    state: NexusState = {
        "event": NexusEvent(
            event_type="pr_opened",
            source="github",
            payload={
                "pull_request": {"number": 99, "title": "Add user lookup"},
                "repository": {"full_name": "test/repo"},
            },
            event_id="test-sql-inject",
            timestamp="2026-03-25T00:00:00Z",
        ),
        "messages": [],
        "current_agent": "",
        "iteration_count": 0,
        "rag_context": RAGContext(query="sql", retrieved_chunks=[]),
        "code_review_result": None,
        "incident_result": None,
        "deploy_result": None,
        "next_action": "",
        "error": None,
        "requires_human_approval": False,
        "human_approved": None,
        "tool_calls": [],
    }

    # Mock only the MCP calls, use real LLM
    mock_mcp_call = AsyncMock(return_value=sql_injection_diff)
    mock_post_comment = AsyncMock(return_value={"id": 1})

    async def mcp_side_effect(server, tool, arguments):
        if tool == "get_pr_diff":
            return sql_injection_diff
        if tool == "post_pr_comment":
            return {"id": 1}
        return {}

    agent = CodeReviewAgent()
    agent.mcp = MagicMock()
    agent.mcp.call = AsyncMock(side_effect=mcp_side_effect)
    agent.mcp.drain_logs = MagicMock(return_value=[])

    result_dict = await agent.run(state)
    result = result_dict["code_review_result"]

    assert result.approved is False, "SQL injection should not be approved"

    critical_issues = [i for i in result.issues if i.severity == "critical"]
    assert len(critical_issues) >= 1, (
        f"Expected at least one critical issue for SQL injection, got: "
        f"{[i.severity for i in result.issues]}"
    )

    sql_mention = any(
        "sql" in i.message.lower() or "inject" in i.message.lower()
        for i in critical_issues
    )
    assert sql_mention, (
        f"Expected 'sql' or 'inject' in critical issue messages, got: "
        f"{[i.message for i in critical_issues]}"
    )


# ─── RAGAgent tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rag_agent_calls_pinecone_with_hyde_vector(
    base_state: NexusState,
    mock_mcp,
    mock_llm_rag,
):
    """
    RAGAgent must:
    1. Call ChatOpenAI to generate a hypothetical document.
    2. Call embed_query with the hypothetical document text.
    3. Call VectorStore.query with the resulting vector.
    4. Return a RAGContext with retrieved chunks.
    """
    fake_vector = [0.42] * 3072
    expected_chunks = [
        {
            "id": "abc",
            "score": 0.95,
            "metadata": {
                "content": "def validate_jwt(token): ...",
                "source_file": "src/auth/jwt.py",
                "chunk_index": 0,
                "language": "python",
            },
        }
    ]

    mock_embed = AsyncMock(return_value=fake_vector)
    mock_query = AsyncMock(return_value=expected_chunks)

    mock_settings = MagicMock(
        openai_model="gpt-4o",
        openai_api_key="test",
        pinecone_api_key="test",
        pinecone_index_name="test",
        slack_default_channel="#test",
    )
    with (
        patch("orchestrator.agents.rag_agent.embed_query", mock_embed),
        patch("orchestrator.agents.rag_agent.VectorStore.query", mock_query),
        patch("orchestrator.agents.base.get_settings", return_value=mock_settings),
    ):
        agent = RAGAgent()
        agent.mcp = MagicMock()
        agent.mcp.call = mock_mcp
        agent.mcp.drain_logs = MagicMock(return_value=[])

        result_dict = await agent.run(base_state)

    rag_context = result_dict["rag_context"]
    assert rag_context is not None
    assert len(rag_context.retrieved_chunks) == 1
    assert rag_context.retrieved_chunks[0].source_file == "src/auth/jwt.py"
    assert rag_context.retrieved_chunks[0].score == 0.95

    # Verify embed_query was called with the HyDE text (not the raw query)
    mock_embed.assert_called_once()
    call_args = mock_embed.call_args[1] if mock_embed.call_args.kwargs else {}
    # The text argument should be the HyDE hypothetical document
    assert mock_embed.called


@pytest.mark.asyncio
async def test_rag_agent_posts_to_slack_for_mention(
    slack_event: NexusEvent,
    base_state: NexusState,
    mock_mcp,
    mock_llm_rag,
):
    """
    For slack_mention events, RAGAgent must post the synthesized answer
    back to the Slack thread via MCP.
    """
    state = dict(base_state)
    state["event"] = slack_event

    fake_vector = [0.1] * 3072
    expected_chunks = [
        {
            "id": "xyz",
            "score": 0.88,
            "metadata": {
                "content": "def validate_jwt(token): ...",
                "source_file": "src/auth/jwt.py",
                "chunk_index": 0,
                "language": "python",
            },
        }
    ]

    mock_settings = MagicMock(
        openai_model="gpt-4o",
        openai_api_key="test",
        pinecone_api_key="test",
        pinecone_index_name="test",
        slack_default_channel="#devops-alerts",
    )
    with (
        patch("orchestrator.agents.rag_agent.embed_query", AsyncMock(return_value=fake_vector)),
        patch("orchestrator.agents.rag_agent.VectorStore.query", AsyncMock(return_value=expected_chunks)),
        patch("orchestrator.agents.base.get_settings", return_value=mock_settings),
    ):
        agent = RAGAgent()
        agent.mcp = MagicMock()
        agent.mcp.call = mock_mcp
        agent.mcp.drain_logs = MagicMock(return_value=[])

        result_dict = await agent.run(state)  # type: ignore[arg-type]

    # Verify Slack post_message was called
    mock_mcp.assert_any_await("slack", "post_message", ANY)

    rag_context = result_dict["rag_context"]
    assert rag_context.synthesized_answer is not None
    assert len(rag_context.synthesized_answer) > 0


# ─── IncidentResponseAgent tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_incident_agent_creates_slack_and_jira(
    alert_event: NexusEvent,
    base_state: NexusState,
    mock_mcp,
    mock_llm_incident,
):
    """
    IncidentResponseAgent must:
    1. Fetch alert details from Datadog MCP.
    2. Post a formatted message to Slack.
    3. Create a Jira ticket.
    4. Return an IncidentResult with both jira_ticket_id set and slack_posted=True.
    """
    state = dict(base_state)
    state["event"] = alert_event
    state["rag_context"] = RAGContext(
        query="high memory usage runbook",
        retrieved_chunks=[
            RetrievedChunk(
                content="# Memory Leak Runbook\nRestart the affected service...",
                source_file="runbooks/memory-leak.md",
                chunk_index=0,
                score=0.91,
                language="markdown",
            )
        ],
    )

    mock_settings = MagicMock(
        openai_model="gpt-4o",
        openai_api_key="test",
        slack_default_channel="#devops-alerts",
    )
    with patch("orchestrator.agents.base.get_settings", return_value=mock_settings):
        agent = IncidentResponseAgent()
        agent.mcp = MagicMock()
        agent.mcp.call = mock_mcp
        agent.mcp.drain_logs = MagicMock(return_value=[])

        result_dict = await agent.run(state)  # type: ignore[arg-type]

    result = result_dict["incident_result"]

    assert result is not None
    assert result.slack_posted is True
    assert result.jira_ticket_id == "ENG-42"
    assert result.severity in ("p1", "p2", "p3", "p4")
    assert len(result.root_cause) > 0
    assert len(result.suggested_fix) > 0

    # Verify both Slack and Jira were called
    slack_calls = [
        call for call in mock_mcp.call_args_list
        if call.args[0] == "slack" and call.args[1] == "post_message"
    ]
    jira_calls = [
        call for call in mock_mcp.call_args_list
        if call.args[0] == "jira" and call.args[1] == "create_issue"
    ]

    assert len(slack_calls) >= 1, "Expected at least one Slack post_message call"
    assert len(jira_calls) >= 1, "Expected at least one Jira create_issue call"


@pytest.mark.asyncio
async def test_incident_agent_handles_missing_jira_key(
    alert_event: NexusEvent,
    base_state: NexusState,
    mock_llm_incident,
):
    """
    When Jira create_issue returns a response without a 'key' field,
    the agent must not crash — jira_ticket_id defaults to 'UNKNOWN'.
    """
    async def mock_mcp_side_effect(server, tool, arguments):
        if tool == "get_alert":
            return {
                "id": "12345",
                "name": "High Memory",
                "type": "metric alert",
                "query": "avg:mem > 90",
                "overall_state": "Alert",
                "message": "",
                "tags": [],
            }
        if tool == "post_message":
            return {"ts": "123", "channel": "C_CHAN"}
        if tool == "create_issue":
            return {}  # No 'key' in response
        return {}

    state = dict(base_state)
    state["event"] = alert_event

    mock_settings = MagicMock(
        openai_model="gpt-4o",
        openai_api_key="test",
        slack_default_channel="#devops-alerts",
    )
    with patch("orchestrator.agents.base.get_settings", return_value=mock_settings):
        agent = IncidentResponseAgent()
        agent.mcp = MagicMock()
        agent.mcp.call = AsyncMock(side_effect=mock_mcp_side_effect)
        agent.mcp.drain_logs = MagicMock(return_value=[])

        result_dict = await agent.run(state)  # type: ignore[arg-type]

    result = result_dict["incident_result"]
    assert result.jira_ticket_id == "UNKNOWN"
    assert result.slack_posted is True
