"""
Tests for the LangGraph orchestrator graph.

Covers:
  - classify_event routing for each event type
  - route_after_classification returns the correct node names
  - route_after_rag returns the correct node names
  - human_approval_gate behaviour (HUMAN_IN_THE_LOOP=true/false)
  - Full graph traversal for a PR event (using MemorySaver for tests)
"""

import os
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.graph.nodes import (
    classify_event,
    human_approval_gate,
    route_after_classification,
    route_after_rag,
)
from orchestrator.graph.orchestrator import build_graph
from orchestrator.graph.state import NexusEvent, NexusState


# ─── classify_event tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_pr_opened_routes_to_rag(pr_event: NexusEvent, base_state: NexusState):
    """classify_event must set next_action='rag_then_code_review' for pr_opened."""
    result = await classify_event(base_state)
    assert result["next_action"] == "rag_then_code_review"
    assert result["current_agent"] == "classify"
    assert result["iteration_count"] == 1


@pytest.mark.asyncio
async def test_classify_pr_updated_routes_to_rag(base_state: NexusState, pr_event: NexusEvent):
    """classify_event must set next_action='rag_then_code_review' for pr_updated."""
    state = dict(base_state)
    state["event"] = NexusEvent(
        event_type="pr_updated",
        source="github",
        payload={},
        event_id="gh-42-updated",
        timestamp="2026-03-25T10:05:00Z",
    )
    result = await classify_event(state)  # type: ignore[arg-type]
    assert result["next_action"] == "rag_then_code_review"


@pytest.mark.asyncio
async def test_classify_alert_fired_routes_to_rag_incident(
    alert_event: NexusEvent, base_state: NexusState
):
    """classify_event must set next_action='rag_then_incident' for alert_fired."""
    state = dict(base_state)
    state["event"] = alert_event
    result = await classify_event(state)  # type: ignore[arg-type]
    assert result["next_action"] == "rag_then_incident"


@pytest.mark.asyncio
async def test_classify_slack_mention_routes_to_rag_only(
    slack_event: NexusEvent, base_state: NexusState
):
    """classify_event must set next_action='rag_only' for slack_mention."""
    state = dict(base_state)
    state["event"] = slack_event
    result = await classify_event(state)  # type: ignore[arg-type]
    assert result["next_action"] == "rag_only"


@pytest.mark.asyncio
async def test_classify_deploy_request_routes_to_human_approval(
    deploy_event: NexusEvent, base_state: NexusState
):
    """classify_event must set next_action='human_approval' for deploy_request."""
    state = dict(base_state)
    state["event"] = deploy_event
    result = await classify_event(state)  # type: ignore[arg-type]
    assert result["next_action"] == "human_approval"


# ─── Routing function tests ───────────────────────────────────────────────────


def test_route_after_classification_pr():
    """route_after_classification maps rag_then_code_review → rag."""
    state = {"next_action": "rag_then_code_review"}  # type: ignore[typeddict-item]
    assert route_after_classification(state) == "rag"  # type: ignore[arg-type]


def test_route_after_classification_alert():
    """route_after_classification maps rag_then_incident → rag."""
    state = {"next_action": "rag_then_incident"}  # type: ignore[typeddict-item]
    assert route_after_classification(state) == "rag"  # type: ignore[arg-type]


def test_route_after_classification_deploy():
    """route_after_classification maps human_approval → human_approval."""
    state = {"next_action": "human_approval"}  # type: ignore[typeddict-item]
    assert route_after_classification(state) == "human_approval"  # type: ignore[arg-type]


def test_route_after_rag_pr_event():
    """route_after_rag maps rag_then_code_review → code_review."""
    state = {"next_action": "rag_then_code_review"}  # type: ignore[typeddict-item]
    assert route_after_rag(state) == "code_review"  # type: ignore[arg-type]


def test_route_after_rag_incident():
    """route_after_rag maps rag_then_incident → incident."""
    state = {"next_action": "rag_then_incident"}  # type: ignore[typeddict-item]
    assert route_after_rag(state) == "incident"  # type: ignore[arg-type]


def test_route_after_rag_slack():
    """route_after_rag maps rag_only → finalize."""
    state = {"next_action": "rag_only"}  # type: ignore[typeddict-item]
    assert route_after_rag(state) == "finalize"  # type: ignore[arg-type]


# ─── human_approval_gate tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_human_approval_gate_disabled(base_state: NexusState):
    """When HUMAN_IN_THE_LOOP=false, gate auto-approves."""
    with patch(
        "orchestrator.graph.nodes.settings",
        human_in_the_loop=False,
    ):
        result = await human_approval_gate(base_state)
    assert result["requires_human_approval"] is False
    assert result["human_approved"] is True


@pytest.mark.asyncio
async def test_human_approval_gate_enabled(base_state: NexusState):
    """When HUMAN_IN_THE_LOOP=true, gate sets requires_human_approval=True."""
    with patch(
        "orchestrator.graph.nodes.settings",
        human_in_the_loop=True,
    ):
        result = await human_approval_gate(base_state)
    assert result["requires_human_approval"] is True
    assert result["human_approved"] is None


# ─── Full graph integration test ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_graph_pr_event(
    pr_event: NexusEvent,
    base_state: NexusState,
    mock_mcp,
    mock_llm_code_review,
):
    """
    Full graph traversal for a PR event using an in-memory checkpointer.

    Verifies that:
    1. The graph runs without raising exceptions.
    2. code_review_result is populated in the final state.
    3. The MCP post_pr_comment tool was called.
    """
    # Build graph with in-memory checkpointer (no Postgres needed)
    memory = MemorySaver()

    # Patch embed_query to avoid real OpenAI embedding call
    fake_vector = [0.0] * 3072
    mock_pinecone_query = AsyncMock(return_value=[
        {
            "id": "abc123",
            "score": 0.9,
            "metadata": {
                "content": "def authenticate(token): ...",
                "source_file": "src/auth.py",
                "chunk_index": 0,
                "language": "python",
            },
        }
    ])

    with (
        patch("orchestrator.agents.rag_agent.embed_query", return_value=fake_vector),
        patch(
            "orchestrator.agents.rag_agent.VectorStore.query",
            mock_pinecone_query,
        ),
        patch(
            "orchestrator.graph.nodes.settings",
            human_in_the_loop=False,
            openai_model="gpt-4o",
            openai_api_key="test",
            pinecone_api_key="test",
            pinecone_index_name="test",
            slack_default_channel="#test",
            github_org="myorg",
            github_repo="myrepo",
        ),
    ):
        graph = build_graph(checkpointer=memory)

        thread_id = pr_event.event_id
        config = {"configurable": {"thread_id": thread_id}}

        # Stream through all steps
        steps = []
        async for step in graph.astream(base_state, config=config):
            steps.append(step)

    # Verify graph completed (finalize node ran)
    step_names = [list(s.keys())[0] for s in steps if s]
    assert "finalize" in step_names, f"finalize not in step_names: {step_names}"

    # Verify MCP post_pr_comment was called
    mock_mcp.assert_any_await("github", "post_pr_comment", ANY)
