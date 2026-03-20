"""
LangGraph node functions for the Nexus orchestrator graph.

Each function maps to a named node in the StateGraph. They receive the
full NexusState and return a dict of state updates to merge.
"""

import time

import structlog

from orchestrator.agents.code_review import CodeReviewAgent
from orchestrator.agents.deploy import DeployAgent
from orchestrator.agents.incident import IncidentResponseAgent
from orchestrator.agents.rag_agent import RAGAgent
from orchestrator.config import get_settings
from orchestrator.graph.state import NexusState

log = structlog.get_logger(__name__)
settings = get_settings()

# ─── Pre-instantiate agents (singletons within the process) ──────────────────
_rag_agent = RAGAgent()
_code_review_agent = CodeReviewAgent()
_incident_agent = IncidentResponseAgent()
_deploy_agent = DeployAgent()


# ─── Classify node ────────────────────────────────────────────────────────────


async def classify_event(state: NexusState) -> dict:
    """
    Map the incoming event type to the ``next_action`` routing signal.

    This is always the entry node. It does no I/O; it simply reads the
    event type and sets ``next_action`` so conditional edges can route
    to the correct subgraph.

    Args:
        state: Current NexusState.

    Returns:
        State update dict with ``next_action`` and ``current_agent``.
    """
    event = state["event"]
    event_type = event.event_type

    routing_map = {
        "pr_opened": "rag_then_code_review",
        "pr_updated": "rag_then_code_review",
        "alert_fired": "rag_then_incident",
        "slack_mention": "rag_only",
        "deploy_request": "human_approval",
    }

    next_action = routing_map.get(event_type, "finalize")

    log.info(
        "event_classified",
        event_id=event.event_id,
        event_type=event_type,
        next_action=next_action,
    )

    return {
        "next_action": next_action,
        "current_agent": "classify",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def route_after_classification(state: NexusState) -> str:
    """
    Conditional edge function: return the next node name based on ``next_action``.

    This function is passed to ``add_conditional_edges`` as the router.

    Args:
        state: Current NexusState after classify_event ran.

    Returns:
        Name of the next node to execute.
    """
    next_action = state.get("next_action", "finalize")

    route_map = {
        "rag_then_code_review": "rag",
        "rag_then_incident": "rag",
        "rag_only": "rag",
        "human_approval": "human_approval",
        "finalize": "finalize",
    }

    return route_map.get(next_action, "finalize")


def route_after_rag(state: NexusState) -> str:
    """
    Conditional edge function: route after RAG completes.

    For ``slack_mention`` events RAG handles the reply itself, so route to
    finalize. For PR and alert events, route to the appropriate action agent.

    Args:
        state: Current NexusState after rag node ran.

    Returns:
        Name of the next node to execute.
    """
    next_action = state.get("next_action", "finalize")

    if next_action == "rag_then_code_review":
        return "code_review"
    if next_action == "rag_then_incident":
        return "incident"
    # slack_mention and fallback
    return "finalize"


def route_after_human_approval(state: NexusState) -> str:
    """
    Conditional edge function: proceed to deploy if approved, else finalize.

    Args:
        state: Current NexusState after human_approval node ran.

    Returns:
        ``"deploy"`` or ``"finalize"``.
    """
    if state.get("human_approved") is True or not settings.human_in_the_loop:
        return "deploy"
    return "finalize"


# ─── RAG node ─────────────────────────────────────────────────────────────────


async def rag_node(state: NexusState) -> dict:
    """
    Execute the RAG agent to pre-fetch codebase context.

    Args:
        state: Current NexusState.

    Returns:
        State update from RAGAgent.run().
    """
    return await _rag_agent.run(state)


# ─── Code Review node ─────────────────────────────────────────────────────────


async def code_review_node(state: NexusState) -> dict:
    """
    Execute the CodeReviewAgent on the current PR.

    Args:
        state: Current NexusState (must have rag_context populated).

    Returns:
        State update from CodeReviewAgent.run().
    """
    return await _code_review_agent.run(state)


# ─── Incident node ────────────────────────────────────────────────────────────


async def incident_node(state: NexusState) -> dict:
    """
    Execute the IncidentResponseAgent for a fired Datadog alert.

    Args:
        state: Current NexusState (must have rag_context populated).

    Returns:
        State update from IncidentResponseAgent.run().
    """
    return await _incident_agent.run(state)


# ─── Human Approval node ──────────────────────────────────────────────────────


async def human_approval_gate(state: NexusState) -> dict:
    """
    Gate that either auto-approves (HUMAN_IN_THE_LOOP=false) or waits for
    a human to resume the graph (HUMAN_IN_THE_LOOP=true).

    When HUMAN_IN_THE_LOOP=true the graph is compiled with
    ``interrupt_before=["human_approval"]`` so execution pauses here until
    ``graph.update_state(thread_id, {"human_approved": True})`` is called
    externally (e.g. via a Slack approval command).

    Args:
        state: Current NexusState.

    Returns:
        State update dict with ``requires_human_approval`` and
        ``human_approved``.
    """
    if not settings.human_in_the_loop:
        log.info("human_approval_bypassed", reason="HUMAN_IN_THE_LOOP=false")
        return {
            "requires_human_approval": False,
            "human_approved": True,
            "current_agent": "human_approval",
        }

    log.info("human_approval_required", event_id=state["event"].event_id)
    return {
        "requires_human_approval": True,
        "human_approved": None,  # Will be set externally before resuming
        "current_agent": "human_approval",
    }


# ─── Deploy node ──────────────────────────────────────────────────────────────


async def deploy_node(state: NexusState) -> dict:
    """
    Execute the DeployAgent to trigger and monitor the CD workflow.

    Args:
        state: Current NexusState (human_approved must be True).

    Returns:
        State update from DeployAgent.run().
    """
    return await _deploy_agent.run(state)


# ─── Finalize node ────────────────────────────────────────────────────────────


async def finalize_node(state: NexusState) -> dict:
    """
    Terminal node — log completion and set final routing signal.

    Records a structured completion log with the outcome of the run.
    No I/O, no agent calls.

    Args:
        state: Current NexusState at graph completion.

    Returns:
        State update dict marking the run as complete.
    """
    event = state["event"]

    # Determine the outcome for logging
    if state.get("code_review_result"):
        outcome = "code_review"
        detail = {
            "approved": state["code_review_result"].approved,
            "issues": len(state["code_review_result"].issues),
        }
    elif state.get("incident_result"):
        outcome = "incident_response"
        detail = {
            "severity": state["incident_result"].severity,
            "jira": state["incident_result"].jira_ticket_id,
        }
    elif state.get("deploy_result"):
        outcome = "deploy"
        detail = {
            "status": state["deploy_result"].status,
            "environment": state["deploy_result"].environment,
        }
    elif state.get("rag_context"):
        outcome = "rag_only"
        detail = {"chunks": len(state["rag_context"].retrieved_chunks)}
    else:
        outcome = "unknown"
        detail = {}

    log.info(
        "graph_finalized",
        event_id=event.event_id,
        event_type=event.event_type,
        outcome=outcome,
        tool_calls_total=len(state.get("tool_calls", [])),
        **detail,
    )

    return {"next_action": "END", "current_agent": "finalize"}
