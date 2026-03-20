"""
LangGraph StateGraph definition for the Nexus orchestrator.

Builds the compiled graph that drives all agent execution.  The graph is
a singleton compiled once at startup and shared across all request handlers.

Node topology:
    START → classify → [conditional] → rag → [conditional] → code_review / incident / finalize
                                     → human_approval → [conditional] → deploy / finalize
                                                                       → finalize → END
"""

from typing import Optional

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from orchestrator.config import get_settings
from orchestrator.graph.nodes import (
    classify_event,
    code_review_node,
    deploy_node,
    finalize_node,
    human_approval_gate,
    incident_node,
    rag_node,
    route_after_classification,
    route_after_human_approval,
    route_after_rag,
)
from orchestrator.graph.state import NexusState

log = structlog.get_logger(__name__)
settings = get_settings()


def build_graph(checkpointer: Optional[AsyncPostgresSaver] = None) -> StateGraph:
    """
    Construct and return the compiled Nexus LangGraph StateGraph.

    Args:
        checkpointer: An initialised AsyncPostgresSaver instance.
                      Pass ``None`` to use an in-memory checkpointer
                      (useful for unit tests).

    Returns:
        The compiled graph ready to be called with ``.astream()`` or
        ``.ainvoke()``.
    """
    graph = StateGraph(NexusState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("classify", classify_event)
    graph.add_node("rag", rag_node)
    graph.add_node("code_review", code_review_node)
    graph.add_node("incident", incident_node)
    graph.add_node("human_approval", human_approval_gate)
    graph.add_node("deploy", deploy_node)
    graph.add_node("finalize", finalize_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "classify")

    # ── Routing after classify ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "rag": "rag",
            "human_approval": "human_approval",
            "finalize": "finalize",
        },
    )

    # ── Routing after RAG ─────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "code_review": "code_review",
            "incident": "incident",
            "finalize": "finalize",
        },
    )

    # ── Terminal agents → finalize ─────────────────────────────────────────────
    graph.add_edge("code_review", "finalize")
    graph.add_edge("incident", "finalize")

    # ── Human approval gate ────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        {
            "deploy": "deploy",
            "finalize": "finalize",
        },
    )

    graph.add_edge("deploy", "finalize")

    # ── End ────────────────────────────────────────────────────────────────────
    graph.add_edge("finalize", END)

    # ── Compile ────────────────────────────────────────────────────────────────
    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    # Interrupt before human_approval when HUMAN_IN_THE_LOOP is enabled so
    # the graph pauses and waits for external approval before deploying.
    if settings.human_in_the_loop:
        compile_kwargs["interrupt_before"] = ["human_approval"]

    compiled = graph.compile(**compile_kwargs)

    log.info(
        "graph_compiled",
        human_in_the_loop=settings.human_in_the_loop,
        checkpointer=type(checkpointer).__name__ if checkpointer else "None",
    )
    return compiled
