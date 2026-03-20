"""
LangGraph state schema for the Nexus multi-agent system.

NexusState is the shared TypedDict that every node reads from and writes to.
Annotated fields use LangGraph reducers so that concurrent node updates are
merged correctly rather than overwritten.
"""

import operator
from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─── Domain Event ─────────────────────────────────────────────────────────────


class NexusEvent(BaseModel):
    """Incoming trigger event that started this agent run."""

    event_type: Literal[
        "pr_opened", "pr_updated", "alert_fired", "slack_mention", "deploy_request"
    ]
    source: Literal["github", "datadog", "slack", "manual"]
    payload: dict = Field(default_factory=dict)
    event_id: str
    timestamp: str


# ─── RAG Context ──────────────────────────────────────────────────────────────


class RetrievedChunk(BaseModel):
    """A single document chunk returned from Pinecone."""

    content: str
    source_file: str
    chunk_index: int
    score: float
    language: str = ""


class RAGContext(BaseModel):
    """Results from the RAG retrieval phase."""

    query: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    synthesized_answer: Optional[str] = None


# ─── Code Review Result ───────────────────────────────────────────────────────


class CodeIssue(BaseModel):
    """A single issue found during code review."""

    severity: Literal["critical", "high", "medium", "low"]
    file: str
    line: int
    message: str
    suggestion: str


class CodeReviewResult(BaseModel):
    """Output produced by the CodeReviewAgent."""

    pr_number: int
    repo: str
    issues: list[CodeIssue] = Field(default_factory=list)
    summary: str
    approved: bool
    posted_to_github: bool = False


# ─── Incident Result ──────────────────────────────────────────────────────────


class IncidentResult(BaseModel):
    """Output produced by the IncidentResponseAgent."""

    alert_id: str
    severity: Literal["p1", "p2", "p3", "p4"]
    root_cause: str
    suggested_fix: str
    runbook_references: list[str] = Field(default_factory=list)
    jira_ticket_id: Optional[str] = None
    slack_posted: bool = False


# ─── Deploy Result ────────────────────────────────────────────────────────────


class DeployResult(BaseModel):
    """Output produced by the DeployAgent."""

    workflow_run_id: str
    status: Literal["pending", "running", "success", "failure", "timed_out"]
    environment: str
    commit_sha: str


# ─── Tool Call Log ────────────────────────────────────────────────────────────


class ToolCallLog(BaseModel):
    """Audit record for every MCP tool invocation."""

    server: str
    tool: str
    arguments: dict
    outcome: Literal["success", "error"]
    error_message: Optional[str] = None
    duration_ms: float


# ─── Main State TypedDict ─────────────────────────────────────────────────────


class NexusState(TypedDict):
    """
    Shared mutable state passed through every node in the LangGraph graph.

    Annotated fields use reducers:
      - messages  →  add_messages  (append-only conversation history)
      - tool_calls → operator.add  (accumulate all tool call logs)
    """

    # The event that triggered this run
    event: NexusEvent

    # Full conversation history including LLM messages
    messages: Annotated[list, add_messages]

    # Which agent/node is currently executing
    current_agent: str

    # Safety counter – graph raises if this exceeds MAX_AGENT_ITERATIONS
    iteration_count: int

    # Pre-fetched retrieval-augmented context
    rag_context: Optional[RAGContext]

    # Domain-specific results (populated by the relevant agent)
    code_review_result: Optional[CodeReviewResult]
    incident_result: Optional[IncidentResult]
    deploy_result: Optional[DeployResult]

    # Routing signal written by classify / agents, read by conditional edges
    next_action: str

    # Non-fatal error message if something went wrong but run should continue
    error: Optional[str]

    # Human-in-the-loop control fields
    requires_human_approval: bool
    human_approved: Optional[bool]

    # Accumulated audit log of every MCP tool call in this run
    tool_calls: Annotated[list[ToolCallLog], operator.add]
