"""
Abstract base class for all Nexus agents.

Every agent receives the shared NexusState, performs its work (typically
one or more LLM calls + MCP tool calls), and returns a dict of state
updates to merge back into the graph.
"""

from abc import ABC, abstractmethod

import structlog
from langchain_openai import ChatOpenAI

from orchestrator.config import get_settings
from orchestrator.graph.state import NexusState
from orchestrator.tools.mcp_client import MCPClient


class BaseAgent(ABC):
    """
    Abstract base agent.

    Subclasses must implement ``run(state)`` which returns a partial
    NexusState dict to be merged by LangGraph.
    """

    def __init__(self) -> None:
        """
        Initialise shared infrastructure: LLM client, MCP client, logger.

        The LLM is configured with ``temperature=0`` and ``max_retries=3``
        so all agent completions are deterministic and resilient to transient
        OpenAI errors.
        """
        settings = get_settings()
        self.settings = settings
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            max_retries=3,
            api_key=settings.openai_api_key,
        )
        self.mcp = MCPClient()
        self.log = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    async def run(self, state: NexusState) -> dict:
        """
        Execute the agent's logic and return state updates.

        Args:
            state: The current shared NexusState snapshot.

        Returns:
            Partial state dict whose keys will be merged into NexusState
            by the LangGraph node wrapper.
        """
        ...

    def _drain_tool_logs(self) -> list:
        """
        Collect accumulated MCP tool call logs from this invocation.

        Returns:
            List of ToolCallLog instances to be appended to state.tool_calls.
        """
        return self.mcp.drain_logs()
