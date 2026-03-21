"""
MCP client used by orchestrator agents to call MCP server tools.

Makes async HTTP POST requests to each MCP server's ``/invoke`` endpoint.
All calls are:
  - Retried up to 3 times with exponential back-off via tenacity.
  - Logged with structlog (server, tool, outcome, duration).
  - Timed and the duration is recorded on the state's ``tool_calls`` list.
"""

import time
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from orchestrator.config import get_settings
from orchestrator.graph.state import ToolCallLog

log = structlog.get_logger(__name__)

_TIMEOUT = httpx.Timeout(30.0)

# Map MCP server names to their base URLs (resolved at call time from settings)
_SERVER_MAP = {
    "github": lambda s: s.mcp_github_url,
    "slack": lambda s: s.mcp_slack_url,
    "jira": lambda s: s.mcp_jira_url,
    "datadog": lambda s: s.mcp_datadog_url,
}


class MCPClient:
    """
    Async client for communicating with Nexus MCP servers.

    Example usage::

        client = MCPClient()
        diff = await client.call("github", "get_pr_diff", {"repo": "org/repo", "pr_number": 42})
    """

    def __init__(self) -> None:
        """Initialise the client by loading settings once."""
        self._settings = get_settings()
        self._tool_call_logs: list[ToolCallLog] = []

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _post(self, url: str, payload: dict) -> Any:
        """
        Perform the raw HTTP POST to an MCP server invoke endpoint.

        Args:
            url:     Full URL of the ``/invoke`` endpoint.
            payload: Dict with ``tool`` and ``arguments`` keys.

        Returns:
            The ``result`` field from the MCP server response.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses after retries.
        """
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        return resp.json()["result"]

    async def call(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Call a named tool on a named MCP server.

        Args:
            server:    One of ``"github"``, ``"slack"``, ``"jira"``,
                       ``"datadog"``.
            tool:      Tool name as registered on the MCP server.
            arguments: Keyword arguments to pass to the tool function.

        Returns:
            The deserialised ``result`` from the MCP server's JSON response.

        Raises:
            ValueError:            If ``server`` is not a known MCP server.
            httpx.HTTPStatusError: On non-2xx responses after all retries.
            RuntimeError:          On other unexpected failures.
        """
        if server not in _SERVER_MAP:
            raise ValueError(
                f"Unknown MCP server: {server!r}. "
                f"Valid servers: {list(_SERVER_MAP)}"
            )

        base_url = _SERVER_MAP[server](self._settings)
        url = f"{base_url}/invoke"
        payload = {"tool": tool, "arguments": arguments}

        log.info("mcp_call_start", server=server, tool=tool)
        start = time.monotonic()
        outcome = "success"
        error_message = None

        try:
            result = await self._post(url, payload)
            return result
        except Exception as exc:
            outcome = "error"
            error_message = str(exc)
            log.error(
                "mcp_call_failed",
                server=server,
                tool=tool,
                error=str(exc),
            )
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            self._tool_call_logs.append(
                ToolCallLog(
                    server=server,
                    tool=tool,
                    arguments=arguments,
                    outcome=outcome,  # type: ignore[arg-type]
                    error_message=error_message,
                    duration_ms=round(duration_ms, 2),
                )
            )
            log.info(
                "mcp_call_complete",
                server=server,
                tool=tool,
                outcome=outcome,
                duration_ms=round(duration_ms, 2),
            )

    def drain_logs(self) -> list[ToolCallLog]:
        """
        Return and clear all accumulated tool call logs.

        Call this after each agent run to collect logs for the state's
        ``tool_calls`` accumulator field.
        """
        logs = list(self._tool_call_logs)
        self._tool_call_logs.clear()
        return logs
