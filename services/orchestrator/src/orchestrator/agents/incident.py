"""
Incident Response Agent.

On a Datadog alert:
1. Fetch full alert details via the Datadog MCP server.
2. Use rag_context (pre-fetched runbook chunks) for remediation guidance.
3. Invoke GPT-4o to generate RCA and remediation plan as JSON.
4. Post a formatted alert to Slack.
5. Create a Jira ticket with the full RCA.
6. Return an IncidentResult to update shared state.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base import BaseAgent
from orchestrator.graph.state import IncidentResult, NexusState, RAGContext

_SEVERITY_EMOJI = {"p1": "🔴 P1", "p2": "🟠 P2", "p3": "🟡 P3", "p4": "🔵 P4"}

_SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) responding to a production
incident. You have been given a Datadog alert and relevant runbook excerpts
retrieved from the knowledge base.

Analyse the alert and produce a root cause analysis (RCA) and remediation plan.

You MUST respond with a single JSON object in exactly this format (no markdown,
no extra text):
{
  "root_cause": "<concise technical explanation of what failed and why>",
  "suggested_fix": "<step-by-step remediation actions in priority order>",
  "runbook_references": ["<file_path_1>", "<file_path_2>"],
  "severity": "p1" | "p2" | "p3" | "p4"
}

Severity guidelines:
- p1: Complete service outage, data loss risk, revenue impact > $10k/hr
- p2: Significant degradation affecting multiple users, < 30 min to resolve
- p3: Minor degradation, single non-critical component, < 4 hrs to resolve
- p4: Informational, no user impact

Be specific and actionable. Do not speculate beyond the evidence.
"""


def _format_slack_message(
    alert: dict,
    result: IncidentResult,
    jira_key: str,
) -> str:
    """
    Format a Slack incident notification message.

    Args:
        alert:    Raw Datadog alert dict.
        result:   IncidentResult with RCA.
        jira_key: Created Jira ticket key.

    Returns:
        mrkdwn-formatted Slack message string.
    """
    sev_label = _SEVERITY_EMOJI.get(result.severity, result.severity.upper())
    lines = [
        f"*{sev_label} INCIDENT DETECTED* :rotating_light:",
        f"*Alert:* {alert.get('name', 'Unknown alert')}",
        f"*State:* {alert.get('overall_state', 'Unknown')}",
        "",
        "*Root Cause:*",
        result.root_cause,
        "",
        "*Suggested Fix:*",
        result.suggested_fix,
        "",
    ]
    if result.runbook_references:
        lines.append("*Runbooks:*")
        for ref in result.runbook_references:
            lines.append(f"• `{ref}`")
        lines.append("")
    lines.append(f"*Jira:* {jira_key}")
    return "\n".join(lines)


class IncidentResponseAgent(BaseAgent):
    """
    Automated incident response agent for Datadog alerts.

    Workflow:
    1. Fetch full alert details via Datadog MCP.
    2. Use rag_context runbook chunks for grounding.
    3. GPT-4o generates structured RCA + severity + remediation.
    4. Post formatted Slack message.
    5. Create Jira ticket with RCA.
    6. Return IncidentResult.
    """

    async def run(self, state: NexusState) -> dict:
        """
        Execute the incident response pipeline.

        Args:
            state: Current NexusState with alert_fired event.

        Returns:
            State update dict with ``incident_result`` and ``tool_calls``.
        """
        event = state["event"]
        payload = event.payload
        alert_id = str(
            payload.get("alert_id")
            or payload.get("id")
            or payload.get("monitor_id", "unknown")
        )

        self.log.info("incident_response_started", alert_id=alert_id)

        # Step 1: Fetch full alert details
        alert: dict = await self.mcp.call(
            "datadog",
            "get_alert",
            {"alert_id": alert_id},
        )

        # Step 2: Build runbook context from RAG
        rag_context: RAGContext | None = state.get("rag_context")
        runbook_text = ""
        if rag_context and rag_context.retrieved_chunks:
            runbook_parts = [
                f"## {chunk.source_file}\n{chunk.content[:800]}"
                for chunk in rag_context.retrieved_chunks[:6]
            ]
            runbook_text = (
                "## Retrieved Runbooks & Context\n\n"
                + "\n\n---\n\n".join(runbook_parts)
            )

        # Step 3: Generate RCA with GPT-4o
        alert_summary = (
            f"Alert Name: {alert.get('name', 'Unknown')}\n"
            f"Type: {alert.get('type', 'Unknown')}\n"
            f"Query: {alert.get('query', 'N/A')}\n"
            f"State: {alert.get('overall_state', 'Unknown')}\n"
            f"Message: {alert.get('message', '')[:500]}\n"
            f"Tags: {', '.join(alert.get('tags', []))}"
        )

        user_message = f"## Alert Details\n{alert_summary}\n\n{runbook_text}"

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        raw = str(response.content).strip()

        # Parse JSON
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.log.error("incident_json_parse_failed", error=str(exc))
            parsed = {
                "root_cause": "RCA generation failed — manual investigation required.",
                "suggested_fix": "Check Datadog, review recent deployments.",
                "runbook_references": [],
                "severity": "p2",
            }

        incident_result = IncidentResult(
            alert_id=alert_id,
            severity=parsed.get("severity", "p2"),
            root_cause=parsed.get("root_cause", ""),
            suggested_fix=parsed.get("suggested_fix", ""),
            runbook_references=parsed.get("runbook_references", []),
        )

        # Step 4: Post to Slack
        slack_response = await self.mcp.call(
            "slack",
            "post_message",
            {
                "channel": self.settings.slack_default_channel,
                "text": _format_slack_message(alert, incident_result, "PENDING"),
            },
        )
        incident_result.slack_posted = True

        # Step 5: Create Jira ticket
        jira_description = (
            f"*Alert ID:* {alert_id}\n\n"
            f"*Alert:* {alert.get('name', 'Unknown')}\n\n"
            f"*Root Cause:*\n{incident_result.root_cause}\n\n"
            f"*Suggested Fix:*\n{incident_result.suggested_fix}\n\n"
            f"*Runbooks:*\n{chr(10).join(incident_result.runbook_references)}\n\n"
            f"*Datadog Alert State:* {alert.get('overall_state', 'Unknown')}"
        )

        priority_map = {"p1": "Critical", "p2": "High", "p3": "Medium", "p4": "Low"}
        jira_priority = priority_map.get(incident_result.severity, "High")

        jira_response = await self.mcp.call(
            "jira",
            "create_issue",
            {
                "summary": f"[{incident_result.severity.upper()}] {alert.get('name', 'Production Incident')}",
                "description": jira_description,
                "priority": jira_priority,
                "labels": ["incident", "automated", incident_result.severity],
                "issue_type": "Bug",
            },
        )

        jira_key = jira_response.get("key", "UNKNOWN")
        incident_result.jira_ticket_id = jira_key

        self.log.info(
            "incident_response_complete",
            alert_id=alert_id,
            severity=incident_result.severity,
            jira=jira_key,
        )

        return {
            "incident_result": incident_result,
            "tool_calls": self._drain_tool_logs(),
            "current_agent": "incident",
        }
