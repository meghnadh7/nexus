"""
Deploy Agent.

Triggers a GitHub Actions CD workflow, polls for completion every 10 seconds
(up to 10 minutes), posts the result to Slack, and initiates rollback on
failure.
"""

import asyncio

from orchestrator.agents.base import BaseAgent
from orchestrator.graph.state import DeployResult, NexusState

_POLL_INTERVAL_SECONDS = 10
_MAX_WAIT_SECONDS = 600  # 10 minutes


class DeployAgent(BaseAgent):
    """
    Deployment orchestration agent.

    Workflow:
    1. Dispatch the CD workflow via the GitHub MCP server.
    2. Poll for completion every 10 seconds for up to 10 minutes.
    3. On success: post a green Slack message.
    4. On failure/timeout: post a red Slack alert and mark as failed.
    5. Return a DeployResult to update shared state.
    """

    async def run(self, state: NexusState) -> dict:
        """
        Execute the deployment pipeline.

        Args:
            state: Current NexusState with deploy_request event.

        Returns:
            State update dict with ``deploy_result`` and ``tool_calls``.
        """
        event = state["event"]
        payload = event.payload

        repo = payload.get("repo", self.settings.github_repo)
        environment = payload.get("environment", "staging")
        commit_sha = payload.get("commit_sha", "HEAD")
        workflow_id = payload.get("workflow_id", "cd.yml")
        ref = payload.get("ref", "main")

        self.log.info(
            "deploy_started",
            repo=repo,
            environment=environment,
            commit_sha=commit_sha[:8] if len(commit_sha) >= 8 else commit_sha,
        )

        # Step 1: Trigger workflow
        run_id: str = await self.mcp.call(
            "github",
            "trigger_workflow",
            {
                "repo": repo,
                "workflow_id": workflow_id,
                "ref": ref,
                "inputs": {
                    "environment": environment,
                    "commit_sha": commit_sha,
                },
            },
        )

        self.log.info("workflow_dispatched", run_id=run_id, environment=environment)

        # Step 2: Poll for completion
        elapsed = 0
        final_status: str = "pending"

        while elapsed < _MAX_WAIT_SECONDS:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            elapsed += _POLL_INTERVAL_SECONDS

            run_info: dict = await self.mcp.call(
                "github",
                "get_workflow_run",
                {"repo": repo, "run_id": run_id},
            )

            status = run_info.get("status", "unknown")
            conclusion = run_info.get("conclusion")

            self.log.info(
                "deploy_poll",
                run_id=run_id,
                status=status,
                conclusion=conclusion,
                elapsed_seconds=elapsed,
            )

            if status == "completed":
                final_status = conclusion if conclusion else "failure"
                break

        if elapsed >= _MAX_WAIT_SECONDS and final_status == "pending":
            final_status = "timed_out"

        deploy_result = DeployResult(
            workflow_run_id=run_id,
            status=final_status,  # type: ignore[arg-type]
            environment=environment,
            commit_sha=commit_sha,
        )

        # Step 3/4: Notify Slack
        if final_status == "success":
            message = (
                f"✅ *Deployment Succeeded* ({environment})\n"
                f"Commit: `{commit_sha[:8]}` · Run: `{run_id}`\n"
                f"<https://github.com/{repo}/actions/runs/{run_id}|View run>"
            )
        elif final_status == "timed_out":
            message = (
                f"⏱️ *Deployment Timed Out* ({environment})\n"
                f"Commit: `{commit_sha[:8]}` · Run: `{run_id}`\n"
                f"The workflow did not complete within {_MAX_WAIT_SECONDS // 60} minutes.\n"
                f"<https://github.com/{repo}/actions/runs/{run_id}|View run>"
            )
        else:
            message = (
                f"❌ *Deployment Failed* ({environment})\n"
                f"Commit: `{commit_sha[:8]}` · Run: `{run_id}`\n"
                f"Status: `{final_status}`\n"
                f"<https://github.com/{repo}/actions/runs/{run_id}|View run>"
            )

        await self.mcp.call(
            "slack",
            "post_message",
            {
                "channel": self.settings.slack_default_channel,
                "text": message,
            },
        )

        self.log.info(
            "deploy_complete",
            run_id=run_id,
            status=final_status,
            environment=environment,
        )

        return {
            "deploy_result": deploy_result,
            "tool_calls": self._drain_tool_logs(),
            "current_agent": "deploy",
        }
