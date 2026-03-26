"""
LangSmith Evaluation Harness for Nexus.

Runs evaluation datasets through the real CodeReviewAgent and IncidentResponseAgent
(with mocked MCP calls) and measures correctness against expected outputs.

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --output-file results.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Add services to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "orchestrator" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "rag-indexer" / "src"))

from langsmith import Client
from langsmith.evaluation import evaluate

from orchestrator.agents.code_review import CodeReviewAgent
from orchestrator.agents.incident import IncidentResponseAgent
from orchestrator.graph.state import NexusEvent, NexusState, RAGContext, RetrievedChunk

# ─── Dataset paths ────────────────────────────────────────────────────────────

_DATASETS_DIR = Path(__file__).parent / "datasets"
_CODE_REVIEW_FILE = _DATASETS_DIR / "code_review_eval.jsonl"
_INCIDENT_FILE = _DATASETS_DIR / "incident_eval.jsonl"


# ─── Load datasets ────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file and return a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ─── Target functions (invoke real agents with mocked MCP) ───────────────────


async def _run_code_review(inputs: dict) -> dict:
    """
    Target function for code review evaluation.

    Invokes CodeReviewAgent with the given diff and context.
    MCP calls are mocked so no real GitHub calls are made.
    """
    diff = inputs["diff"]
    context_text = inputs.get("context", "")

    # Build a minimal NexusState
    state: NexusState = {
        "event": NexusEvent(
            event_type="pr_opened",
            source="github",
            payload={
                "pull_request": {"number": 1, "title": "Eval PR", "body": context_text},
                "repository": {"full_name": "eval/repo"},
            },
            event_id="eval-code-review",
            timestamp="2026-03-25T00:00:00Z",
        ),
        "messages": [],
        "current_agent": "",
        "iteration_count": 0,
        "rag_context": RAGContext(
            query=context_text,
            retrieved_chunks=[],
        ),
        "code_review_result": None,
        "incident_result": None,
        "deploy_result": None,
        "next_action": "",
        "error": None,
        "requires_human_approval": False,
        "human_approved": None,
        "tool_calls": [],
    }

    async def mock_mcp_call(server: str, tool: str, arguments: dict) -> Any:
        if tool == "get_pr_diff":
            return diff
        if tool == "post_pr_comment":
            return {"id": 1}
        return {}

    agent = CodeReviewAgent()
    agent.mcp = MagicMock()
    agent.mcp.call = AsyncMock(side_effect=mock_mcp_call)
    agent.mcp.drain_logs = MagicMock(return_value=[])

    result_dict = await agent.run(state)
    result = result_dict["code_review_result"]

    return {
        "approved": result.approved,
        "issues": [
            {
                "severity": issue.severity,
                "file": issue.file,
                "line": issue.line,
                "message": issue.message,
            }
            for issue in result.issues
        ],
        "summary": result.summary,
    }


def code_review_target(inputs: dict) -> dict:
    """Synchronous wrapper for the async code review target."""
    return asyncio.run(_run_code_review(inputs))


async def _run_incident(inputs: dict) -> dict:
    """
    Target function for incident response evaluation.

    Invokes IncidentResponseAgent with the given alert and runbook chunks.
    MCP calls are mocked.
    """
    alert = inputs["alert"]
    runbook_chunks_text = inputs.get("runbook_chunks", [])

    chunks = [
        RetrievedChunk(
            content=text,
            source_file=f"runbooks/chunk_{i}.md",
            chunk_index=i,
            score=0.9 - (i * 0.05),
            language="markdown",
        )
        for i, text in enumerate(runbook_chunks_text)
    ]

    state: NexusState = {
        "event": NexusEvent(
            event_type="alert_fired",
            source="datadog",
            payload={"alert_id": alert["id"], **alert},
            event_id=f"eval-incident-{alert['id']}",
            timestamp="2026-03-25T00:00:00Z",
        ),
        "messages": [],
        "current_agent": "",
        "iteration_count": 0,
        "rag_context": RAGContext(
            query=alert["name"],
            retrieved_chunks=chunks,
        ),
        "code_review_result": None,
        "incident_result": None,
        "deploy_result": None,
        "next_action": "",
        "error": None,
        "requires_human_approval": False,
        "human_approved": None,
        "tool_calls": [],
    }

    async def mock_mcp_call(server: str, tool: str, arguments: dict) -> Any:
        if tool == "get_alert":
            return alert
        if tool == "post_message":
            return {"ts": "12345", "channel": "#eval"}
        if tool == "create_issue":
            return {"id": "10001", "key": "EVAL-1", "url": "http://jira.example.com/EVAL-1"}
        return {}

    agent = IncidentResponseAgent()
    agent.mcp = MagicMock()
    agent.mcp.call = AsyncMock(side_effect=mock_mcp_call)
    agent.mcp.drain_logs = MagicMock(return_value=[])

    result_dict = await agent.run(state)
    result = result_dict["incident_result"]

    return {
        "severity": result.severity,
        "root_cause": result.root_cause,
        "suggested_fix": result.suggested_fix,
        "jira_created": result.jira_ticket_id is not None,
        "slack_posted": result.slack_posted,
    }


def incident_target(inputs: dict) -> dict:
    """Synchronous wrapper for the async incident target."""
    return asyncio.run(_run_incident(inputs))


# ─── Evaluator functions ──────────────────────────────────────────────────────


def code_review_correctness_evaluator(run: Any, example: Any) -> dict:
    """
    Evaluate a code review result against the expected output.

    Checks:
    1. ``approved`` field matches expected.
    2. All expected severity levels appear in the predicted issues.

    Returns:
        Dict with keys ``key`` and ``score`` (0.0 or 1.0).
    """
    predicted = run.outputs or {}
    expected = example.outputs or {}

    predicted_approved = predicted.get("approved")
    expected_approved = expected.get("approved")

    if predicted_approved != expected_approved:
        return {
            "key": "correctness",
            "score": 0.0,
            "comment": (
                f"approved mismatch: expected={expected_approved}, "
                f"predicted={predicted_approved}"
            ),
        }

    predicted_severities = {i.get("severity") for i in predicted.get("issues", [])}
    expected_issues = expected.get("issues", [])

    for exp_issue in expected_issues:
        exp_sev = exp_issue.get("severity")
        if exp_sev not in predicted_severities:
            return {
                "key": "correctness",
                "score": 0.0,
                "comment": (
                    f"Expected severity '{exp_sev}' not found. "
                    f"Found: {predicted_severities}"
                ),
            }
        exp_msg_keyword = exp_issue.get("message", "").lower()
        matching = [
            i for i in predicted.get("issues", [])
            if i.get("severity") == exp_sev
            and any(kw in i.get("message", "").lower() for kw in exp_msg_keyword.split())
        ]
        if not matching:
            return {
                "key": "correctness",
                "score": 0.5,
                "comment": (
                    f"Severity '{exp_sev}' found but keyword '{exp_msg_keyword}' "
                    f"not in any issue message."
                ),
            }

    return {"key": "correctness", "score": 1.0, "comment": "All checks passed"}


def incident_correctness_evaluator(run: Any, example: Any) -> dict:
    """
    Evaluate an incident response against the expected output.

    Checks:
    1. Severity matches expected.
    2. Root cause contains at least one expected keyword.
    3. Suggested fix contains at least one expected keyword.
    4. Slack was posted and Jira ticket was created.
    """
    predicted = run.outputs or {}
    expected = example.outputs or {}

    score = 1.0
    comments = []

    # Severity check
    if predicted.get("severity") != expected.get("severity"):
        score -= 0.25
        comments.append(
            f"severity: expected={expected.get('severity')}, "
            f"predicted={predicted.get('severity')}"
        )

    # Root cause keyword check
    root_cause = predicted.get("root_cause", "").lower()
    rc_keywords = expected.get("root_cause_keywords", [])
    if rc_keywords and not any(kw in root_cause for kw in rc_keywords):
        score -= 0.25
        comments.append(f"root_cause missing keywords: {rc_keywords}")

    # Suggested fix keyword check
    suggested_fix = predicted.get("suggested_fix", "").lower()
    fix_keywords = expected.get("suggested_fix_keywords", [])
    if fix_keywords and not any(kw in suggested_fix for kw in fix_keywords):
        score -= 0.25
        comments.append(f"suggested_fix missing keywords: {fix_keywords}")

    # Slack + Jira check
    if not predicted.get("slack_posted"):
        score -= 0.15
        comments.append("slack was not posted")
    if not predicted.get("jira_created"):
        score -= 0.1
        comments.append("jira was not created")

    return {
        "key": "correctness",
        "score": max(0.0, score),
        "comment": "; ".join(comments) if comments else "All checks passed",
    }


# ─── Main runner ──────────────────────────────────────────────────────────────


def print_results_table(results: list[dict], dataset_name: str) -> None:
    """Print a human-readable summary table of eval results."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")
    print(f"{'Test':<8} {'Score':>8} {'Comment'}")
    print(f"{'-'*60}")

    scores = []
    for i, result in enumerate(results):
        score = result.get("score", 0.0)
        comment = result.get("comment", "")[:40]
        status = "✅" if score == 1.0 else ("⚠️ " if score > 0.5 else "❌")
        print(f"  {i+1:<6} {status} {score:>5.2f}  {comment}")
        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"{'-'*60}")
    print(f"  Average score: {avg:.2f} ({sum(1 for s in scores if s == 1.0)}/{len(scores)} perfect)")


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Nexus LangSmith Evaluation Harness")
    parser.add_argument("--output-file", help="Write results to JSON file")
    args = parser.parse_args()

    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "nexus-evals")

    print("🧪 Nexus LangSmith Evaluation Harness")
    print(f"   Project: {langsmith_project}")
    print()

    all_results: dict[str, Any] = {}

    # ── Code Review Evals ────────────────────────────────────────────────────
    print("Running code review evaluations...")
    cr_cases = load_jsonl(_CODE_REVIEW_FILE)
    cr_results = []

    for i, case in enumerate(cr_cases):
        try:
            predicted = code_review_target(case["input"])
            run_mock = type("Run", (), {"outputs": predicted})()
            example_mock = type("Example", (), {"outputs": case["expected"]})()
            eval_result = code_review_correctness_evaluator(run_mock, example_mock)
            cr_results.append(eval_result)
        except Exception as exc:
            cr_results.append({
                "key": "correctness",
                "score": 0.0,
                "comment": f"Exception: {str(exc)[:80]}",
            })

    print_results_table(cr_results, "Code Review Evaluations")
    all_results["code_review"] = cr_results

    # ── Incident Evals ───────────────────────────────────────────────────────
    print("\nRunning incident response evaluations...")
    inc_cases = load_jsonl(_INCIDENT_FILE)
    inc_results = []

    for i, case in enumerate(inc_cases):
        try:
            predicted = incident_target(case["input"])
            run_mock = type("Run", (), {"outputs": predicted})()
            example_mock = type("Example", (), {"outputs": case["expected"]})()
            eval_result = incident_correctness_evaluator(run_mock, example_mock)
            inc_results.append(eval_result)
        except Exception as exc:
            inc_results.append({
                "key": "correctness",
                "score": 0.0,
                "comment": f"Exception: {str(exc)[:80]}",
            })

    print_results_table(inc_results, "Incident Response Evaluations")
    all_results["incident"] = inc_results

    # ── Summary ──────────────────────────────────────────────────────────────
    all_scores = [
        r.get("score", 0.0)
        for results in all_results.values()
        for r in results
    ]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    perfect_count = sum(1 for s in all_scores if s == 1.0)

    print(f"\n{'='*60}")
    print(f"  OVERALL: {overall_avg:.2f} avg score ({perfect_count}/{len(all_scores)} perfect)")
    print(f"{'='*60}\n")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results written to: {args.output_file}")

    # Exit non-zero if average score < 0.7
    if overall_avg < 0.7:
        sys.exit(1)


if __name__ == "__main__":
    main()
