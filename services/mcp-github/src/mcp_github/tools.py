"""
GitHub API tool implementations for the mcp-github MCP server.

Every function uses httpx.AsyncClient with tenacity retries and proper
error propagation so callers receive descriptive exceptions.
"""

import re

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_TIMEOUT = httpx.Timeout(30.0)


def _make_headers(token: str) -> dict[str, str]:
    """Build standard GitHub API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _make_diff_headers(token: str) -> dict[str, str]:
    """Build headers that request the unified diff media type."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def get_pr_diff(token: str, repo: str, pr_number: int) -> str:
    """
    Fetch the unified diff for a pull request.

    Args:
        token:     GitHub personal access token.
        repo:      Full repo name, e.g. ``owner/repo``.
        pr_number: Pull request number.

    Returns:
        The raw unified-diff string.
    """
    url = f"{_GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, headers=_make_diff_headers(token))
        resp.raise_for_status()
    log.info("pr_diff_fetched", repo=repo, pr_number=pr_number, bytes=len(resp.text))
    return resp.text


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def post_pr_comment(token: str, repo: str, pr_number: int, body: str) -> dict:
    """
    Post a comment on a pull request.

    Args:
        token:     GitHub personal access token.
        repo:      Full repo name.
        pr_number: Pull request number.
        body:      Markdown-formatted comment body.

    Returns:
        The created comment object from the GitHub API.
    """
    url = f"{_GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            url, headers=_make_headers(token), json={"body": body}
        )
        resp.raise_for_status()
    data: dict = resp.json()
    log.info("pr_comment_posted", repo=repo, pr_number=pr_number, comment_id=data.get("id"))
    return data


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def trigger_workflow(
    token: str,
    repo: str,
    workflow_id: str,
    ref: str,
    inputs: dict,
) -> str:
    """
    Dispatch a GitHub Actions workflow and return the newly created run ID.

    Args:
        token:       GitHub personal access token.
        repo:        Full repo name.
        workflow_id: Workflow filename or numeric ID.
        ref:         Git ref (branch/tag/SHA) to run against.
        inputs:      Workflow input key-value pairs.

    Returns:
        The workflow run ID as a string.
    """
    dispatch_url = f"{_GITHUB_API}/repos/{repo}/actions/workflows/{workflow_id}/dispatches"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            dispatch_url,
            headers=_make_headers(token),
            json={"ref": ref, "inputs": inputs},
        )
        resp.raise_for_status()  # 204 No Content on success

        # GitHub does not return the run ID from the dispatch endpoint; we
        # must poll the runs list immediately and grab the newest item.
        runs_url = f"{_GITHUB_API}/repos/{repo}/actions/workflows/{workflow_id}/runs"
        runs_resp = await client.get(
            runs_url,
            headers=_make_headers(token),
            params={"per_page": 1, "event": "workflow_dispatch"},
        )
        runs_resp.raise_for_status()
        runs = runs_resp.json().get("workflow_runs", [])
        if not runs:
            raise RuntimeError("Workflow dispatched but no run found in list")
        run_id = str(runs[0]["id"])

    log.info("workflow_triggered", repo=repo, workflow_id=workflow_id, run_id=run_id)
    return run_id


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def get_workflow_run(token: str, repo: str, run_id: str) -> dict:
    """
    Return the status and conclusion of a workflow run.

    Args:
        token:  GitHub personal access token.
        repo:   Full repo name.
        run_id: Workflow run numeric ID.

    Returns:
        Dict with keys ``id``, ``status``, ``conclusion``, ``html_url``.
    """
    url = f"{_GITHUB_API}/repos/{repo}/actions/runs/{run_id}"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, headers=_make_headers(token))
        resp.raise_for_status()
    data = resp.json()
    return {
        "id": str(data["id"]),
        "status": data["status"],
        "conclusion": data.get("conclusion"),
        "html_url": data.get("html_url", ""),
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def list_open_prs(token: str, repo: str) -> list[dict]:
    """
    List all open pull requests in a repository.

    Args:
        token: GitHub personal access token.
        repo:  Full repo name.

    Returns:
        List of dicts with keys ``number``, ``title``, ``user``, ``labels``.
    """
    url = f"{_GITHUB_API}/repos/{repo}/pulls"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            url,
            headers=_make_headers(token),
            params={"state": "open", "per_page": 100},
        )
        resp.raise_for_status()
    return [
        {
            "number": pr["number"],
            "title": pr["title"],
            "user": pr["user"]["login"],
            "labels": [lbl["name"] for lbl in pr.get("labels", [])],
        }
        for pr in resp.json()
    ]


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def get_pr_files(token: str, repo: str, pr_number: int) -> list[dict]:
    """
    List changed files in a pull request with addition/deletion counts.

    Args:
        token:     GitHub personal access token.
        repo:      Full repo name.
        pr_number: Pull request number.

    Returns:
        List of dicts with keys ``filename``, ``additions``, ``deletions``,
        ``changes``, ``status``.
    """
    url = f"{_GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            url,
            headers=_make_headers(token),
            params={"per_page": 100},
        )
        resp.raise_for_status()
    return [
        {
            "filename": f["filename"],
            "additions": f["additions"],
            "deletions": f["deletions"],
            "changes": f["changes"],
            "status": f["status"],
        }
        for f in resp.json()
    ]
