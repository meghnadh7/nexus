"""
Jira REST API tool implementations for the mcp-jira MCP server.

Uses Jira's REST API v3 with Basic auth (email + API token).
All functions are async, retried with tenacity.
"""

import base64
from typing import Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_TIMEOUT = httpx.Timeout(30.0)


def _auth_headers(email: str, api_token: str) -> dict[str, str]:
    """Return Basic-auth headers for the Jira REST API."""
    creds = base64.b64encode(f"{email}:{api_token}".encode()).decode()
    return {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _adf_doc(text: str) -> dict:
    """
    Wrap plain text into Atlassian Document Format (ADF) for Jira description.

    Args:
        text: Plain text content.

    Returns:
        ADF document dict compatible with Jira REST API v3.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    content = [
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": para}],
        }
        for para in paragraphs
    ]
    return {"version": 1, "type": "doc", "content": content}


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def create_issue(
    base_url: str,
    email: str,
    api_token: str,
    project_key: str,
    summary: str,
    description: str,
    priority: str = "High",
    labels: Optional[list[str]] = None,
    issue_type: str = "Bug",
) -> dict:
    """
    Create a new Jira issue with ADF-formatted description.

    Args:
        base_url:    Jira instance base URL (e.g. ``https://company.atlassian.net``).
        email:       Jira user email for Basic auth.
        api_token:   Jira API token.
        project_key: Jira project key (e.g. ``ENG``).
        summary:     Issue title.
        description: Issue body (converted to ADF automatically).
        priority:    Priority name (e.g. ``High``, ``Critical``).
        labels:      List of label strings to attach.
        issue_type:  Issue type name (default: ``Bug``).

    Returns:
        Dict with keys ``id``, ``key``, ``url``.
    """
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": _adf_doc(description),
            "issuetype": {"name": issue_type},
            "priority": {"name": priority},
            "labels": labels or [],
        }
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/rest/api/3/issue",
            headers=_auth_headers(email, api_token),
            json=payload,
        )
        resp.raise_for_status()
    data = resp.json()
    ticket_key = data["key"]
    log.info("jira_issue_created", key=ticket_key, project=project_key)
    return {
        "id": data["id"],
        "key": ticket_key,
        "url": f"{base_url}/browse/{ticket_key}",
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def update_issue_status(
    base_url: str,
    email: str,
    api_token: str,
    issue_key: str,
    status: str,
) -> dict:
    """
    Transition a Jira issue to the given status name.

    Fetches available transitions first, matches by name (case-insensitive),
    then performs the transition.

    Args:
        base_url:  Jira instance base URL.
        email:     Jira user email.
        api_token: Jira API token.
        issue_key: Issue key (e.g. ``ENG-123``).
        status:    Target status name (e.g. ``In Progress``, ``Done``).

    Returns:
        Dict with key ``transitioned_to``.
    """
    headers = _auth_headers(email, api_token)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        trans_resp = await client.get(
            f"{base_url}/rest/api/3/issue/{issue_key}/transitions",
            headers=headers,
        )
        trans_resp.raise_for_status()
        transitions = trans_resp.json().get("transitions", [])

        matching = [
            t for t in transitions
            if t["name"].lower() == status.lower()
        ]
        if not matching:
            available = [t["name"] for t in transitions]
            raise ValueError(
                f"Transition {status!r} not found for {issue_key}. "
                f"Available: {available}"
            )

        transition_id = matching[0]["id"]
        do_resp = await client.post(
            f"{base_url}/rest/api/3/issue/{issue_key}/transitions",
            headers=headers,
            json={"transition": {"id": transition_id}},
        )
        do_resp.raise_for_status()

    log.info("jira_status_updated", issue=issue_key, status=status)
    return {"transitioned_to": status}


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def add_comment(
    base_url: str,
    email: str,
    api_token: str,
    issue_key: str,
    body: str,
) -> dict:
    """
    Add a comment to an existing Jira issue.

    Args:
        base_url:  Jira instance base URL.
        email:     Jira user email.
        api_token: Jira API token.
        issue_key: Issue key to comment on.
        body:      Comment text (converted to ADF).

    Returns:
        Dict with keys ``id`` and ``created``.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/rest/api/3/issue/{issue_key}/comment",
            headers=_auth_headers(email, api_token),
            json={"body": _adf_doc(body)},
        )
        resp.raise_for_status()
    data = resp.json()
    log.info("jira_comment_added", issue=issue_key, comment_id=data.get("id"))
    return {"id": data["id"], "created": data.get("created", "")}


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def search_issues(
    base_url: str,
    email: str,
    api_token: str,
    jql: str,
    max_results: int = 50,
) -> list[dict]:
    """
    Search Jira issues using JQL.

    Args:
        base_url:    Jira instance base URL.
        email:       Jira user email.
        api_token:   Jira API token.
        jql:         JQL query string.
        max_results: Maximum number of results (default 50).

    Returns:
        List of dicts with keys ``key``, ``summary``, ``status``, ``assignee``.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            f"{base_url}/rest/api/3/search",
            headers=_auth_headers(email, api_token),
            params={
                "jql": jql,
                "maxResults": max_results,
                "fields": "summary,status,assignee",
            },
        )
        resp.raise_for_status()
    issues = resp.json().get("issues", [])
    return [
        {
            "key": issue["key"],
            "summary": issue["fields"].get("summary", ""),
            "status": issue["fields"].get("status", {}).get("name", ""),
            "assignee": (
                issue["fields"].get("assignee") or {}
            ).get("displayName", "Unassigned"),
        }
        for issue in issues
    ]
