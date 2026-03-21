"""
Slack API tool implementations for the mcp-slack MCP server.

Uses the Slack Web API (https://slack.com/api/*) with Bearer token auth.
All functions are async, retried with tenacity, and return typed dicts.
"""

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

_SLACK_API = "https://slack.com/api"
_TIMEOUT = httpx.Timeout(30.0)


def _headers(token: str) -> dict[str, str]:
    """Return Slack API auth headers."""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _check_slack_ok(data: dict) -> None:
    """Raise RuntimeError if Slack returned an error envelope."""
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error: {data.get('error', 'unknown')}")


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException, RuntimeError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def post_message(
    token: str,
    channel: str,
    text: str,
    thread_ts: Optional[str] = None,
) -> dict:
    """
    Post a message to a Slack channel, optionally in a thread.

    Args:
        token:     Slack bot OAuth token.
        channel:   Channel ID or name (e.g. ``#devops-alerts`` or ``C01ABCDEF``).
        text:      Message text (supports mrkdwn).
        thread_ts: If provided, posts as a reply to this thread timestamp.

    Returns:
        Dict with keys ``ts``, ``channel``, ``message``.
    """
    payload: dict = {"channel": channel, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_SLACK_API}/chat.postMessage",
            headers=_headers(token),
            json=payload,
        )
        resp.raise_for_status()
    data = resp.json()
    _check_slack_ok(data)
    log.info("slack_message_posted", channel=channel, ts=data.get("ts"))
    return {"ts": data["ts"], "channel": data["channel"], "message": data.get("message", {})}


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def get_channel_history(token: str, channel: str, limit: int = 20) -> list[dict]:
    """
    Fetch recent messages from a Slack channel.

    Args:
        token:   Slack bot OAuth token.
        channel: Channel ID.
        limit:   Maximum number of messages to return (max 1000).

    Returns:
        List of message dicts with keys ``ts``, ``user``, ``text``, ``type``.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            f"{_SLACK_API}/conversations.history",
            headers=_headers(token),
            params={"channel": channel, "limit": limit},
        )
        resp.raise_for_status()
    data = resp.json()
    _check_slack_ok(data)
    messages = data.get("messages", [])
    return [
        {
            "ts": m.get("ts", ""),
            "user": m.get("user", ""),
            "text": m.get("text", ""),
            "type": m.get("type", ""),
        }
        for m in messages
    ]


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def lookup_user(token: str, user_id: str) -> dict:
    """
    Fetch a Slack user's profile.

    Args:
        token:   Slack bot OAuth token.
        user_id: Slack user ID (e.g. ``U01ABCDEF``).

    Returns:
        Dict with keys ``id``, ``display_name``, ``real_name``, ``email``.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            f"{_SLACK_API}/users.info",
            headers=_headers(token),
            params={"user": user_id},
        )
        resp.raise_for_status()
    data = resp.json()
    _check_slack_ok(data)
    user = data["user"]
    profile = user.get("profile", {})
    return {
        "id": user["id"],
        "display_name": profile.get("display_name", ""),
        "real_name": profile.get("real_name", ""),
        "email": profile.get("email", ""),
    }


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException, RuntimeError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def update_message(token: str, channel: str, ts: str, text: str) -> dict:
    """
    Edit an existing Slack message.

    Args:
        token:   Slack bot OAuth token.
        channel: Channel ID containing the message.
        ts:      Timestamp of the message to update.
        text:    New message text.

    Returns:
        Dict with keys ``ts``, ``channel``, ``text``.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_SLACK_API}/chat.update",
            headers=_headers(token),
            json={"channel": channel, "ts": ts, "text": text},
        )
        resp.raise_for_status()
    data = resp.json()
    _check_slack_ok(data)
    log.info("slack_message_updated", channel=channel, ts=ts)
    return {"ts": data["ts"], "channel": data["channel"], "text": data.get("text", "")}
