"""
Centralised configuration for the Nexus Orchestrator service.

All values are loaded from environment variables via pydantic-settings so
that local development (via .env), Docker, and Kubernetes (via Secrets) all
use the exact same code path.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration backed by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="LLM model name")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", description="Embedding model name"
    )

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: str = Field(default="true")
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_project: str = Field(default="nexus-production")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_index_name: str = Field(default="nexus-codebase")

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: str = Field(..., description="GitHub personal access token")
    github_webhook_secret: str = Field(..., description="GitHub webhook HMAC secret")
    github_org: str = Field(..., description="GitHub organisation name")
    github_repo: str = Field(..., description="Default GitHub repository name")

    # ── Slack ─────────────────────────────────────────────────────────────────
    slack_bot_token: str = Field(..., description="Slack bot OAuth token")
    slack_signing_secret: str = Field(..., description="Slack request signing secret")
    slack_default_channel: str = Field(default="#devops-alerts")

    # ── Jira ──────────────────────────────────────────────────────────────────
    jira_base_url: str = Field(..., description="Jira instance base URL")
    jira_email: str = Field(..., description="Jira user email for Basic auth")
    jira_api_token: str = Field(..., description="Jira API token")
    jira_project_key: str = Field(default="ENG")

    # ── Datadog ───────────────────────────────────────────────────────────────
    datadog_api_key: str = Field(..., description="Datadog API key")
    datadog_app_key: str = Field(..., description="Datadog application key")

    # ── Databases ─────────────────────────────────────────────────────────────
    postgres_url: str = Field(
        default="postgresql+asyncpg://nexus:password@postgres:5432/nexus"
    )
    redis_url: str = Field(default="redis://redis:6379/0")

    # ── MCP Server URLs ───────────────────────────────────────────────────────
    mcp_github_url: str = Field(default="http://mcp-github:8001")
    mcp_slack_url: str = Field(default="http://mcp-slack:8002")
    mcp_jira_url: str = Field(default="http://mcp-jira:8003")
    mcp_datadog_url: str = Field(default="http://mcp-datadog:8004")

    # ── Agent Behaviour ───────────────────────────────────────────────────────
    human_in_the_loop: bool = Field(default=False)
    max_agent_iterations: int = Field(default=10)

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    env: Literal["development", "production", "test"] = Field(default="production")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure the log level is one of the standard Python logging levels."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
