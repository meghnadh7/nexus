"""
AsyncPostgresSaver checkpointer factory for the Nexus orchestrator.

LangGraph's AsyncPostgresSaver persists every graph step to Postgres so that:
  - Interrupted runs (pod restarts, SIGTERM) can be resumed from the last
    checkpoint.
  - Human-in-the-loop approval flows can pause indefinitely and resume when
    the human responds.
  - Multiple orchestrator replicas can safely share state.

Uses a psycopg_pool.AsyncConnectionPool for production-grade connection
management — this keeps connections open across requests and avoids the
overhead of reconnecting for every graph run.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from orchestrator.config import get_settings

log = structlog.get_logger(__name__)


async def create_checkpointer() -> tuple[AsyncPostgresSaver, AsyncConnectionPool]:
    """
    Create an initialised AsyncPostgresSaver backed by a connection pool.

    Returns both the checkpointer and the pool so the caller can close
    the pool on shutdown.

    Steps:
      1. Open an ``AsyncConnectionPool`` to Postgres.
      2. Instantiate ``AsyncPostgresSaver(pool)``.
      3. Call ``setup()`` to create LangGraph's checkpoint tables (idempotent).

    Returns:
        Tuple of (checkpointer, pool).

    Raises:
        psycopg.OperationalError: If Postgres is unreachable.
    """
    settings = get_settings()

    # AsyncConnectionPool expects a plain postgresql:// URL (not sqlalchemy+asyncpg)
    conn_string = settings.postgres_url.replace(
        "postgresql+asyncpg://", "postgresql://"
    )

    log.info("checkpointer_pool_opening", url_prefix=conn_string[:40] + "...")

    pool = AsyncConnectionPool(
        conninfo=conn_string,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,  # We will open explicitly below
    )
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    log.info("checkpointer_ready")
    return checkpointer, pool


@asynccontextmanager
async def checkpointer_lifespan() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """
    Async context manager that creates the checkpointer, yields it, and
    closes the underlying connection pool on exit.

    Usage in FastAPI lifespan::

        async with checkpointer_lifespan() as cp:
            compiled_graph = build_graph(checkpointer=cp)
            yield {"graph": compiled_graph}

    Yields:
        Initialised AsyncPostgresSaver.
    """
    checkpointer, pool = await create_checkpointer()
    try:
        yield checkpointer
    finally:
        log.info("checkpointer_pool_closing")
        await pool.close()
        log.info("checkpointer_pool_closed")
