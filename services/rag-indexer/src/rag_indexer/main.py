"""
RAG Indexer — main entry point.

Walks a repository recursively, chunks each eligible source file,
embeds the chunks in batches, and upserts them to Pinecone.

Intended to run as a Kubernetes CronJob (every hour) and as a GitHub
Actions step on every push to main.

Usage:
    python -m rag_indexer.main --repo-path /path/to/repo
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import structlog

from rag_indexer.chunker import Chunk, chunk_file
from rag_indexer.embedder import embed_chunks
from rag_indexer.store import VectorStore

# ─── Logging setup ────────────────────────────────────────────────────────────

_env = os.getenv("ENV", "production")

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if _env == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

_SKIP_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "site-packages",
    ".tox",
    "htmlcov",
}

_ELIGIBLE_EXTENSIONS = {
    ".py", ".ts", ".js", ".go", ".java",
    ".md", ".yaml", ".yml", ".json",
}

_MAX_FILE_SIZE_BYTES = 500_000  # Skip files larger than 500 KB


# ─── File collection ──────────────────────────────────────────────────────────


def collect_files(repo_path: Path) -> list[Path]:
    """
    Walk ``repo_path`` recursively and return all eligible source files.

    Skips directories listed in ``_SKIP_DIRS`` and files larger than
    ``_MAX_FILE_SIZE_BYTES``.

    Args:
        repo_path: Root directory of the repository.

    Returns:
        Sorted list of eligible file paths.
    """
    files: list[Path] = []
    for path in repo_path.rglob("*"):
        # Skip hidden and blacklisted directories
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in _ELIGIBLE_EXTENSIONS:
            try:
                if path.stat().st_size <= _MAX_FILE_SIZE_BYTES:
                    files.append(path)
            except OSError:
                pass
    return sorted(files)


# ─── Main pipeline ────────────────────────────────────────────────────────────


async def index_repository(repo_path: Path) -> None:
    """
    Run the full indexing pipeline for a repository.

    Steps:
    1. Collect eligible files.
    2. Chunk each file with language-aware chunker.
    3. Embed all chunks with OpenAI in batches of 100.
    4. Upsert all vectors to Pinecone in batches of 100.

    Args:
        repo_path: Root directory of the repository to index.
    """
    openai_api_key = os.environ["OPENAI_API_KEY"]
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pinecone_index = os.environ.get("PINECONE_INDEX_NAME", "nexus-codebase")

    log.info("indexer_started", repo_path=str(repo_path))

    # 1. Collect files
    files = collect_files(repo_path)
    log.info("files_collected", count=len(files))

    # 2. Chunk all files
    all_chunks: list[Chunk] = []
    for file_idx, file_path in enumerate(files):
        if file_idx % 100 == 0 and file_idx > 0:
            log.info("chunking_progress", files_processed=file_idx, total=len(files))

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_file(file_path, text)
            all_chunks.extend(chunks)
        except OSError as exc:
            log.warning("file_read_error", path=str(file_path), error=str(exc))

    log.info("chunking_complete", total_chunks=len(all_chunks))

    if not all_chunks:
        log.warning("no_chunks_produced")
        return

    # 3. Embed chunks
    records = await embed_chunks(all_chunks, api_key=openai_api_key)

    # 4. Upsert to Pinecone
    store = VectorStore(api_key=pinecone_api_key, index_name=pinecone_index)
    upserted = await store.upsert(records)

    log.info(
        "indexer_complete",
        total_files=len(files),
        total_chunks=len(all_chunks),
        upserted=upserted,
    )


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point — parse args and run the async pipeline."""
    parser = argparse.ArgumentParser(description="Nexus RAG Indexer")
    parser.add_argument(
        "--repo-path",
        default=os.getenv("REPO_PATH", "."),
        help="Path to the repository root to index (default: current directory)",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    if not repo_path.is_dir():
        log.error("repo_path_not_found", path=str(repo_path))
        sys.exit(1)

    asyncio.run(index_repository(repo_path))


if __name__ == "__main__":
    main()
