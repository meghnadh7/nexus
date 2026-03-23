"""
Batch embedding pipeline for the Nexus RAG indexer.

Uses the OpenAI async client to embed chunks in batches of 100,
then returns records formatted for Pinecone upsert.
"""

import asyncio
import hashlib
import os
from typing import Any

import openai
import structlog

from rag_indexer.chunker import Chunk

log = structlog.get_logger(__name__)

_BATCH_SIZE = 100
_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def _chunk_id(chunk: Chunk) -> str:
    """
    Generate a deterministic, stable vector ID for a chunk.

    Uses SHA-256 of ``source_file:chunk_index`` so re-indexing the same
    file produces identical IDs (enabling Pinecone upsert idempotency).
    """
    raw = f"{chunk.source_file}:{chunk.chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:48]


async def embed_chunks(
    chunks: list[Chunk],
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Embed a list of Chunk objects in batches and return Pinecone-ready records.

    Args:
        chunks:  List of Chunk objects to embed.
        api_key: OpenAI API key.

    Returns:
        List of dicts, each with keys ``id``, ``values`` (embedding vector),
        and ``metadata`` (content preview, source_file, chunk_index, language).
    """
    client = openai.AsyncOpenAI(api_key=api_key)
    records: list[dict[str, Any]] = []

    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _BATCH_SIZE]
        texts = [c.content for c in batch]

        log.info(
            "embedding_batch",
            batch_start=batch_start,
            batch_size=len(batch),
            total=len(chunks),
        )

        response = await client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=texts,
        )

        for chunk, embedding_obj in zip(batch, response.data):
            records.append(
                {
                    "id": _chunk_id(chunk),
                    "values": embedding_obj.embedding,
                    "metadata": {
                        # Store a trimmed preview to keep Pinecone metadata small
                        "content": chunk.content[:1000],
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "language": chunk.language,
                    },
                }
            )

    log.info("embedding_complete", total_records=len(records))
    return records


async def embed_query(text: str, api_key: str) -> list[float]:
    """
    Embed a single query string for retrieval.

    Args:
        text:    Query text (or HyDE hypothetical answer).
        api_key: OpenAI API key.

    Returns:
        Embedding vector as a list of floats.
    """
    client = openai.AsyncOpenAI(api_key=api_key)
    response = await client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=[text],
    )
    return response.data[0].embedding
