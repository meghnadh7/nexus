"""
Pinecone vector store interface for the Nexus RAG indexer.

The official Pinecone Python client is synchronous, so we wrap all calls
in ``asyncio.get_event_loop().run_in_executor(None, ...)`` to keep the
indexer pipeline non-blocking.
"""

import asyncio
from typing import Any

import structlog
from pinecone import Pinecone, ServerlessSpec

log = structlog.get_logger(__name__)

_BATCH_SIZE = 100


class VectorStore:
    """Thin async wrapper around the synchronous Pinecone client."""

    def __init__(self, api_key: str, index_name: str) -> None:
        """
        Initialise the Pinecone client and connect to (or create) the index.

        Args:
            api_key:    Pinecone API key.
            index_name: Name of the Pinecone index to upsert into.
        """
        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._index = None  # Lazily connected

    def _get_index(self):  # type: ignore[no-untyped-def]
        """Return a connected Pinecone index handle, creating the index if needed."""
        if self._index is not None:
            return self._index

        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing:
            log.info("pinecone_creating_index", index=self._index_name)
            self._pc.create_index(
                name=self._index_name,
                dimension=3072,  # text-embedding-3-large output dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self._pc.Index(self._index_name)
        return self._index

    async def upsert(self, records: list[dict[str, Any]]) -> int:
        """
        Upsert embedding records to Pinecone in batches of 100.

        Args:
            records: List of dicts with keys ``id``, ``values``, ``metadata``.

        Returns:
            Total number of vectors upserted.
        """
        loop = asyncio.get_event_loop()
        total = 0
        index = await loop.run_in_executor(None, self._get_index)

        for batch_start in range(0, len(records), _BATCH_SIZE):
            batch = records[batch_start : batch_start + _BATCH_SIZE]
            await loop.run_in_executor(None, lambda b=batch: index.upsert(vectors=b))
            total += len(batch)
            log.info(
                "pinecone_upserted",
                batch_start=batch_start,
                batch_size=len(batch),
                total_so_far=total,
            )

        return total

    async def query(
        self,
        vector: list[float],
        top_k: int = 8,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        """
        Query Pinecone for the top-k nearest neighbours.

        Args:
            vector:    Query embedding vector.
            top_k:     Number of results to return.
            namespace: Optional Pinecone namespace.

        Returns:
            List of dicts with keys ``id``, ``score``, and ``metadata``.
        """
        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(None, self._get_index)

        def _do_query() -> Any:
            return index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace or "",
            )

        result = await loop.run_in_executor(None, _do_query)
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {}),
            }
            for match in result.get("matches", [])
        ]
