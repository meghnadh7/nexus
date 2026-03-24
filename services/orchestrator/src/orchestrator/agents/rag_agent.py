"""
RAG Agent — retrieval-augmented generation with HyDE.

This agent runs FIRST for every event type to pre-fetch codebase context.
It uses Hypothetical Document Embedding (HyDE):
  1. Ask the LLM to generate a hypothetical answer to the query.
  2. Embed the hypothetical answer (not the raw query) — this aligns the
     embedding space between the query and indexed code.
  3. Query Pinecone with the HyDE vector, returning the top 8 chunks.
  4. For ``slack_mention`` events: additionally synthesize a final answer
     from the chunks and post it back to the Slack thread.
"""

import asyncio
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.base import BaseAgent
from orchestrator.graph.state import NexusState, RAGContext, RetrievedChunk
from rag_indexer.embedder import embed_query
from rag_indexer.store import VectorStore

_HYDE_SYSTEM = """\
You are a senior software engineer. Given a question about a codebase, write
a concise but technically detailed answer that could plausibly appear in the
codebase itself — a docstring, inline comment, or code snippet. This hypothetical
answer will be used for semantic search. Be specific: include function names,
module paths, and implementation details as if you know them.
"""

_SYNTHESIZE_SYSTEM = """\
You are a helpful developer assistant with deep knowledge of the codebase.
Using ONLY the retrieved code chunks provided, answer the developer's question.
Always cite the source file and line reference. If the answer cannot be found
in the provided chunks, say so clearly.
"""


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent using HyDE for improved retrieval.

    For all event types: populates ``state.rag_context`` with the retrieved
    chunks so downstream agents (CodeReview, Incident) have codebase context.

    For ``slack_mention`` events: additionally synthesizes and posts a final
    answer to the Slack thread.
    """

    def __init__(self) -> None:
        """Initialise the agent and the Pinecone vector store."""
        super().__init__()
        self._store = VectorStore(
            api_key=self.settings.pinecone_api_key,
            index_name=self.settings.pinecone_index_name,
        )

    async def _build_query(self, state: NexusState) -> str:
        """
        Derive the retrieval query from the event payload.

        Args:
            state: Current NexusState.

        Returns:
            Query string appropriate for the event type.
        """
        event = state["event"]
        payload = event.payload

        if event.event_type in ("pr_opened", "pr_updated"):
            # Use the PR title and description as the query
            title = payload.get("pull_request", {}).get("title", "code review")
            body = payload.get("pull_request", {}).get("body", "")
            return f"{title}\n{body}"[:500]

        if event.event_type == "alert_fired":
            monitor_name = payload.get("alert_title") or payload.get("title", "system alert")
            return f"runbook for: {monitor_name}"

        if event.event_type == "slack_mention":
            return payload.get("text", "").replace("<@U", "").strip()

        if event.event_type == "deploy_request":
            env = payload.get("environment", "production")
            return f"deployment steps for {env} environment"

        return "general codebase context"

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Perform HyDE retrieval: generate a hypothetical doc, embed it,
        query Pinecone, return the top chunks.

        Args:
            query: The developer's question or event-derived query.

        Returns:
            List of up to 8 RetrievedChunk objects ordered by score descending.
        """
        # Step 1: Generate hypothetical document
        response = await self.llm.ainvoke(
            [
                SystemMessage(content=_HYDE_SYSTEM),
                HumanMessage(content=query),
            ]
        )
        hypothetical_doc = response.content

        self.log.info(
            "hyde_generated",
            query_len=len(query),
            hyde_len=len(str(hypothetical_doc)),
        )

        # Step 2: Embed the hypothetical document
        vector = await embed_query(
            text=str(hypothetical_doc),
            api_key=self.settings.openai_api_key,
        )

        # Step 3: Query Pinecone
        matches = await self._store.query(vector=vector, top_k=8)

        chunks = [
            RetrievedChunk(
                content=m["metadata"].get("content", ""),
                source_file=m["metadata"].get("source_file", ""),
                chunk_index=int(m["metadata"].get("chunk_index", 0)),
                score=float(m["score"]),
                language=m["metadata"].get("language", ""),
            )
            for m in matches
        ]

        self.log.info("retrieval_complete", query=query[:80], chunks_found=len(chunks))
        return chunks

    async def _synthesize_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """
        Synthesize a final answer from retrieved chunks using GPT-4o.

        Args:
            query:  The original developer question.
            chunks: Retrieved code chunks providing context.

        Returns:
            A synthesized answer string grounded in the retrieved context.
        """
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"**File:** `{chunk.source_file}` (score: {chunk.score:.3f})\n"
                f"```\n{chunk.content[:800]}\n```"
            )
        context_text = "\n\n---\n\n".join(context_parts)

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=_SYNTHESIZE_SYSTEM),
                HumanMessage(
                    content=(
                        f"Question: {query}\n\n"
                        f"Retrieved context:\n{context_text}"
                    )
                ),
            ]
        )
        return str(response.content)

    async def run(self, state: NexusState) -> dict:
        """
        Execute retrieval (and optionally synthesis + Slack reply).

        Args:
            state: Current NexusState.

        Returns:
            State update dict with ``rag_context`` and optionally
            ``tool_calls`` populated.
        """
        event = state["event"]
        query = await self._build_query(state)

        chunks = await self.retrieve(query)

        synthesized_answer = None

        if event.event_type == "slack_mention":
            # Synthesize answer and post to Slack thread
            synthesized_answer = await self._synthesize_answer(query, chunks)

            thread_ts = event.payload.get("thread_ts") or event.payload.get("ts")
            channel = event.payload.get("channel", self.settings.slack_default_channel)

            await self.mcp.call(
                "slack",
                "post_message",
                {
                    "channel": channel,
                    "text": synthesized_answer,
                    "thread_ts": thread_ts,
                },
            )
            self.log.info("slack_answer_posted", channel=channel)

        rag_context = RAGContext(
            query=query,
            retrieved_chunks=chunks,
            synthesized_answer=synthesized_answer,
        )

        return {
            "rag_context": rag_context,
            "tool_calls": self._drain_tool_logs(),
            "current_agent": "rag",
        }
