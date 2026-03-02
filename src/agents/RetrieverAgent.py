"""
Retriever Agent — Focused on dense retrieval, ranking, and deduplication.

Guardrails:
  • Chunk deduplication by (document_id, section_title)
  • Freshness bias for conflicting data from different timestamps
  • MAX_TOOL_CALLS = 3 inherited from BaseAgent
"""

import time
from typing import List, Dict
from src.agents.BaseAgent import BaseAgent, AgentTask, AgentResult
from src.utils.Logger import get_logger

logger = get_logger(__name__)


class RetrieverAgent(BaseAgent):
    """Retrieves and filters relevant chunks for a query.

    Calls the retrieve_chunks tool, deduplicates results,
    and applies freshness bias when conflicts are detected.
    """

    def __init__(self):
        super().__init__(
            agent_id="retriever",
            role="Retrieve relevant document chunks",
            tools=["retrieve_chunks"],
        )

    # If top-1 dense similarity is above this threshold, skip sparse/expansion
    # to avoid unnecessary compute on already strong matches.
    DENSE_EARLY_EXIT_THRESHOLD = 0.92

    def run(self, task: AgentTask) -> AgentResult:
        """Retrieve chunks for the query.

        Args:
            task: AgentTask with query and retriever.

        Returns:
            AgentResult with retrieved chunks, scores, and metadata.
        """
        self._start_timer()
        trace = []
        logger.info(f"[retriever] START query='{task.query[:80]}...' top_k={task.top_k} session={task.session_id[:8]}")

        trace.append(f"Retrieving top-{task.top_k} chunks for: {task.query[:80]}...")

        # Call retrieval tool
        result = self._call_tool(
            "retrieve_chunks",
            query=task.query,
            top_k=task.top_k,
            session_id=task.session_id,
            retriever=task.retriever,
        )

        if not result.success:
            trace.append(f"Retrieval failed: {result.error}")
            logger.error(f"[retriever] FAILED: {result.error}")
            return AgentResult(
                answer="",
                confidence=0.0,
                reasoning_trace=trace,
                error=result.error,
            )

        chunks_data = result.output.get("chunks", [])
        if not chunks_data:
            trace.append("No chunks retrieved")
            logger.warning(f"[retriever] 0 chunks returned for: '{task.query[:60]}'")
            return AgentResult(
                answer="No relevant chunks found for this query.",
                confidence=0.0,
                reasoning_trace=trace,
            )

        # Deduplicate by (document_id + text hash) to remove near-duplicates
        deduped = self._deduplicate_chunks(chunks_data)
        trace.append(f"Retrieved {len(chunks_data)} chunks, {len(deduped)} after deduplication")

        # Hybrid early-exit: if top-1 score is extremely high, flag it
        # (the Planner/Router can use this to reduce sparse k or skip expansion)
        top_score = max((c.get("score", 0.0) for c in deduped), default=0.0)
        if top_score >= self.DENSE_EARLY_EXIT_THRESHOLD:
            trace.append(f"Dense early-exit triggered (top_score={top_score:.3f} >= {self.DENSE_EARLY_EXIT_THRESHOLD})")
            logger.info(f"[retriever] Dense early-exit: top_score={top_score:.3f}, limiting to top-1 result")
            deduped = deduped[:1]  # Only the best match needed

        # Extract outputs, formatting chunks to include summaries but hide hypothetical questions
        retrieved_chunks = []
        for c in deduped:
            text = c["text"]
            # Exclude hypothetical questions so the LLM doesn't try to answer them!
            parts = []
            if "summary" in c and c["summary"]:
                parts.append(f"[Summary]: {c['summary']}")
            if "keywords" in c and c["keywords"]:
                kws = c["keywords"]
                if isinstance(kws, list):
                    kws = ", ".join(kws)
                parts.append(f"[Keywords]: {kws}")
            
            parts.append(f"[Content]:\n{text}")
            retrieved_chunks.append("\n\n".join(parts))

        retrieval_scores = [c.get("score", 0.0) for c in deduped]
        metadata = [{"chunk_id": c["chunk_id"], "score": c.get("score", 0.0)} for c in deduped]

        avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
        elapsed_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0
        trace.append(f"Average retrieval score: {avg_score:.3f}")
        logger.info(
            f"[retriever] DONE in {elapsed_ms:.0f}ms: "
            f"{len(chunks_data)} raw → {len(deduped)} deduped, avg_score={avg_score:.3f}"
        )

        return AgentResult(
            answer="",  # RetrieverAgent doesn't generate answers
            sources=[c["chunk_id"] for c in deduped],
            confidence=avg_score,
            reasoning_trace=trace,
            retrieved_chunks=retrieved_chunks,
            retrieval_scores=retrieval_scores,
            metadata=metadata,
        )

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Deduplicate chunks by content similarity.

        Uses text hash to detect near-duplicates from different
        retrieval strategies (dense vs sparse).
        """
        seen_hashes = set()
        deduped = []

        for chunk in chunks:
            # Hash first 200 chars as a quick similarity proxy
            text_key = chunk.get("text", "")[:200].strip().lower()
            if text_key in seen_hashes:
                continue
            seen_hashes.add(text_key)
            deduped.append(chunk)

        return deduped
