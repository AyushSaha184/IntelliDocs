"""
Synthesizer Agent — Combines retrieved chunks into a coherent answer.

Guardrails:
  • Conflict surfacing: if sources contradict, states this explicitly
  • Uses shared LLM singleton (no separate instance)
"""

import re
import time
from src.agents.BaseAgent import BaseAgent, AgentTask, AgentResult
from src.utils.llm_provider import get_shared_llm
from src.utils.Logger import get_logger

logger = get_logger(__name__)


class SynthesizerAgent(BaseAgent):
    """Takes retrieval results and generates a coherent answer using LLM.

    The synthesis prompt explicitly instructs the model to surface
    contradictions rather than blending conflicting information.
    """

    SYSTEM_PROMPT = (
        "You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. "
        "If the context doesn't contain relevant information, say so honestly. "
        "Be concise and accurate. Do NOT cite sources inline or use bracketed citations like [Context 1]. "
        "The system will automatically attach the source documents to your response.\n\n"
        "IMPORTANT ENTITY RULE: If the user asks about a specific entity (e.g., a company, product, or person) "
        "and that entity is NOT explicitly named or clearly described in the context, you MUST state that you do not have information about it. "
        "DO NOT assume generic text like 'our company' or 'the business' refers to the specific entity the user asked about.\n\n"
        "IMPORTANT CONFLICT RULE: If retrieved sources contain contradictory information "
        "(e.g., different dates, different policy versions, conflicting data), "
        "you MUST state the contradiction clearly to the user. "
        "Do NOT guess or blend conflicting facts."
    )

    def __init__(self):
        super().__init__(
            agent_id="synthesizer",
            role="Synthesize retrieved context into a coherent answer",
            tools=[],  # SynthesizerAgent uses LLM directly, no tools needed
        )

    def run(self, task: AgentTask) -> AgentResult:
        """Generate an answer from retrieved context.

        Args:
            task: AgentTask with query and previous retrieval results.

        Returns:
            AgentResult with synthesized answer.
        """
        self._start_timer()
        trace = []
        logger.info(f"[synthesizer] START query='{task.query[:80]}...' prev_results={len(task.previous_results)}")

        llm = get_shared_llm()
        if not llm:
            trace.append("No LLM available for synthesis")
            logger.error("[synthesizer] FAILED: LLM singleton is None")
            return AgentResult(
                answer="Unable to generate a response — LLM is not available.",
                confidence=0.0,
                reasoning_trace=trace,
                error="LLM not initialized",
            )

        # Collect retrieved chunks from previous results
        all_chunks = []
        all_sources = []
        for prev in task.previous_results:
            all_chunks.extend(prev.retrieved_chunks)
            all_sources.extend(prev.sources)

        # Also use context if provided directly
        if task.context:
            all_chunks.insert(0, task.context)

        if not all_chunks:
            trace.append("No context available for synthesis")
            logger.warning("[synthesizer] No context chunks available, cannot synthesize")
            return AgentResult(
                answer="I don't have enough context to answer this question.",
                confidence=0.0,
                reasoning_trace=trace,
            )

        # Format context
        context_parts = []
        for i, chunk in enumerate(all_chunks, 1):
            context_parts.append(f"[Context {i}]\n{chunk}")
        context = "\n\n".join(context_parts)

        trace.append(f"Synthesizing from {len(all_chunks)} context chunks")

        # Create prompt and generate
        prompt = llm.create_rag_prompt(
            query=task.query,
            context=context,
            system_prompt=self.SYSTEM_PROMPT,
        )

        try:
            response = llm.generate(prompt, temperature=0.3)

            # Strip reasoning blocks from logic-heavy models
            answer = re.sub(r'<think>.*?</think>', '', response.response, flags=re.DOTALL).strip()

            trace.append(f"Generated answer ({response.total_tokens} tokens)")
            elapsed_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0
            logger.info(
                f"[synthesizer] DONE in {elapsed_ms:.0f}ms: "
                f"{len(all_chunks)} chunks, {response.total_tokens} tokens, answer_len={len(answer)}"
            )

            return AgentResult(
                answer=answer,
                sources=all_sources,
                confidence=0.8,  # Will be adjusted by ValidatorAgent
                reasoning_trace=trace,
                retrieved_chunks=all_chunks,
            )

        except Exception as e:
            trace.append(f"Synthesis failed: {e}")
            logger.error(f"[synthesizer] FAILED: {e}", exc_info=True)
            return AgentResult(
                answer="",
                confidence=0.0,
                reasoning_trace=trace,
                error=str(e),
            )
