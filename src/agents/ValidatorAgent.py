"""
Validator Agent — Checks answer groundedness against retrieved sources.

Guardrails:
  • Groundedness check: verifies answer is supported by context (not just retrieval score)
  • Catches hallucination: low retrieval quality + high LLM confidence
  • Sets grounded flag on AgentResult for downstream use by Gatekeeper
"""

import re
import time
from src.agents.BaseAgent import BaseAgent, AgentTask, AgentResult
from src.utils.llm_provider import get_shared_llm
from src.utils.Logger import get_logger

logger = get_logger(__name__)


class ValidatorAgent(BaseAgent):
    """Validates that generated answers are grounded in retrieved context.

    This catches the critical failure mode where retrieval quality is low
    but the LLM still generates a confident-sounding answer (hallucination).
    """

    VALIDATION_PROMPT = (
        "You are a factual accuracy validator. Given a context and an answer, "
        "determine if the answer is fully supported by the context.\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Respond with ONLY a JSON object (no markdown, no explanation):\n"
        '{{"grounded": <true|false>, "confidence": <0.0-1.0>, '
        '"issues": "<brief explanation if not grounded, empty string if grounded>"}}'
    )

    def __init__(self):
        super().__init__(
            agent_id="validator",
            role="Validate answer groundedness against sources",
            tools=[],  # Uses LLM directly
        )

    def run(self, task: AgentTask) -> AgentResult:
        """Validate an answer against its source context.

        Expects task.previous_results to contain the SynthesizerAgent's result.

        Args:
            task: AgentTask with query and previous synthesis result.

        Returns:
            AgentResult with grounded flag and adjusted confidence.
        """
        self._start_timer()
        trace = []
        logger.info(f"[validator] START query='{task.query[:80]}...'")

        # Get the synthesis result to validate
        synthesis_result = None
        for prev in task.previous_results:
            if prev.answer:  # The synthesizer's result has an answer
                synthesis_result = prev
                break

        if not synthesis_result or not synthesis_result.answer:
            trace.append("No answer to validate")
            logger.warning("[validator] No answer found in previous results to validate")
            return AgentResult(
                answer="",
                confidence=0.0,
                reasoning_trace=trace,
                grounded=None,
            )

        answer = synthesis_result.answer
        context = "\n\n".join(synthesis_result.retrieved_chunks) if synthesis_result.retrieved_chunks else task.context

        if not context:
            trace.append("No context available for validation — marking as ungrounded")
            logger.warning("[validator] No context for validation, marking answer as ungrounded")
            result = AgentResult(
                answer=answer,
                sources=synthesis_result.sources,
                confidence=0.2,
                reasoning_trace=trace,
                grounded=False,
                retrieved_chunks=synthesis_result.retrieved_chunks,
                retrieval_scores=synthesis_result.retrieval_scores,
                metadata=synthesis_result.metadata,
            )
            return result

        llm = get_shared_llm()
        if not llm:
            # Can't validate without LLM — pass through with moderate confidence
            trace.append("No LLM available for validation, passing through")
            logger.warning("[validator] LLM unavailable, skipping groundedness check")
            result = AgentResult(
                answer=answer,
                sources=synthesis_result.sources,
                confidence=synthesis_result.confidence,
                reasoning_trace=trace,
                grounded=None,  # Unknown
                retrieved_chunks=synthesis_result.retrieved_chunks,
                retrieval_scores=synthesis_result.retrieval_scores,
                metadata=synthesis_result.metadata,
            )
            return result

        try:
            # Truncate context for validation prompt
            truncated_context = context[:4000] if len(context) > 4000 else context

            prompt = self.VALIDATION_PROMPT.format(
                context=truncated_context,
                answer=answer,
            )

            response = llm.generate(prompt, max_tokens=200, temperature=0.1)
            trace.append("Groundedness check completed")

            # Parse validation result
            grounded, confidence, issues = self._parse_validation(response.response)

            if grounded is False:
                trace.append(f"UNGROUNDED: {issues}")
                logger.warning(f"[validator] UNGROUNDED: {issues}")
            else:
                elapsed_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0
                trace.append(f"Grounded (confidence: {confidence:.2f})")
                logger.info(f"[validator] DONE in {elapsed_ms:.0f}ms: grounded={grounded}, confidence={confidence:.2f}")

            return AgentResult(
                answer=answer,
                sources=synthesis_result.sources,
                confidence=confidence,
                reasoning_trace=trace,
                grounded=grounded,
                retrieved_chunks=synthesis_result.retrieved_chunks,
                retrieval_scores=synthesis_result.retrieval_scores,
                metadata=synthesis_result.metadata,
            )

        except Exception as e:
            trace.append(f"Validation error: {e}, passing through")
            logger.error(f"[validator] FAILED: {e}", exc_info=True)
            return AgentResult(
                answer=answer,
                sources=synthesis_result.sources,
                confidence=synthesis_result.confidence * 0.8,  # Slight penalty
                reasoning_trace=trace,
                grounded=None,
                retrieved_chunks=synthesis_result.retrieved_chunks,
                retrieval_scores=synthesis_result.retrieval_scores,
                metadata=synthesis_result.metadata,
            )

    def _parse_validation(self, response_text: str):
        """Parse the LLM's validation response.

        Returns:
            Tuple of (grounded: bool, confidence: float, issues: str)
        """
        import json

        if not response_text:
            return None, 0.5, ""

        try:
            # Clean up response
            text = response_text.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            parsed = json.loads(text)
            grounded = parsed.get("grounded", True)
            confidence = float(parsed.get("confidence", 0.5))
            issues = parsed.get("issues", "")

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            return grounded, confidence, issues

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[validator] JSON parse failed: {e}, falling back to keyword check")
            # If parsing fails, do a simple keyword check
            text_lower = response_text.lower()
            if "not grounded" in text_lower or '"grounded": false' in text_lower:
                return False, 0.3, "Parsed as ungrounded from text"
            return True, 0.6, ""
