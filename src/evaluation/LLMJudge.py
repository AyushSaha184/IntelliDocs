"""
LLM-as-Judge — Automated evaluation of RAG answer quality.

Runs as a background task after each answer (zero latency impact).
Scores along three dimensions:
  • Relevance    — Is the answer relevant to the query?
  • Faithfulness — Is the answer grounded in the retrieved context?
  • Completeness — Does the answer fully address the query?
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, List
from src.utils.llm_provider import get_judge_llm
from src.utils.Logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalScore:
    """Evaluation scores for a single query-answer pair."""
    relevance: float       # 0.0 – 1.0
    faithfulness: float    # 0.0 – 1.0 (groundedness)
    completeness: float    # 0.0 – 1.0
    overall: float         # Weighted average
    reasoning: str = ""


class LLMJudge:
    """Evaluates RAG answer quality using the shared LLM.

    Designed to run as a background task — never blocks the user response.
    """

    EVAL_PROMPT = (
        "You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system. "
        "Evaluate the quality of the following answer.\n\n"
        "Query: {query}\n\n"
        "Retrieved Context:\n{context}\n\n"
        "Generated Answer:\n{answer}\n\n"
        "Score each dimension from 0.0 to 1.0. Respond with ONLY a JSON object:\n"
        '{{"relevance": <0.0-1.0>, "faithfulness": <0.0-1.0>, '
        '"completeness": <0.0-1.0>, '
        '"reasoning": "<brief explanation of scores>"}}\n\n'
        "Scoring guide:\n"
        "- relevance: Does the answer address the query? (1.0 = perfectly relevant)\n"
        "- faithfulness: Is the answer supported by the context? (1.0 = fully grounded, no hallucination)\n"
        "- completeness: Does the answer fully address all aspects of the query? (1.0 = comprehensive)\n"
    )

    WEIGHTS = {
        "relevance": 0.3,
        "faithfulness": 0.5,  # Highest weight — groundedness is critical
        "completeness": 0.2,
    }

    def score(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[str],
    ) -> Optional[EvalScore]:
        """Score a query-answer pair.

        Args:
            query: Original user query.
            answer: Generated answer.
            retrieved_chunks: Retrieved context chunks.

        Returns:
            EvalScore, or None if evaluation fails.
        """
        llm = get_judge_llm()
        if not llm:
            logger.warning("LLMJudge: No LLM available, skipping evaluation")
            return None

        context = "\n\n---\n\n".join(retrieved_chunks[:10])  # Limit context
        if len(context) > 4000:
            context = context[:4000] + "\n\n[...truncated for evaluation...]"

        prompt = self.EVAL_PROMPT.format(
            query=query,
            context=context,
            answer=answer,
        )

        try:
            response = llm.generate(prompt, max_tokens=200, temperature=0.1)

            if not response.response:
                return None

            return self._parse_scores(response.response)

        except Exception as e:
            logger.warning(f"LLMJudge evaluation failed: {e}")
            return None

    def _parse_scores(self, response_text: str) -> Optional[EvalScore]:
        """Parse LLM evaluation response into EvalScore."""
        try:
            text = response_text.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            parsed = json.loads(text)

            relevance = max(0.0, min(1.0, float(parsed.get("relevance", 0.5))))
            faithfulness = max(0.0, min(1.0, float(parsed.get("faithfulness", 0.5))))
            completeness = max(0.0, min(1.0, float(parsed.get("completeness", 0.5))))
            reasoning = parsed.get("reasoning", "")

            overall = (
                relevance * self.WEIGHTS["relevance"]
                + faithfulness * self.WEIGHTS["faithfulness"]
                + completeness * self.WEIGHTS["completeness"]
            )

            return EvalScore(
                relevance=relevance,
                faithfulness=faithfulness,
                completeness=completeness,
                overall=overall,
                reasoning=reasoning,
            )

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"LLMJudge: Failed to parse scores: {e}")
            return None
