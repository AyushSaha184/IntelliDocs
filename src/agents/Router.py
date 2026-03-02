"""
Conditional Router — Routes query plans to appropriate execution paths.

Routes based on QueryPlan.query_type:
  • "trivial"         → DIRECT_LLM (fast-path, no retrieval)
  • "factual" (simple)→ SINGLE_AGENT (one retrieval + one generation)
  • "analytical"      → MULTI_AGENT (retrieval + synthesis + validation)
  • "comparative"     → MULTI_AGENT
  • "multi_hop"       → MULTI_AGENT (per-subquery retrieval + aggregation)
  • Low confidence    → HUMAN_REVIEW (avg retrieval score < threshold)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from src.agents.Planner import QueryPlan
from src.utils.Logger import get_logger

logger = get_logger(__name__)


# ── Route decision enum ─────────────────────────────────────────────────

class RouteDecision(Enum):
    """Where to route a query."""
    DIRECT_LLM = "direct_llm"          # Trivial queries — straight to LLM, no retrieval
    SINGLE_AGENT = "single_agent"      # Factual queries — one retrieval + one generation
    MULTI_AGENT = "multi_agent"        # Complex queries — full agent pipeline
    HUMAN_REVIEW = "human_review"      # Low confidence — escalate to human


# ── Router result ───────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """Result of routing decision."""
    decision: RouteDecision
    query_plan: QueryPlan
    reason: str
    adjusted_top_k: Optional[int] = None  # Reduced k for multi-hop fan-out control


# ── ConditionalRouter ───────────────────────────────────────────────────

class ConditionalRouter:
    """Routes QueryPlans to execution strategies based on query complexity.

    Guardrails:
      • Trivial queries bypass agents entirely (< 2s response time)
      • Multi-hop queries reduce top_k to stay within fan-out budget
      • Confidence threshold gates human review escalation
    """

    CONFIDENCE_THRESHOLD = 0.4   # avg retrieval score below this → escalate
    MULTI_HOP_MAX_K = 3          # reduced k per sub-query for multi-hop

    def route(self, plan: QueryPlan) -> RouteResult:
        """Decide where to route the query plan.

        Args:
            plan: QueryPlan from QueryPlanner.

        Returns:
            RouteResult with decision and adjusted parameters.
        """
        # 1. Fast-path: trivial queries
        if plan.query_type == "trivial":
            logger.info(f"Router: DIRECT_LLM (trivial query)")
            return RouteResult(
                decision=RouteDecision.DIRECT_LLM,
                query_plan=plan,
                reason="Trivial query — direct LLM response, no retrieval needed",
            )

        # 2. Simple factual queries — single agent path
        if plan.query_type == "factual" and not plan.needs_multi_hop:
            logger.info(f"Router: SINGLE_AGENT (factual)")
            return RouteResult(
                decision=RouteDecision.SINGLE_AGENT,
                query_plan=plan,
                reason="Factual query — single retrieval + generation",
            )

        # 3. Summarization — single agent (but may need more chunks)
        if plan.query_type == "summarization":
            logger.info(f"Router: SINGLE_AGENT (summarization)")
            return RouteResult(
                decision=RouteDecision.SINGLE_AGENT,
                query_plan=plan,
                reason="Summarization — single retrieval with higher k",
                adjusted_top_k=10,  # More chunks for better summaries
            )

        # 4. Complex queries — multi-agent path
        if plan.query_type in ("analytical", "comparative", "multi_hop") or plan.needs_multi_hop:
            adjusted_k = self.MULTI_HOP_MAX_K if plan.needs_multi_hop else None
            logger.info(
                f"Router: MULTI_AGENT ({plan.query_type})"
                + (f", adjusted k={adjusted_k}" if adjusted_k else "")
            )
            return RouteResult(
                decision=RouteDecision.MULTI_AGENT,
                query_plan=plan,
                reason=f"Complex {plan.query_type} query — multi-agent pipeline",
                adjusted_top_k=adjusted_k,
            )

        # 5. Default: single agent (safe fallback)
        logger.info(f"Router: SINGLE_AGENT (default fallback)")
        return RouteResult(
            decision=RouteDecision.SINGLE_AGENT,
            query_plan=plan,
            reason="Default routing — single agent path",
        )

    def should_escalate_to_human(self, avg_retrieval_score: float) -> bool:
        """Check if retrieval confidence is below threshold.

        This is called AFTER retrieval to decide if the result should
        be routed to human review instead of returned directly.

        Args:
            avg_retrieval_score: Average similarity score from retrieval (0-1).

        Returns:
            True if should escalate to human review.
        """
        if avg_retrieval_score < self.CONFIDENCE_THRESHOLD:
            logger.info(
                f"Router: ESCALATE to human review "
                f"(avg_score={avg_retrieval_score:.3f} < threshold={self.CONFIDENCE_THRESHOLD})"
            )
            return True
        return False
