"""
Conditional Router - Routes query plans to execution paths with strategy hints.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.agents.Planner import QueryPlan
from src.utils.Logger import get_logger

logger = get_logger(__name__)


class RouteDecision(Enum):
    DIRECT_LLM = "direct_llm"
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    HUMAN_REVIEW = "human_review"


@dataclass
class RouteResult:
    decision: RouteDecision
    query_plan: QueryPlan
    reason: str
    adjusted_top_k: Optional[int] = None
    retrieval_options: Dict[str, Any] = field(default_factory=dict)
    synthesis_style: Optional[str] = None


class ConditionalRouter:
    """Routes QueryPlans to execution strategies based on query complexity."""

    CONFIDENCE_THRESHOLD = 0.4
    MULTI_HOP_MAX_K = 4

    def route(self, plan: QueryPlan) -> RouteResult:
        if plan.query_type == "trivial":
            return RouteResult(
                decision=RouteDecision.DIRECT_LLM,
                query_plan=plan,
                reason="Trivial query - direct LLM response",
            )

        if plan.query_type == "factual" and not plan.needs_multi_hop:
            return RouteResult(
                decision=RouteDecision.SINGLE_AGENT,
                query_plan=plan,
                reason="Factual query - low-latency single-agent retrieval",
                retrieval_options={"candidate_pool_multiplier": 2},
            )

        if plan.query_type == "follow_up":
            return RouteResult(
                decision=RouteDecision.SINGLE_AGENT,
                query_plan=plan,
                reason="Follow-up query - history-expanded retrieval",
                adjusted_top_k=6,
                retrieval_options={
                    "force_rerank": True,
                    "candidate_pool_multiplier": 3,
                },
            )

        if plan.query_type == "summarization":
            return RouteResult(
                decision=RouteDecision.SINGLE_AGENT,
                query_plan=plan,
                reason="Summarization - retrieve broader context",
                adjusted_top_k=10,
                retrieval_options={"candidate_pool_multiplier": 3},
            )

        if plan.query_type == "procedural":
            return RouteResult(
                decision=RouteDecision.SINGLE_AGENT,
                query_plan=plan,
                reason="Procedural query - higher recall and reranking",
                adjusted_top_k=8,
                retrieval_options={
                    "force_rerank": True,
                    "candidate_pool_multiplier": 4,
                    "rerank_top_k": 8,
                },
            )

        if plan.query_type == "ambiguous":
            return RouteResult(
                decision=RouteDecision.SINGLE_AGENT,
                query_plan=plan,
                reason="Ambiguous query - broad retrieval with structured response",
                adjusted_top_k=12,
                retrieval_options={
                    "force_rerank": True,
                    "candidate_pool_multiplier": 4,
                    "rerank_top_k": 12,
                },
                synthesis_style="structured_scenarios",
            )

        if plan.query_type in ("analytical", "comparative", "multi_hop") or plan.needs_multi_hop:
            adjusted_k = self.MULTI_HOP_MAX_K if plan.needs_multi_hop else 6
            return RouteResult(
                decision=RouteDecision.MULTI_AGENT,
                query_plan=plan,
                reason=f"Complex {plan.query_type} query - multi-agent pipeline",
                adjusted_top_k=adjusted_k,
                retrieval_options={
                    "force_rerank": True,
                    "candidate_pool_multiplier": 4,
                },
            )

        return RouteResult(
            decision=RouteDecision.SINGLE_AGENT,
            query_plan=plan,
            reason="Default routing - single-agent path",
            retrieval_options={"candidate_pool_multiplier": 2},
        )

    def should_escalate_to_human(self, avg_retrieval_score: float) -> bool:
        if avg_retrieval_score < self.CONFIDENCE_THRESHOLD:
            logger.info(
                "Router: ESCALATE to human review (avg_score=%.3f < threshold=%.3f)",
                avg_retrieval_score,
                self.CONFIDENCE_THRESHOLD,
            )
            return True
        return False