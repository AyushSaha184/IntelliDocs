"""
Query Planner - Decomposes user queries into execution plans.

Determines query type, decomposes multi-hop queries into sub-queries,
and enforces retrieval fan-out caps to prevent latency spikes.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from src.utils.Logger import get_logger
from src.utils.llm_provider import get_shared_llm

logger = get_logger(__name__)

MAX_TOTAL_RETRIEVAL_CALLS = 6
MAX_DECOMPOSED_QUERIES = 4

SUPPORTED_QUERY_TYPES = {
    "factual",
    "analytical",
    "comparative",
    "summarization",
    "multi_hop",
    "procedural",
    "ambiguous",
    "follow_up",
    "trivial",
}

TRIVIAL_PATTERNS = [
    re.compile(r"^(hi|hello|hey|greetings|good\\s+(morning|afternoon|evening))[\\s!.]*$", re.IGNORECASE),
    re.compile(r"^(thanks|thank\\s+you|bye|goodbye|cheers)[\\s!.]*$", re.IGNORECASE),
    re.compile(r"^(yes|no|ok|okay|sure|alright)[\\s!.]*$", re.IGNORECASE),
    re.compile(r"^what\\s+is\\s+(a|an)\\s+\\w+\\??$", re.IGNORECASE),
    re.compile(r"^who\\s+are\\s+you\\??$", re.IGNORECASE),
]

FOLLOW_UP_PATTERNS = [
    re.compile(r"^(what about|how about|and what about|in that case|then what|what if)\\b", re.IGNORECASE),
    re.compile(r"^(and|also|so)\\b", re.IGNORECASE),
]


@dataclass
class QueryPlan:
    """Execution plan for a user query."""

    steps: List[str]
    query_type: str
    needs_multi_hop: bool
    decomposed_queries: List[str]
    skip_expansion: bool
    original_query: str = ""
    estimated_retrieval_calls: int = 1


class QueryPlanner:
    """Decomposes queries into structured execution plans."""

    CLASSIFY_PROMPT = (
        "Analyze this query and respond with ONLY a JSON object (no markdown, no explanation):\\n\\n"
        "Query: {query}\\n\\n"
        "Respond with exactly this format:\\n"
        '{{"query_type": "<factual|analytical|comparative|summarization|multi_hop|procedural|ambiguous|follow_up>",'
        ' "needs_multi_hop": <true|false>,'
        ' "sub_queries": ["<sub-query1>", "<sub-query2>"],'
        ' "reasoning": "<one sentence explaining why>"}}\\n\\n'
        "Rules:\\n"
        "- factual: simple fact lookup\\n"
        "- analytical: requires analysis or interpretation\\n"
        "- comparative: compares two or more things\\n"
        "- summarization: asks for summary\\n"
        "- multi_hop: requires combining info from multiple sources\\n"
        "- procedural: step-by-step or process-oriented question\\n"
        "- ambiguous: broad query with missing scope/context\\n"
        "- follow_up: depends on prior turns\\n"
        "- sub_queries: break down only if multi_hop/comparative (max {max_sub})\\n"
    )

    def plan(self, query: str, top_k: int = 5) -> QueryPlan:
        query = query.strip()

        if self._is_trivial(query):
            return QueryPlan(
                steps=["direct_llm_response"],
                query_type="trivial",
                needs_multi_hop=False,
                decomposed_queries=[],
                skip_expansion=True,
                original_query=query,
                estimated_retrieval_calls=0,
            )

        llm = get_shared_llm()
        if llm:
            plan = self._classify_with_llm(query, llm, top_k)
            if plan:
                return plan

        return self._classify_heuristic(query, top_k)

    def _is_trivial(self, query: str) -> bool:
        return any(p.match(query) for p in TRIVIAL_PATTERNS)

    def _normalize_query_type(self, value: str) -> str:
        qtype = (value or "factual").strip().lower()
        if qtype not in SUPPORTED_QUERY_TYPES:
            return "factual"
        return qtype

    def _classify_with_llm(self, query: str, llm, top_k: int) -> Optional[QueryPlan]:
        try:
            import json

            prompt = self.CLASSIFY_PROMPT.format(query=query, max_sub=MAX_DECOMPOSED_QUERIES)
            response = llm.generate(prompt, max_tokens=300, temperature=0.1)
            if not response.response:
                return None

            text = response.response.strip()
            text = re.sub(r"^```(?:json)?\\s*", "", text)
            text = re.sub(r"\\s*```$", "", text)
            parsed = json.loads(text)

            query_type = self._normalize_query_type(parsed.get("query_type", "factual"))
            needs_multi_hop = bool(parsed.get("needs_multi_hop", False))
            sub_queries = parsed.get("sub_queries", []) or []
            if isinstance(sub_queries, str):
                sub_queries = [sub_queries]
            sub_queries = [q.strip() for q in sub_queries if isinstance(q, str) and q.strip()]

            if len(sub_queries) > MAX_DECOMPOSED_QUERIES:
                sub_queries = sub_queries[:MAX_DECOMPOSED_QUERIES]

            if query_type in ("comparative", "multi_hop") and len(sub_queries) > 1:
                needs_multi_hop = True

            skip_expansion = needs_multi_hop or query_type in {"comparative", "follow_up"}
            num_queries = max(1, len(sub_queries)) if sub_queries else 1
            estimated_calls = num_queries * top_k

            if estimated_calls > MAX_TOTAL_RETRIEVAL_CALLS:
                adjusted_k = max(2, MAX_TOTAL_RETRIEVAL_CALLS // num_queries)
                estimated_calls = num_queries * adjusted_k
                logger.info(
                    "Fan-out capped: %s sub-queries x k=%s = %s calls",
                    num_queries,
                    adjusted_k,
                    estimated_calls,
                )

            if query_type == "trivial":
                steps = ["direct_llm_response"]
            elif needs_multi_hop:
                steps = [f"retrieve_for_subquery_{i + 1}: {sq}" for i, sq in enumerate(sub_queries)]
                steps.extend(["synthesize_all_results", "validate_answer"])
            else:
                steps = ["retrieve_context", "generate_answer", "validate_answer"]

            return QueryPlan(
                steps=steps,
                query_type=query_type,
                needs_multi_hop=needs_multi_hop,
                decomposed_queries=sub_queries if sub_queries else [query],
                skip_expansion=skip_expansion,
                original_query=query,
                estimated_retrieval_calls=estimated_calls,
            )
        except Exception as e:
            logger.warning("LLM classification failed, falling back to heuristic: %s", e)
            return None

    def _classify_heuristic(self, query: str, top_k: int) -> QueryPlan:
        query_lower = query.lower().strip()

        if self._is_follow_up(query):
            return QueryPlan(
                steps=["retrieve_context", "generate_answer", "validate_answer"],
                query_type="follow_up",
                needs_multi_hop=False,
                decomposed_queries=[query],
                skip_expansion=True,
                original_query=query,
                estimated_retrieval_calls=top_k,
            )

        procedural_words = ["step by step", "process", "procedure", "how to file", "how do i"]
        if any(w in query_lower for w in procedural_words):
            return QueryPlan(
                steps=["retrieve_context", "generate_answer", "validate_answer"],
                query_type="procedural",
                needs_multi_hop=False,
                decomposed_queries=[query],
                skip_expansion=False,
                original_query=query,
                estimated_retrieval_calls=top_k,
            )

        ambiguous_words = ["my rights", "what are rights", "help me", "what should i do"]
        if len(query_lower.split()) <= 8 and any(w in query_lower for w in ambiguous_words):
            return QueryPlan(
                steps=["retrieve_context", "generate_answer", "validate_answer"],
                query_type="ambiguous",
                needs_multi_hop=False,
                decomposed_queries=[query],
                skip_expansion=False,
                original_query=query,
                estimated_retrieval_calls=top_k,
            )

        comparative_words = ["compare", "versus", "vs", "difference between", "better than", "worse than"]
        if any(w in query_lower for w in comparative_words):
            return QueryPlan(
                steps=["retrieve_context", "generate_answer", "validate_answer"],
                query_type="comparative",
                needs_multi_hop=True,
                decomposed_queries=[query],
                skip_expansion=True,
                original_query=query,
                estimated_retrieval_calls=top_k,
            )

        summarize_words = ["summarize", "summary", "overview", "recap", "tldr"]
        if any(w in query_lower for w in summarize_words):
            return QueryPlan(
                steps=["retrieve_context", "generate_answer"],
                query_type="summarization",
                needs_multi_hop=False,
                decomposed_queries=[query],
                skip_expansion=False,
                original_query=query,
                estimated_retrieval_calls=top_k,
            )

        analytical_words = ["why", "how does", "explain", "analyze", "evaluate", "what causes"]
        if any(w in query_lower for w in analytical_words):
            return QueryPlan(
                steps=["retrieve_context", "generate_answer", "validate_answer"],
                query_type="analytical",
                needs_multi_hop=False,
                decomposed_queries=[query],
                skip_expansion=False,
                original_query=query,
                estimated_retrieval_calls=top_k,
            )

        return QueryPlan(
            steps=["retrieve_context", "generate_answer"],
            query_type="factual",
            needs_multi_hop=False,
            decomposed_queries=[query],
            skip_expansion=False,
            original_query=query,
            estimated_retrieval_calls=top_k,
        )

    def _is_follow_up(self, query: str) -> bool:
        q = query.strip()
        if len(q.split()) <= 7 and any(p.match(q) for p in FOLLOW_UP_PATTERNS):
            return True
        return False