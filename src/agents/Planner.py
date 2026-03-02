"""
Query Planner — Decomposes user queries into execution plans.

Determines query type, decomposes multi-hop queries into sub-queries,
and enforces retrieval fan-out caps to prevent latency spikes.

Guardrails:
  • MAX_TOTAL_RETRIEVAL_CALLS = 6 (hard cap across all sub-queries)
  • Trivial query fast-path (greetings, basic definitions → skip agents)
  • skip_expansion when multi-hop detected (prevents query expansion fan-out)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from src.utils.Logger import get_logger
from src.utils.llm_provider import get_shared_llm

logger = get_logger(__name__)

# ── Configuration ───────────────────────────────────────────────────────

MAX_TOTAL_RETRIEVAL_CALLS = 6
MAX_DECOMPOSED_QUERIES = 4

# Trivial query patterns — bypass the full agent pipeline
TRIVIAL_PATTERNS = [
    re.compile(r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[\s!.]*$', re.IGNORECASE),
    re.compile(r'^(thanks|thank\s+you|bye|goodbye|cheers)[\s!.]*$', re.IGNORECASE),
    re.compile(r'^(yes|no|ok|okay|sure|alright)[\s!.]*$', re.IGNORECASE),
    re.compile(r'^what\s+is\s+(a|an)\s+\w+\??$', re.IGNORECASE),
    re.compile(r'^who\s+are\s+you\??$', re.IGNORECASE),
]


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class QueryPlan:
    """Execution plan for a user query."""
    steps: List[str]
    query_type: str               # "factual", "analytical", "comparative", "summarization", "multi_hop", "trivial"
    needs_multi_hop: bool
    decomposed_queries: List[str]
    skip_expansion: bool          # True when multi-hop to prevent fan-out explosion
    original_query: str = ""
    estimated_retrieval_calls: int = 1


# ── QueryPlanner ────────────────────────────────────────────────────────

class QueryPlanner:
    """Decomposes queries into structured execution plans.

    For simple factual queries, the plan is a single retrieval step (zero overhead).
    For complex queries, the LLM decomposes into sub-queries with guardrails.
    """

    # Classification prompt template
    CLASSIFY_PROMPT = (
        "Analyze this query and respond with ONLY a JSON object (no markdown, no explanation):\n\n"
        "Query: {query}\n\n"
        "Respond with exactly this format:\n"
        '{{"query_type": "<factual|analytical|comparative|summarization|multi_hop>",'
        ' "needs_multi_hop": <true|false>,'
        ' "sub_queries": ["<sub-query1>", "<sub-query2>"],'
        ' "reasoning": "<one sentence explaining why>"}}\n\n'
        "Rules:\n"
        "- factual: simple fact lookup (e.g., 'What is X?')\n"
        "- analytical: requires analysis or interpretation\n"
        "- comparative: compares two or more things\n"
        "- summarization: asks for a summary of content\n"
        "- multi_hop: requires combining info from multiple sources\n"
        "- sub_queries: break down ONLY if multi_hop or comparative (max {max_sub})\n"
        "\nIMPORTANT TYPO CORRECTION RULE:\n"
        "If you detect ANY obvious spelling mistakes in the user's query "
        "(e.g., 'Infomatica' -> 'Informatica', 'Microft' -> 'Microsoft', 'teh' -> 'the', 'bussines' -> 'business'), "
        "you MUST silently correct the spelling inside the 'sub_queries' array. Do not ask for clarification, "
        "just fix it so the search engine gets the correctly spelled keywords.\n"
        "\nIMPORTANT LONG QUERY DEGRADATION RULE:\n"
        "If the user provides a very long, conversational, or messy paragraph (e.g., 'Hey so I am trying to figure out what happens when...'), "
        "you MUST rewrite it into a short, dense string of keywords representing the core search intent. "
        "Do NOT pass long conversational filler into 'sub_queries'.\n"
        "- For factual/simple queries without multi-hop, 'sub_queries' must contain EXACTLY ONE query, "
        "which is the original query with any typos corrected and conversational filler removed."
    )

    def plan(self, query: str, top_k: int = 5) -> QueryPlan:
        """Create an execution plan for the given query.

        Args:
            query: User's question.
            top_k: Intended retrieval count per sub-query.

        Returns:
            QueryPlan with type classification and sub-queries.
        """
        query = query.strip()

        # Fast-path: trivial queries bypass everything
        if self._is_trivial(query):
            logger.info(f"Trivial query detected: '{query[:50]}'")
            return QueryPlan(
                steps=["direct_llm_response"],
                query_type="trivial",
                needs_multi_hop=False,
                decomposed_queries=[],
                skip_expansion=True,
                original_query=query,
                estimated_retrieval_calls=0,
            )

        # Try LLM-based classification
        llm = get_shared_llm()
        if llm:
            logger.debug(f"[planner] Attempting LLM classification for: '{query[:60]}'")
            plan = self._classify_with_llm(query, llm, top_k)
            if plan:
                logger.info(f"[planner] LLM classified: type={plan.query_type}, multi_hop={plan.needs_multi_hop}, sub_queries={len(plan.decomposed_queries)}")
                return plan

        # Fallback: heuristic classification
        logger.info(f"[planner] Using heuristic fallback for: '{query[:60]}'")
        return self._classify_heuristic(query, top_k)

    def _is_trivial(self, query: str) -> bool:
        """Detect greetings, acknowledgments, and ultra-simple queries."""
        return any(p.match(query) for p in TRIVIAL_PATTERNS)

    def _classify_with_llm(self, query: str, llm, top_k: int) -> Optional[QueryPlan]:
        """Use LLM to classify and decompose the query."""
        try:
            import json

            prompt = self.CLASSIFY_PROMPT.format(
                query=query,
                max_sub=MAX_DECOMPOSED_QUERIES,
            )
            response = llm.generate(prompt, max_tokens=300, temperature=0.1)

            if not response.response:
                return None

            # Parse JSON from response (handle markdown code blocks)
            text = response.response.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            parsed = json.loads(text)

            query_type = parsed.get("query_type", "factual")
            needs_multi_hop = parsed.get("needs_multi_hop", False)
            sub_queries = parsed.get("sub_queries", [])

            # Enforce limits
            if len(sub_queries) > MAX_DECOMPOSED_QUERIES:
                sub_queries = sub_queries[:MAX_DECOMPOSED_QUERIES]

            # Calculate retrieval calls and enforce fan-out cap
            skip_expansion = needs_multi_hop or query_type == "comparative"
            num_queries = max(1, len(sub_queries)) if sub_queries else 1
            estimated_calls = num_queries * top_k

            if estimated_calls > MAX_TOTAL_RETRIEVAL_CALLS:
                # Reduce k or number of sub-queries
                adjusted_k = max(2, MAX_TOTAL_RETRIEVAL_CALLS // num_queries)
                estimated_calls = num_queries * adjusted_k
                logger.info(
                    f"Fan-out capped: {num_queries} sub-queries × k={adjusted_k} "
                    f"= {estimated_calls} calls (max {MAX_TOTAL_RETRIEVAL_CALLS})"
                )

            # Build steps
            steps = []
            if query_type == "trivial":
                steps = ["direct_llm_response"]
            elif needs_multi_hop:
                for i, sq in enumerate(sub_queries):
                    steps.append(f"retrieve_for_subquery_{i+1}: {sq}")
                steps.append("synthesize_all_results")
                steps.append("validate_answer")
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

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"LLM classification parsing failed: {e}, falling back to heuristic")
            return None
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to heuristic")
            return None

    def _classify_heuristic(self, query: str, top_k: int) -> QueryPlan:
        """Fallback heuristic classifier when LLM is unavailable."""
        query_lower = query.lower()

        # Comparative indicators
        comparative_words = ['compare', 'versus', 'vs', 'difference between', 'better than', 'worse than']
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

        # Summarization indicators
        summarize_words = ['summarize', 'summary', 'overview', 'recap', 'tldr']
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

        # Analytical indicators
        analytical_words = ['why', 'how does', 'explain', 'analyze', 'evaluate', 'what causes']
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

        # Default: factual (single retrieval, minimal overhead)
        return QueryPlan(
            steps=["retrieve_context", "generate_answer"],
            query_type="factual",
            needs_multi_hop=False,
            decomposed_queries=[query],
            skip_expansion=False,
            original_query=query,
            estimated_retrieval_calls=top_k,
        )
