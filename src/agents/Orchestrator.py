"""
Agent Orchestrator — Coordinates the agent pipeline for query processing.

Replaces QueryHandler.process_query_with_response() — returns QueryResult
for API compatibility with the /ask endpoint.

Guardrails:
  • MAX_AGENT_STEPS = 5 (total steps across all agents)
  • MAX_TOOL_CALLS = 8 (total tool calls across entire orchestration)
  • TIMEOUT_SECONDS = 30 (hard timeout for entire orchestration)
  • TOKEN_BUDGET = 6000 (max tokens in combined context for synthesis)
  • Cache integration (ports QueryCache from QueryHandler)

Flow:
  Single-hop: RetrieverAgent → SynthesizerAgent → ValidatorAgent → result
  Multi-hop:  RetrieverAgent × n sub-queries → SynthesizerAgent → ValidatorAgent → result
  Trivial:    Direct LLM response (no agents, no retrieval)
"""

import re
import time
import concurrent.futures
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from src.agents.Planner import QueryPlanner, QueryPlan
from src.agents.Router import ConditionalRouter, RouteDecision, RouteResult
from src.agents.BaseAgent import AgentTask, AgentResult
from src.agents.RetrieverAgent import RetrieverAgent
from src.agents.SynthesizerAgent import SynthesizerAgent
from src.agents.ValidatorAgent import ValidatorAgent
from src.modules.QueryGeneration import QueryResult
from backend.cache.AuxiliaryCaches import get_cached_approved_answer, set_cached_approved_answer
from src.modules.QueryCache import get_retrieval_cache, get_llm_cache
from src.utils.llm_provider import get_shared_llm
from src.utils.Logger import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """Coordinates the multi-agent pipeline.

    Returns QueryResult (NOT a new FinalResult) for API compatibility.
    The /ask endpoint in routes.py unpacks QueryResult fields directly.
    """

    MAX_AGENT_STEPS = 5
    MAX_TOOL_CALLS = 8
    TIMEOUT_SECONDS = 30
    TOKEN_BUDGET = 6000

    # Conditional validation: only run ValidatorAgent when answer quality is uncertain.
    # This avoids a silent LLM cost multiplier for every simple factual query.
    VALIDATOR_CONFIDENCE_THRESHOLD = 0.7  # run validator if retrieval_conf < this
    VALIDATOR_ANSWER_LEN_THRESHOLD = 800  # run validator if answer len > this (chars)

    # Query length guard: reject/summarize extremely long prompts (prompt-stuffing)
    MAX_QUERY_CHARS = 2000

    def __init__(self):
        self.planner = QueryPlanner()
        self.router = ConditionalRouter()
        self.retriever_agent = RetrieverAgent()
        self.synthesizer_agent = SynthesizerAgent()
        self.validator_agent = ValidatorAgent()

    def run(
        self,
        query: str,
        retriever,
        embedding_service,
        session_id: str = "",
        top_k: int = 5,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryResult:
        """Execute the full agent pipeline for a query.

        This is the main entry point — replaces QueryHandler.process_query_with_response().

        Args:
            query: User's question.
            retriever: RAGRetriever instance for this session.
            embedding_service: Embedding service for query encoding.
            session_id: Session identifier.
            top_k: Number of chunks to retrieve.
            system_prompt: Optional custom system prompt.
            temperature: Optional LLM temperature.
            max_tokens: Optional LLM max tokens.

        Returns:
            QueryResult compatible with the /ask endpoint.
        """
        start_time = time.time()
        logger.info(f"[orchestrator] ===== QUERY START ===== session={session_id[:8]} query='{query[:80]}'")

        # 0. Query length guard (prompt-stuffing / latency protection)
        if len(query) > self.MAX_QUERY_CHARS:
            logger.warning(f"[orchestrator] Query too long ({len(query)} chars > {self.MAX_QUERY_CHARS}), truncating")
            query = query[:self.MAX_QUERY_CHARS] + "... [query truncated for safety]"

        # 0b. Check for human-approved answer (freshness-guarded by session_id)
        approved = self._lookup_approved_answer(session_id, query)
        if approved:
            return QueryResult(
                query=query,
                retrieved_chunks=[],
                metadata=[],
                retrieval_scores=[],
                llm_response=approved,
                llm_metadata={"route": "human_approved", "grounded": True, "confidence": 1.0},
            )

        # 1. Check retrieval cache first
        cached = get_retrieval_cache().get_cache(session_id, query, top_k)
        if cached:
            # Also check LLM cache
            cached_llm = get_llm_cache().get_cache(
                session_id, query, cached.get("retrieved_chunks", [])
            )
            if cached_llm:
                logger.info(f"Full cache hit for query: {query[:60]}...")
                return QueryResult(
                    query=query,
                    retrieved_chunks=cached.get("retrieved_chunks", []),
                    metadata=cached.get("metadata", []),
                    retrieval_scores=cached.get("retrieval_scores", []),
                    llm_response=cached_llm.get("llm_response"),
                    llm_metadata=cached_llm.get("llm_metadata"),
                )

        # 2. Expand follow-up query using recent history (if provided)
        normalized_history = self._normalize_chat_history(chat_history)
        expanded_query = self._expand_follow_up_query(query, normalized_history)

        # 3. Plan and route
        plan = self.planner.plan(expanded_query, top_k)
        logger.info(f"Query plan: type={plan.query_type}, multi_hop={plan.needs_multi_hop}")

        route = self.router.route(plan)
        logger.info(f"Route decision: {route.decision.value}")

        effective_k = route.adjusted_top_k or top_k

        # 4. Execute based on route
        operative_query = plan.decomposed_queries[0] if plan and plan.decomposed_queries else expanded_query
        if route.decision == RouteDecision.DIRECT_LLM:
            return self._handle_trivial(operative_query, session_id)

        elif route.decision == RouteDecision.SINGLE_AGENT:
            return self._handle_single_agent(
                query,
                operative_query,
                retriever,
                session_id,
                effective_k,
                system_prompt,
                temperature,
                max_tokens,
                start_time,
                retrieval_options=route.retrieval_options,
                synthesis_style=route.synthesis_style,
            )

        elif route.decision == RouteDecision.MULTI_AGENT:
            return self._handle_multi_agent(
                query,
                plan,
                retriever,
                session_id,
                effective_k,
                system_prompt,
                temperature,
                max_tokens,
                start_time,
                retrieval_options=route.retrieval_options,
            )

        else:
            # Fallback: single agent
            return self._handle_single_agent(
                query,
                operative_query,
                retriever,
                session_id,
                effective_k,
                system_prompt,
                temperature,
                max_tokens,
                start_time,
                retrieval_options=route.retrieval_options,
                synthesis_style=route.synthesis_style,
            )

    def _handle_trivial(self, query: str, session_id: str) -> QueryResult:
        """Handle trivial queries — direct LLM, no retrieval."""
        llm = get_shared_llm()
        if not llm:
            logger.error("[orchestrator] Trivial path: LLM singleton is None")
            return QueryResult(
                query=query,
                retrieved_chunks=[],
                metadata=[],
                llm_response="I'm sorry, I'm unable to respond right now.",
            )

        try:
            response = llm.generate(query, temperature=0.7, max_tokens=200)
            answer = re.sub(r'<think>.*?</think>', '', response.response, flags=re.DOTALL).strip()

            return QueryResult(
                query=query,
                retrieved_chunks=[],
                metadata=[],
                retrieval_scores=[],
                llm_response=answer,
                llm_metadata={
                    "model": response.model,
                    "total_tokens": response.total_tokens,
                    "route": "trivial_direct_llm",
                },
            )
        except Exception as e:
            logger.error(f"Trivial query handling failed: {e}")
            return QueryResult(
                query=query,
                retrieved_chunks=[],
                metadata=[],
                llm_response=f"Error: {str(e)}",
            )

    def _handle_single_agent(
        self,
        original_query,
        retrieval_query,
        retriever,
        session_id,
        top_k,
        system_prompt,
        temperature,
        max_tokens,
        start_time,
        retrieval_options: Optional[Dict[str, Any]] = None,
        synthesis_style: Optional[str] = None,
    ) -> QueryResult:
        """Handle factual/simple queries - retrieve -> synthesize -> validate."""

        task = AgentTask(
            query=retrieval_query,
            top_k=top_k,
            session_id=session_id,
            retriever=retriever,
            retrieval_options=retrieval_options or {},
        )
        retrieval_result = self.retriever_agent.run(task)

        if retrieval_result.error or not retrieval_result.retrieved_chunks:
            logger.warning(
                f"[orchestrator] Single-agent: retrieval returned 0 chunks or error: {retrieval_result.error}"
            )
            return QueryResult(
                query=original_query,
                retrieved_chunks=[],
                metadata=[],
                llm_response="I couldn't find relevant information to answer this question.",
            )

        avg_score = retrieval_result.confidence
        if self.router.should_escalate_to_human(avg_score):
            logger.info(f"Low confidence ({avg_score:.3f}), adding warning")

        self._cache_retrieval(session_id, original_query, top_k, retrieval_result)

        char_budget = self.TOKEN_BUDGET * 4
        compressed_chunks = self._compress_context(retrieval_result.retrieved_chunks, char_budget)
        if len(compressed_chunks) < len(retrieval_result.retrieved_chunks):
            logger.debug(
                f"[orchestrator] Context compressed: {len(retrieval_result.retrieved_chunks)} -> {len(compressed_chunks)} chunks"
            )
            retrieval_result.retrieved_chunks = compressed_chunks

        synth_task = AgentTask(
            query=self._build_synthesis_query(original_query, synthesis_style),
            previous_results=[retrieval_result],
            session_id=session_id,
        )
        synthesis_result = self.synthesizer_agent.run(synth_task)

        run_validator = (
            avg_score < self.VALIDATOR_CONFIDENCE_THRESHOLD
            or len(synthesis_result.answer) > self.VALIDATOR_ANSWER_LEN_THRESHOLD
        )
        if run_validator:
            logger.info(
                f"[orchestrator] Running ValidatorAgent (conf={avg_score:.2f}, ans_len={len(synthesis_result.answer)})"
            )
            validate_task = AgentTask(
                query=original_query,
                previous_results=[synthesis_result],
                session_id=session_id,
            )
            validation_result = self.validator_agent.run(validate_task)
        else:
            logger.debug(
                f"[orchestrator] Skipping ValidatorAgent (conf={avg_score:.2f} >= {self.VALIDATOR_CONFIDENCE_THRESHOLD}, ans_len={len(synthesis_result.answer)})"
            )
            validation_result = synthesis_result

        elapsed = time.time() - start_time
        logger.info(f"Single-agent pipeline completed in {elapsed:.2f}s")

        result = self._build_query_result(original_query, validation_result, retrieval_result, "single_agent")
        self._cache_llm(session_id, original_query, result)
        return result
    def run_stream(
        self,
        query: str,
        retriever,
        embedding_service,
        session_id: str = "",
        top_k: int = 5,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """Streaming version of run()."""
        import time as _time

        start_time = _time.time()
        logger.info(f"[orchestrator:stream] START session={session_id[:8]} query='{query[:80]}'")

        if len(query) > self.MAX_QUERY_CHARS:
            query = query[:self.MAX_QUERY_CHARS] + "... [query truncated for safety]"

        normalized_history = self._normalize_chat_history(chat_history)
        expanded_query = self._expand_follow_up_query(query, normalized_history)

        plan = self.planner.plan(expanded_query, top_k)
        route = self.router.route(plan)
        effective_k = route.adjusted_top_k or top_k
        operative_query = plan.decomposed_queries[0] if plan and plan.decomposed_queries else expanded_query

        if route.decision.value == "direct_llm":
            result = self._handle_trivial(operative_query, session_id)
            yield {"event": "chunk", "data": result.llm_response or ""}
            yield {"event": "success", "data": {"grounded": True, "confidence": 1.0}}
            return

        task = AgentTask(
            query=operative_query,
            top_k=effective_k,
            session_id=session_id,
            retriever=retriever,
            retrieval_options=route.retrieval_options,
        )
        retrieval_result = self.retriever_agent.run(task)

        if retrieval_result.error or not retrieval_result.retrieved_chunks:
            yield {"event": "error", "data": "I couldn't find relevant information to answer this question."}
            return

        yield {
            "event": "metadata",
            "data": {
                "sources": retrieval_result.sources,
                "retrieval_scores": retrieval_result.retrieval_scores,
                "metadata": retrieval_result.metadata,
            },
        }

        self._cache_retrieval(session_id, query, top_k, retrieval_result)

        char_budget = self.TOKEN_BUDGET * 4
        retrieval_result.retrieved_chunks = self._compress_context(retrieval_result.retrieved_chunks, char_budget)

        synth_task = AgentTask(
            query=self._build_synthesis_query(query, route.synthesis_style),
            previous_results=[retrieval_result],
            session_id=session_id,
        )

        accumulated_answer = []
        accumulated_chunks = []
        accumulated_sources = []

        for event_type, payload in self.synthesizer_agent.run_stream(synth_task):
            if event_type == "chunk":
                accumulated_answer.append(payload)
                yield {"event": "chunk", "data": payload}
            elif event_type == "sources":
                accumulated_sources = payload
            elif event_type == "all_chunks":
                accumulated_chunks = payload

        full_answer = "".join(accumulated_answer)
        logger.info(f"[orchestrator:stream] Validating answer ({len(full_answer)} chars)")

        synth_result_for_validation = type("_R", (), {
            "answer": full_answer,
            "retrieved_chunks": accumulated_chunks,
            "sources": accumulated_sources,
            "confidence": 0.8,
            "retrieval_scores": retrieval_result.retrieval_scores,
            "metadata": retrieval_result.metadata,
        })()

        try:
            validate_task = AgentTask(
                query=query,
                previous_results=[synth_result_for_validation],
                session_id=session_id,
            )
            validation_result = self.validator_agent.run(validate_task)

            if validation_result.grounded is False:
                yield {
                    "event": "warning",
                    "data": {
                        "grounded": False,
                        "confidence": validation_result.confidence,
                        "message": "Verification failed. This response was removed for safety.",
                    },
                }
            else:
                yield {
                    "event": "success",
                    "data": {
                        "grounded": validation_result.grounded,
                        "confidence": validation_result.confidence,
                    },
                }
        except Exception as e:
            logger.error(f"[orchestrator:stream] Validation error: {e}")
            yield {"event": "success", "data": {"grounded": None, "confidence": 0.7}}

        elapsed = _time.time() - start_time
        logger.info(f"[orchestrator:stream] DONE in {elapsed:.2f}s")

    def _handle_multi_agent(
        self, query, plan, retriever, session_id, top_k,
        system_prompt, temperature, max_tokens, start_time,
        retrieval_options: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Handle complex queries — multi-hop retrieval + synthesis + validation."""

        all_retrieval_results = []

        # Cap sub-queries to MAX_TOOL_CALLS before spawning any threads
        sub_queries = plan.decomposed_queries[:self.MAX_TOOL_CALLS]
        logger.info(
            f"[orchestrator] Multi-agent: running {len(sub_queries)} sub-queries "
            f"concurrently (capped from {len(plan.decomposed_queries)})"
        )

        # Run all RetrieverAgent calls in parallel — latency becomes max(individual)
        # instead of sum(individual), cutting 4-doc comparison from ~4x to ~1x retrieval time.
        def _retrieve_one(i_sub_query):
            i, sub_query = i_sub_query
            if self._check_timeout(start_time):
                logger.warning(f"[orchestrator] Sub-query {i+1} skipped — timeout already reached")
                return None
            logger.info(f"[orchestrator] Sub-query {i+1}/{len(sub_queries)}: '{sub_query[:60]}'")
            task = AgentTask(
                query=sub_query, top_k=top_k, session_id=session_id, retriever=retriever,
                retrieval_options=retrieval_options or {},
            )
            return self.retriever_agent.run(task)

        remaining_timeout = max(1.0, self.TIMEOUT_SECONDS - (time.time() - start_time))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
            futures = executor.map(_retrieve_one, enumerate(sub_queries), timeout=remaining_timeout)
            try:
                for result in futures:
                    if result is not None:
                        all_retrieval_results.append(result)
            except concurrent.futures.TimeoutError:
                logger.warning("[orchestrator] Multi-agent: parallel retrieval timed out, using partial results")

        if not all_retrieval_results or all(not r.retrieved_chunks for r in all_retrieval_results):
            logger.warning("[orchestrator] Multi-agent: all sub-queries returned 0 chunks")
            return QueryResult(
                query=query,
                retrieved_chunks=[],
                metadata=[],
                llm_response="I couldn't find relevant information across the sub-queries.",
            )

        # Synthesize all retrieval results
        synth_task = AgentTask(
            query=query,
            previous_results=all_retrieval_results,
            session_id=session_id,
        )
        synthesis_result = self.synthesizer_agent.run(synth_task)

        # Validate (always run on multi-agent path — higher hallucination risk)
        logger.info("[orchestrator] Running ValidatorAgent (multi-agent path always validates)")
        validate_task = AgentTask(
            query=query,
            previous_results=[synthesis_result],
            session_id=session_id,
        )
        validation_result = self.validator_agent.run(validate_task)

        # Merge retrieval results for the final QueryResult
        merged_retrieval = AgentResult(
            answer="",
            retrieved_chunks=[],
            retrieval_scores=[],
            metadata=[],
            sources=[],
            confidence=0.0,
        )
        for r in all_retrieval_results:
            merged_retrieval.retrieved_chunks.extend(r.retrieved_chunks)
            merged_retrieval.retrieval_scores.extend(r.retrieval_scores)
            merged_retrieval.metadata.extend(r.metadata)
            merged_retrieval.sources.extend(r.sources)

        if merged_retrieval.retrieval_scores:
            merged_retrieval.confidence = sum(merged_retrieval.retrieval_scores) / len(merged_retrieval.retrieval_scores)

        elapsed = time.time() - start_time
        logger.info(f"Multi-agent pipeline completed in {elapsed:.2f}s ({len(plan.decomposed_queries)} sub-queries)")

        result = self._build_query_result(query, validation_result, merged_retrieval, "multi_agent")

        # Cache
        self._cache_retrieval(session_id, query, top_k, merged_retrieval)
        self._cache_llm(session_id, query, result)

        return result

    def _build_query_result(
        self, query: str, validation_result: AgentResult,
        retrieval_result: AgentResult, route: str,
    ) -> QueryResult:
        """Convert agent results to a QueryResult for API compatibility."""

        # Build metadata list
        metadata_list = []
        for i, meta in enumerate(retrieval_result.metadata):
            metadata_list.append({
                "chunk_id": meta.get("chunk_id", ""),
                "document_id": meta.get("document_id", ""),
                "document_name": meta.get("document_name", ""),
                "page_number": meta.get("page_number"),
                "distance": 1.0 - meta.get("score", 0.0),
                "similarity_score": meta.get("score", 0.0),
            })

        return QueryResult(
            query=query,
            retrieved_chunks=retrieval_result.retrieved_chunks,
            metadata=metadata_list,
            retrieval_scores=retrieval_result.retrieval_scores,
            llm_response=validation_result.answer,
            llm_metadata={
                "route": route,
                "grounded": validation_result.grounded,
                "confidence": validation_result.confidence,
                "reasoning_trace": validation_result.reasoning_trace,
            },
        )

    def _normalize_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> List[Tuple[str, str]]:
        """Normalize incoming history payload into (role, content) tuples."""
        if not chat_history:
            return []
        normalized: List[Tuple[str, str]] = []
        for item in chat_history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant", "system"} and content:
                normalized.append((role, content))
        return normalized[-6:]

    def _expand_follow_up_query(self, query: str, history: List[Tuple[str, str]]) -> str:
        """Expand short follow-up prompts with the latest meaningful context."""
        q = query.strip()
        follow_up_prefixes = ("what about", "how about", "and", "also", "then", "in that case", "what if")
        is_short_follow_up = len(q.split()) <= 12 and q.lower().startswith(follow_up_prefixes)
        if not is_short_follow_up or not history:
            return q

        prior_user = next((content for role, content in reversed(history) if role == "user"), "")
        prior_assistant = next((content for role, content in reversed(history) if role == "assistant"), "")
        anchor = prior_user or prior_assistant
        if not anchor:
            return q

        anchor = anchor[:400]
        expanded = f"{anchor} | Follow-up: {q}"
        logger.info("[orchestrator] Expanded follow-up query for better retrieval")
        return expanded

    def _build_synthesis_query(self, query: str, synthesis_style: Optional[str]) -> str:
        """Inject style hints for synthesis without changing retrieval query semantics."""
        if synthesis_style == "structured_scenarios":
            return (
                "Answer in a structured format with 2-4 clearly labeled scenarios, "
                "state assumptions, and mention when user context is missing.\\n\\n"
                f"User query: {query}"
            )
        return query
    def _cache_retrieval(self, session_id, query, top_k, retrieval_result):
        """Cache retrieval results."""
        try:
            cache_data = {
                "retrieved_chunks": retrieval_result.retrieved_chunks,
                "metadata": retrieval_result.metadata,
                "retrieval_scores": retrieval_result.retrieval_scores,
            }
            get_retrieval_cache().set_cache(session_id, query, top_k, cache_data)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def _cache_llm(self, session_id, query, result):
        """Cache LLM results."""
        try:
            get_llm_cache().set_cache(
                session_id, query, result.retrieved_chunks,
                {"llm_response": result.llm_response, "llm_metadata": result.llm_metadata},
            )
        except Exception as e:
            logger.debug(f"LLM cache write failed: {e}")

    def _check_timeout(self, start_time: float) -> bool:
        """Check if orchestration has exceeded timeout."""
        return (time.time() - start_time) > self.TIMEOUT_SECONDS

    def _compress_context(self, chunks: list, char_budget: int) -> list:
        """Two-stage context compression to stay within TOKEN_BUDGET.

        Stage 1 (cheap): Truncate least-relevant chunks to fit.
        Stage 2 (expensive): LLM summarization ONLY if still >20% over budget.

        This keeps most queries fast — only complex overflows pay the LLM cost.
        """
        total_chars = sum(len(c) for c in chunks)
        if total_chars <= char_budget:
            return chunks  # Already fits, no work needed

        overflow_pct = (total_chars - char_budget) / char_budget

        # Stage 1: Cheap truncation (works for <20% overflow)
        if overflow_pct <= 0.20:
            logger.debug(f"[orchestrator] Context overflow {overflow_pct:.0%}: cheap truncation")
            truncated = []
            budget_remaining = char_budget
            for chunk in chunks:  # Chunks are ordered best-to-worst by score
                if budget_remaining <= 0:
                    break
                take = min(len(chunk), budget_remaining)
                truncated.append(chunk[:take])
                budget_remaining -= take
            return truncated

        # Stage 2: LLM summarization (for >20% overflow — worth the extra call)
        logger.warning(f"[orchestrator] Context overflow {overflow_pct:.0%} > 20%: LLM summarization")
        llm = get_shared_llm()
        if not llm:
            # Fallback to truncation if LLM unavailable
            return self._compress_context.__wrapped__(chunks, char_budget) if hasattr(self._compress_context, '__wrapped__') else chunks[:3]

        summarized = []
        for chunk in chunks:
            if len(chunk) <= char_budget // max(len(chunks), 1):
                summarized.append(chunk)
                continue
            try:
                resp = llm.generate(
                    f"Summarize this in 2 sentences:\n\n{chunk[:3000]}",
                    max_tokens=120, temperature=0.2
                )
                summarized.append(resp.response.strip() if resp.response else chunk[:300])
            except Exception:
                summarized.append(chunk[:300])  # Hard truncate on LLM error
        return summarized

    def _lookup_approved_answer(self, session_id: str, query: str) -> Optional[str]:
        """Check if a human-approved corrected answer exists for this query.

        Risk 5 freshness guard: only returns if the approved answer is still
        associated with the current session (same index version proxy).
        """
        try:
            import psycopg2
            from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

            conn = psycopg2.connect(
                host=POSTGRES_HOST, port=POSTGRES_PORT, database=POSTGRES_DB,
                user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=5,
            )
            with conn.cursor() as cursor:
                cursor.execute("SELECT to_regclass('public.approved_answers')")
                exists_row = cursor.fetchone()
                if not exists_row or not exists_row[0]:
                    conn.close()
                    return None
                # Freshness guard: only serve if approved within the same session
                # (session_id acts as a proxy for the current document set / index version)
                cursor.execute(
                    """
                    SELECT answer FROM approved_answers
                    WHERE query = %s AND session_id = %s
                    LIMIT 1
                    """,
                    (query.strip().lower(), session_id)
                )
                row = cursor.fetchone()
            conn.close()

            if row:
                # Refresh read-through cache from source-of-truth DB.
                set_cached_approved_answer(session_id, query, row[0])
                logger.info(f"[orchestrator] Serving human-approved answer for: '{query[:60]}'")
                return row[0]
        except Exception as e:
            logger.debug(f"[orchestrator] Approved answer lookup failed (non-critical): {e}")
            cached = get_cached_approved_answer(session_id, query)
            if cached:
                logger.info(
                    f"[orchestrator] DB unavailable; serving Redis-cached approved answer for: '{query[:60]}'"
                )
                return cached
        return None
