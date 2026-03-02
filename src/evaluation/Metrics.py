"""
Metrics Collector — Tracks RAG pipeline performance metrics.

Metrics:
  • Retrieval: precision@k, recall@k, MRR
  • Latency: avg response time, p95/p99
  • Cost: LLM token usage per query
  • Quality: cache hit rate, groundedness rate, escalation rate
  • Evaluation: LLMJudge scores (relevance, faithfulness, completeness)
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque
from datetime import datetime, timedelta
import psycopg2
from src.utils.Logger import get_logger
from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = get_logger(__name__)


@dataclass
class QueryMetric:
    """Metrics for a single query execution."""
    session_id: str
    query: str
    latency_ms: float
    retrieval_scores: List[float] = field(default_factory=list)
    total_tokens: int = 0
    cache_hit: bool = False
    grounded: Optional[bool] = None
    escalated: bool = False
    route: str = ""  # trivial | single_agent | multi_agent
    eval_relevance: Optional[float] = None
    eval_faithfulness: Optional[float] = None
    eval_completeness: Optional[float] = None
    eval_overall: Optional[float] = None
    timestamp: str = ""


class MetricsCollector:
    """Collects and aggregates pipeline metrics.

    Thread-safe. Stores metrics in-memory (rolling window) and
    periodically flushes to PostgreSQL for persistence.
    """

    ROLLING_WINDOW = 1000  # Keep last N metrics in memory

    def __init__(self):
        self._metrics: deque = deque(maxlen=self.ROLLING_WINDOW)
        self._lock = threading.Lock()
        self._total_queries = 0
        self._cache_hits = 0
        self._escalations = 0
        self._grounded_count = 0
        self._ungrounded_count = 0

    def record(self, metric: QueryMetric):
        """Record a query metric."""
        metric.timestamp = datetime.utcnow().isoformat()

        with self._lock:
            self._metrics.append(metric)
            self._total_queries += 1
            if metric.cache_hit:
                self._cache_hits += 1
            if metric.escalated:
                self._escalations += 1
            if metric.grounded is True:
                self._grounded_count += 1
            elif metric.grounded is False:
                self._ungrounded_count += 1

        # Async flush to DB
        self._flush_to_db(metric)

    def get_summary(self, last_n: Optional[int] = None) -> Dict:
        """Get aggregated metrics summary.

        Args:
            last_n: Only consider last N queries. None = all in rolling window.

        Returns:
            Dict with aggregated metrics.
        """
        with self._lock:
            metrics = list(self._metrics)

        if last_n:
            metrics = metrics[-last_n:]

        if not metrics:
            return {"total_queries": 0, "message": "No metrics collected yet"}

        latencies = [m.latency_ms for m in metrics]
        token_counts = [m.total_tokens for m in metrics if m.total_tokens > 0]
        eval_scores = [m.eval_overall for m in metrics if m.eval_overall is not None]

        # Calculate retrieval metrics
        all_scores = []
        for m in metrics:
            all_scores.extend(m.retrieval_scores)

        # Route distribution
        routes = {}
        for m in metrics:
            routes[m.route] = routes.get(m.route, 0) + 1

        summary = {
            "total_queries": self._total_queries,
            "window_size": len(metrics),
            "latency": {
                "avg_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
                "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            },
            "tokens": {
                "avg_per_query": sum(token_counts) / len(token_counts) if token_counts else 0,
                "total": sum(token_counts),
            },
            "retrieval": {
                "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            },
            "cache_hit_rate": self._cache_hits / self._total_queries if self._total_queries > 0 else 0,
            "escalation_rate": self._escalations / self._total_queries if self._total_queries > 0 else 0,
            "groundedness_rate": (
                self._grounded_count / (self._grounded_count + self._ungrounded_count)
                if (self._grounded_count + self._ungrounded_count) > 0 else None
            ),
            "eval_scores": {
                "avg_overall": sum(eval_scores) / len(eval_scores) if eval_scores else None,
                "count": len(eval_scores),
            },
            "route_distribution": routes,
        }

        return summary

    def _flush_to_db(self, metric: QueryMetric):
        """Persist a metric to PostgreSQL (fire-and-forget)."""
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                database=POSTGRES_DB, user=POSTGRES_USER,
                password=POSTGRES_PASSWORD, connect_timeout=5,
            )
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO eval_metrics 
                    (session_id, query, latency_ms, total_tokens, cache_hit, grounded,
                     escalated, route, eval_relevance, eval_faithfulness, 
                     eval_completeness, eval_overall, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    metric.session_id, metric.query, metric.latency_ms,
                    metric.total_tokens, metric.cache_hit, metric.grounded,
                    metric.escalated, metric.route,
                    metric.eval_relevance, metric.eval_faithfulness,
                    metric.eval_completeness, metric.eval_overall,
                    metric.timestamp,
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Metrics DB flush failed (non-critical): {e}")


# ── Global singleton ────────────────────────────────────────────────────

_metrics_collector: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get or create singleton MetricsCollector."""
    global _metrics_collector
    with _metrics_lock:
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector()
        return _metrics_collector
