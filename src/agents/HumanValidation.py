"""
Human Validation — Gatekeeper and review pipeline.

Routes low-confidence, ungrounded, or sensitive queries to human review.

Guardrails:
  • Confidence threshold check (avg retrieval score)
  • Groundedness check (from ValidatorAgent)
  • Sensitivity classifier (keyword rules + optional ML classifier)
"""

import re
import uuid
import json
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import psycopg2
from src.agents.BaseAgent import AgentResult
from src.utils.Logger import get_logger
from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = get_logger(__name__)


# ── Sensitivity patterns (pre-compiled once at import time) ─────────────
# Pre-compiling avoids per-query regex compilation overhead at high QPS.

SENSITIVE_PATTERNS = [
    re.compile(r'salary|compensation|pay\s*grade', re.IGNORECASE),
    re.compile(r'password|secret|api.?key|credentials|private.?key', re.IGNORECASE),
    re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
    re.compile(r'internal\s+only|confidential|restricted|classified', re.IGNORECASE),
    re.compile(r'personal\s+data|employee\s+record|social\s+security', re.IGNORECASE),
    re.compile(r'bank\s+account|credit\s+card|routing\s+number', re.IGNORECASE),
    re.compile(r'system\s+prompt|reveal\s+your\s+instructions', re.IGNORECASE),
    re.compile(r'pretend\s+(you\s+are|to\s+be)|act\s+as\s+if', re.IGNORECASE),
]


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ReviewItem:
    """A query pending human review."""
    review_id: str
    session_id: str
    query: str
    answer: str
    confidence: float
    reason: str
    grounded: Optional[bool]
    sensitive: bool
    created_at: str
    status: str = "pending"  # pending | approved | corrected | rejected


# ── Gatekeeper ──────────────────────────────────────────────────────────

class Gatekeeper:
    """Decides whether an answer should be escalated to human review.

    Checks (in order):
      1. Sensitivity (keyword patterns)
      2. Groundedness (from ValidatorAgent)
      3. Confidence (avg retrieval score)
    """

    CONFIDENCE_THRESHOLD = 0.4

    def should_escalate(self, query: str, agent_result: AgentResult) -> tuple:
        """Check if the result should be escalated.

        Returns:
            Tuple of (should_escalate: bool, reason: str)
        """
        # 1. Sensitivity check (always first — fast)
        if self._is_sensitive(query):
            logger.warning(f"[gatekeeper] ESCALATE: sensitive query detected — '{query[:60]}'")
            return True, "Sensitive query detected (keyword match)"

        # 2. Groundedness check (from ValidatorAgent)
        if agent_result.grounded is False:
            logger.warning(f"[gatekeeper] ESCALATE: answer not grounded (confidence={agent_result.confidence:.3f})")
            return True, "Answer not grounded in retrieved context"

        # 3. Confidence check
        if agent_result.confidence < self.CONFIDENCE_THRESHOLD:
            logger.warning(f"[gatekeeper] ESCALATE: low confidence ({agent_result.confidence:.3f} < {self.CONFIDENCE_THRESHOLD})")
            return True, f"Low confidence ({agent_result.confidence:.3f} < {self.CONFIDENCE_THRESHOLD})"

        logger.debug(f"[gatekeeper] PASS: confidence={agent_result.confidence:.3f}, grounded={agent_result.grounded}")
        return False, ""

    def _is_sensitive(self, query: str) -> bool:
        """Check query against sensitivity patterns."""
        return any(p.search(query) for p in SENSITIVE_PATTERNS)

    def create_review_item(
        self, session_id: str, query: str, agent_result: AgentResult, reason: str
    ) -> ReviewItem:
        """Create a review item and store it in the database."""
        review = ReviewItem(
            review_id=str(uuid.uuid4()),
            session_id=session_id,
            query=query,
            answer=agent_result.answer,
            confidence=agent_result.confidence,
            reason=reason,
            grounded=agent_result.grounded,
            sensitive=self._is_sensitive(query),
            created_at=datetime.utcnow().isoformat(),
        )

        self._store_review(review)
        self._log_audit(review)

        return review

    def _store_review(self, review: ReviewItem):
        """Store review in pending_reviews table."""
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                database=POSTGRES_DB, user=POSTGRES_USER,
                password=POSTGRES_PASSWORD, connect_timeout=10,
            )
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO pending_reviews 
                    (review_id, session_id, query, answer, confidence, reason, grounded, sensitive, created_at, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    review.review_id, review.session_id, review.query,
                    review.answer, review.confidence, review.reason,
                    review.grounded, review.sensitive, review.created_at, review.status,
                ))
            conn.commit()
            conn.close()
            logger.info(f"Review item created: {review.review_id}")
        except Exception as e:
            logger.error(f"Failed to store review item: {e}")

    def _log_audit(self, review: ReviewItem):
        """Log escalation to audit_log table."""
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                database=POSTGRES_DB, user=POSTGRES_USER,
                password=POSTGRES_PASSWORD, connect_timeout=10,
            )
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO audit_log (event_type, review_id, session_id, query, reason, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    "escalation", review.review_id, review.session_id,
                    review.query, review.reason, review.created_at,
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")


# ── Review Manager (for API endpoints) ──────────────────────────────────

class ReviewManager:
    """Manages human review workflow — used by API endpoints."""

    def _get_conn(self):
        return psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            database=POSTGRES_DB, user=POSTGRES_USER,
            password=POSTGRES_PASSWORD, connect_timeout=10,
        )

    def get_pending_reviews(self, limit: int = 50) -> List[dict]:
        """Fetch pending reviews."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT review_id, session_id, query, answer, confidence, reason, 
                           grounded, sensitive, created_at, status
                    FROM pending_reviews
                    WHERE status = 'pending'
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "review_id": r[0], "session_id": r[1], "query": r[2],
                    "answer": r[3], "confidence": r[4], "reason": r[5],
                    "grounded": r[6], "sensitive": r[7], "created_at": r[8],
                    "status": r[9],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to fetch pending reviews: {e}")
            return []

    def approve_review(self, review_id: str) -> bool:
        """Approve a pending review — caches the original answer."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE pending_reviews SET status = 'approved'
                    WHERE review_id = %s AND status = 'pending'
                """, (review_id,))

                # Log approval
                cursor.execute("""
                    INSERT INTO audit_log (event_type, review_id, reason, created_at)
                    VALUES ('approval', %s, 'Human approved', %s)
                """, (review_id, datetime.utcnow().isoformat()))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to approve review {review_id}: {e}")
            return False

    def correct_review(self, review_id: str, corrected_answer: str) -> bool:
        """Correct a pending review — stores the corrected answer for future use."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cursor:
                # Get the original query
                cursor.execute(
                    "SELECT query, session_id FROM pending_reviews WHERE review_id = %s",
                    (review_id,)
                )
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return False

                query, session_id = row

                # Update review status
                cursor.execute("""
                    UPDATE pending_reviews SET status = 'corrected', answer = %s
                    WHERE review_id = %s
                """, (corrected_answer, review_id))

                # Store approved answer for future cache lookup
                cursor.execute("""
                    INSERT INTO approved_answers (query, session_id, answer, review_id, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (query, session_id) DO UPDATE SET answer = EXCLUDED.answer
                """, (query, session_id, corrected_answer, review_id, datetime.utcnow().isoformat()))

                # Log correction
                cursor.execute("""
                    INSERT INTO audit_log (event_type, review_id, reason, created_at)
                    VALUES ('correction', %s, 'Human corrected answer', %s)
                """, (review_id, datetime.utcnow().isoformat()))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to correct review {review_id}: {e}")
            return False
