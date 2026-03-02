"""
Metadata Enricher — Async post-ingest enrichment for chunk quality.

Runs as a background worker AFTER ingestion completes (never inline).
Enriches chunks with:
  • SummaryGenerator     — 2-3 sentence summary per chunk via LLM (true batching)
  • KeywordExtractor     — Term-frequency based lightweight keyword extraction
  • QuestionGenerator    — hypothetical questions per chunk (HyDE-style retrieval)

Pattern:
  ingest → save chunks → mark as "pending_enrichment" → return to user
                            ↓
            background worker picks up batches → enriches → updates DB

All enrichment is:
  • Async      — never blocks ingestion
  • Batched    — processes chunks in groups to reduce LLM calls
  • Optional   — controlled by ENABLE_CHUNK_ENRICHMENT config flag
  • Resumable  — re-processes only "pending" chunks on restart
"""

import json
import re
import math
import time
import threading
from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
import psycopg2
from src.utils.Logger import get_logger
from src.utils.llm_provider import get_shared_llm
from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = get_logger(__name__)

# Configuration
ENRICHMENT_BATCH_SIZE = 10         # Chunks per DB fetch batch
LLM_BATCH_SIZE = 5                 # Chunks per single LLM call
MAX_KEYWORDS_PER_CHUNK = 10
MAX_QUESTIONS_PER_CHUNK = 3
MIN_CHUNK_LENGTH = 200             # Skip enrichment for tiny chunks
MAX_LLM_RETRIES = 2                # Retry transient LLM failures

# Concurrency cap: prevents bulk-upload enrichment from overwhelming LLM quota
MAX_ENRICH_WORKERS = 3
_enrich_semaphore = threading.Semaphore(MAX_ENRICH_WORKERS)


# ── Dataclass for enrichment results ────────────────────────────────────

@dataclass
class EnrichmentResult:
    """Enrichment output for a single chunk."""
    chunk_id: str
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    hypothetical_questions: Optional[List[str]] = None
    success: bool = True
    error: Optional[str] = None


# ── KeywordExtractor (term-frequency based, no LLM needed) ─────────────

class KeywordExtractor:
    """Extract keywords using term-frequency scoring.

    This is a lightweight, LLM-free extractor that uses raw TF
    with length normalization. For richer keywords, use the
    LLM-based prompt in SummaryGenerator's batch call.
    """

    # Common English stop words
    STOP_WORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but',
        'and', 'or', 'if', 'while', 'this', 'that', 'these', 'those',
        'it', 'its', 'he', 'she', 'they', 'them', 'we', 'us', 'you',
        'your', 'my', 'his', 'her', 'their', 'our', 'what', 'which',
        'who', 'whom', 'also', 'about', 'up', 'one', 'two', 'three',
    })

    _WORD_PATTERN = re.compile(r'[a-zA-Z]{3,}')

    def extract(self, text: str, max_keywords: int = MAX_KEYWORDS_PER_CHUNK) -> List[str]:
        """Extract top keywords from text using term frequency.

        Args:
            text: Chunk text.
            max_keywords: Maximum keywords to return.

        Returns:
            List of keywords sorted by relevance.
        """
        if not text:
            return []

        # Tokenize and filter
        words = self._WORD_PATTERN.findall(text.lower())
        words = [w for w in words if w not in self.STOP_WORDS and len(w) >= 3]

        if not words:
            return []

        # Term frequency with length normalization
        tf = Counter(words)
        total = len(words)

        scored = [(word, count / total) for word, count in tf.items()]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [word for word, _ in scored[:max_keywords]]


# ── SummaryGenerator (LLM-based, true batching) ───────────────────────

class SummaryGenerator:
    """Generate concise summaries for chunks using the shared LLM.
    
    Supports true batching: multiple chunks are sent in a single LLM call
    to reduce cost and improve throughput.
    """

    PROMPT_TEMPLATE = (
        "Summarize the following text in 2-3 sentences. Be concise and factual.\n\n"
        "Text:\n{text}\n\n"
        "Summary:"
    )

    BATCH_PROMPT_TEMPLATE = (
        "Summarize each of the following {n} text passages in 2-3 sentences each. "
        "Be concise and factual. Return summaries separated by \"---\" on its own line.\n\n"
        "{passages}\n\n"
        "Summaries (separated by ---):"
    )

    def generate(self, text: str) -> Optional[str]:
        """Generate a summary for a single chunk (with retry)."""
        llm = get_shared_llm()
        if not llm:
            logger.warning("SummaryGenerator: No LLM available, skipping")
            return None

        for attempt in range(MAX_LLM_RETRIES):
            try:
                prompt = self.PROMPT_TEMPLATE.format(text=text[:3000])
                response = llm.generate(prompt, max_tokens=150, temperature=0.3)
                return response.response.strip() if response.response else None
            except Exception as e:
                if attempt < MAX_LLM_RETRIES - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                logger.warning(f"SummaryGenerator failed after {MAX_LLM_RETRIES} attempts: {e}")
                return None

    def generate_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Generate summaries for multiple chunks in a single LLM call.
        
        Returns:
            List of summary strings (or None for failures), same length as input.
        """
        if not texts:
            return []
            
        llm = get_shared_llm()
        if not llm:
            return [None] * len(texts)

        passages = "\n\n".join(
            f"[Passage {i+1}]\n{t[:3000]}" for i, t in enumerate(texts)
        )
        prompt = self.BATCH_PROMPT_TEMPLATE.format(n=len(texts), passages=passages)
        
        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = llm.generate(prompt, max_tokens=150 * len(texts), temperature=0.3)
                if not response.response:
                    return [None] * len(texts)
                    
                parts = response.response.strip().split('---')
                results = [p.strip() or None for p in parts]
                
                # Pad or trim to match input length
                while len(results) < len(texts):
                    results.append(None)
                return results[:len(texts)]
            except Exception as e:
                if attempt < MAX_LLM_RETRIES - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                logger.warning(f"SummaryGenerator batch failed after {MAX_LLM_RETRIES} attempts: {e}")
                return [None] * len(texts)


# ── QuestionGenerator (LLM-based, with retry) ─────────────────────────

class QuestionGenerator:
    """Generate hypothetical questions a chunk could answer (HyDE-style)."""

    PROMPT_TEMPLATE = (
        "Given the following text, generate exactly {n} questions that this text "
        "could answer. Return ONLY the questions, one per line, with no numbering or bullets.\n\n"
        "Text:\n{text}\n\n"
        "Questions:"
    )

    def generate(self, text: str, n: int = MAX_QUESTIONS_PER_CHUNK) -> Optional[List[str]]:
        """Generate hypothetical questions for a chunk (with retry)."""
        llm = get_shared_llm()
        if not llm:
            logger.warning("QuestionGenerator: No LLM available, skipping")
            return None

        for attempt in range(MAX_LLM_RETRIES):
            try:
                prompt = self.PROMPT_TEMPLATE.format(text=text[:3000], n=n)
                response = llm.generate(prompt, max_tokens=200, temperature=0.5)

                if not response.response:
                    return None

                lines = response.response.strip().split('\n')
                questions = []
                for line in lines:
                    line = line.strip()
                    line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line).strip()
                    if line and line.endswith('?'):
                        questions.append(line)

                return questions[:n] if questions else None
            except Exception as e:
                if attempt < MAX_LLM_RETRIES - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                logger.warning(f"QuestionGenerator failed after {MAX_LLM_RETRIES} attempts: {e}")
                return None


# ── MetadataEnricher (orchestrator) ─────────────────────────────────────

class MetadataEnricher:
    """Orchestrates async chunk enrichment with true LLM batching.

    Usage:
        enricher = MetadataEnricher()
        enricher.enrich_pending_chunks(batch_size=10)
    """

    def __init__(self):
        self.keyword_extractor = KeywordExtractor()
        self.summary_generator = SummaryGenerator()
        self.question_generator = QuestionGenerator()
        # Observability counters
        self._llm_calls = 0
        self._llm_failures = 0
        self._keywords_extracted = 0
        self._chunks_skipped = 0

    def _get_db_connection(self):
        """Get a PostgreSQL connection."""
        return psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=10,
        )

    def enrich_single_chunk(self, chunk_id: str, text: str) -> EnrichmentResult:
        """Enrich a single chunk. Gated by MAX_ENRICH_WORKERS semaphore."""
        with _enrich_semaphore:
            # Length guard: skip tiny chunks that add noise
            if len(text) < MIN_CHUNK_LENGTH:
                self._chunks_skipped += 1
                return EnrichmentResult(
                    chunk_id=chunk_id,
                    keywords=self.keyword_extractor.extract(text),
                    success=True,
                )
                
            try:
                keywords = self.keyword_extractor.extract(text)
                self._keywords_extracted += len(keywords)
                
                self._llm_calls += 1
                summary = self.summary_generator.generate(text)
                
                self._llm_calls += 1
                questions = self.question_generator.generate(text)

                return EnrichmentResult(
                    chunk_id=chunk_id,
                    summary=summary,
                    keywords=keywords,
                    hypothetical_questions=questions,
                    success=True,
                )
            except Exception as e:
                self._llm_failures += 1
                logger.error(f"Enrichment failed for chunk {chunk_id}: {e}")
                return EnrichmentResult(
                    chunk_id=chunk_id,
                    success=False,
                    error=str(e),
                )

    def _enrich_batch_llm(self, rows: List[Tuple[str, str]]) -> List[EnrichmentResult]:
        """True batched enrichment: sends multiple chunks per LLM call."""
        results: List[EnrichmentResult] = []
        
        # Split into sub-batches for LLM batching
        for i in range(0, len(rows), LLM_BATCH_SIZE):
            sub_batch = rows[i:i + LLM_BATCH_SIZE]
            
            # Separate enrichable chunks from tiny ones
            enrichable = [(cid, txt) for cid, txt in sub_batch if len(txt) >= MIN_CHUNK_LENGTH]
            tiny = [(cid, txt) for cid, txt in sub_batch if len(txt) < MIN_CHUNK_LENGTH]
            
            # Handle tiny chunks without LLM
            for chunk_id, text in tiny:
                self._chunks_skipped += 1
                keywords = self.keyword_extractor.extract(text)
                self._keywords_extracted += len(keywords)
                results.append(EnrichmentResult(
                    chunk_id=chunk_id,
                    keywords=keywords,
                    success=True,
                ))
            
            if not enrichable:
                continue
                
            # Extract keywords (no LLM needed)
            all_keywords = []
            for _, text in enrichable:
                kw = self.keyword_extractor.extract(text)
                self._keywords_extracted += len(kw)
                all_keywords.append(kw)
            
            # Batch summary generation (single LLM call)
            texts = [txt for _, txt in enrichable]
            self._llm_calls += 1
            summaries = self.summary_generator.generate_batch(texts)
            
            # Questions still per-chunk (harder to batch reliably)
            questions_list = []
            for _, text in enrichable:
                self._llm_calls += 1
                questions_list.append(self.question_generator.generate(text))
            
            for j, (chunk_id, text) in enumerate(enrichable):
                summary = summaries[j] if j < len(summaries) else None
                questions = questions_list[j] if j < len(questions_list) else None
                
                if summary is None and questions is None:
                    self._llm_failures += 1
                    
                results.append(EnrichmentResult(
                    chunk_id=chunk_id,
                    summary=summary,
                    keywords=all_keywords[j],
                    hypothetical_questions=questions,
                    success=True,
                ))
        
        return results

    def enrich_pending_chunks(self, batch_size: int = ENRICHMENT_BATCH_SIZE) -> int:
        """Process all pending chunks in the database.

        Fetches chunks with enrichment_status='pending', enriches them,
        and updates the database. Resumable — only processes pending chunks.
        Uses true LLM batching for summaries.

        Args:
            batch_size: Number of chunks to process per DB fetch batch.

        Returns:
            Total number of chunks enriched.
        """
        total_enriched = 0

        try:
            conn = self._get_db_connection()

            while True:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT chunk_id, text FROM chunks
                        WHERE enrichment_status = 'pending'
                        ORDER BY created_timestamp ASC
                        LIMIT %s
                    """, (batch_size,))
                    rows = cursor.fetchall()

                if not rows:
                    break

                logger.info(f"Enriching batch of {len(rows)} chunks")

                # True batched enrichment
                batch_results = self._enrich_batch_llm(rows)

                # Batch DB update — single commit per batch
                with conn.cursor() as cursor:
                    for result in batch_results:
                        if result.success:
                            cursor.execute("""
                                UPDATE chunks SET
                                    summary = %s,
                                    keywords = %s,
                                    hypothetical_questions = %s,
                                    enrichment_status = 'enriched'
                                WHERE chunk_id = %s
                            """, (
                                result.summary,
                                json.dumps(result.keywords) if result.keywords else None,
                                json.dumps(result.hypothetical_questions) if result.hypothetical_questions else None,
                                result.chunk_id,
                            ))
                            total_enriched += 1
                        else:
                            cursor.execute("""
                                UPDATE chunks SET enrichment_status = 'failed'
                                WHERE chunk_id = %s
                            """, (result.chunk_id,))

                conn.commit()  # Single commit per batch, not per chunk

                logger.info(f"Enriched {total_enriched} chunks so far")

            conn.close()

        except Exception as e:
            logger.error(f"Enrichment worker error: {e}")

        logger.info(
            f"Enrichment complete: {total_enriched} enriched, "
            f"{self._llm_calls} LLM calls, "
            f"{self._llm_failures} failures, "
            f"{self._chunks_skipped} skipped (too small), "
            f"{self._keywords_extracted} keywords extracted"
        )
        return total_enriched
