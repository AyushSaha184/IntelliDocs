"""
Tool Registry — Standardized tools available to agents.

Each tool returns a ToolResult dataclass with output, success flag, and optional error.
All tool calls are wrapped in try/except — errors are returned as text to agents
so they can self-correct rather than crashing the endpoint.

Tools:
  • retrieve_chunks     — wraps existing RAGRetriever
  • search_web          — optional external search (stub for future implementation)
  • calculate           — safe math evaluation (uses simpleeval, NEVER eval())
  • summarize_document  — cached or LLM-generated document summary
"""

import ast
import json
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Callable
import psycopg2
from src.utils.Logger import get_logger
from src.utils.llm_provider import get_shared_llm
from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = get_logger(__name__)


# ── ToolResult dataclass ────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Standardized result from any tool execution."""
    output: Any
    success: bool
    error: Optional[str] = None
    tool_name: str = ""


# ── Safe math evaluator ─────────────────────────────────────────────────

class SafeCalculator:
    """Safe arithmetic evaluator — NEVER uses eval().

    Uses ast.literal_eval for simple expressions and a custom
    AST walker for basic arithmetic (+, -, *, /, **, %).
    """

    SAFE_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if b != 0 else float('inf'),
        ast.Mod: lambda a, b: a % b if b != 0 else float('inf'),
        ast.Pow: lambda a, b: a ** b if abs(b) <= 100 else float('inf'),
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    def evaluate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression.

        Args:
            expression: Math expression string (e.g., "2 + 3 * 4").

        Returns:
            Numeric result.

        Raises:
            ValueError: If expression contains unsafe operations.
        """
        try:
            tree = ast.parse(expression.strip(), mode='eval')
            return self._eval_node(tree.body)
        except (SyntaxError, TypeError) as e:
            raise ValueError(f"Invalid math expression: {expression}") from e

    def _eval_node(self, node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.SAFE_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.SAFE_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self.SAFE_OPS[op_type](operand)

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ── Tool implementations ───────────────────────────────────────────────

_calculator = SafeCalculator()


def retrieve_chunks(query: str, top_k: int = 5, session_id: str = "", retriever=None) -> ToolResult:
    """Retrieve relevant chunks using the existing RAGRetriever.

    Args:
        query: Search query.
        top_k: Number of results.
        session_id: Session identifier.
        retriever: RAGRetriever instance (injected by the agent).

    Returns:
        ToolResult with list of retrieved chunk texts, scores, and enriched metadata.
    """
    if retriever is None:
        logger.warning("[tool:retrieve_chunks] No retriever provided")
        return ToolResult(
            output=None,
            success=False,
            error="No retriever provided. Cannot retrieve chunks.",
            tool_name="retrieve_chunks",
        )

    try:
        results = retriever.retrieve(query, k=top_k)
        chunks = [{"text": r.text, "chunk_id": r.chunk_id, "score": 1.0 - r.distance} for r in results]
        
        # Attach enriched metadata from DB (summary, keywords, hypothetical_questions)
        chunk_ids = [r.chunk_id for r in results]
        enriched = _fetch_enriched_metadata(chunk_ids)
        if enriched:
            for chunk in chunks:
                cid = chunk["chunk_id"]
                if cid in enriched:
                    chunk.update(enriched[cid])
        
        output = {"chunks": chunks, "count": len(results)}
        logger.debug(f"[tool:retrieve_chunks] Retrieved {len(results)} chunks")
        return ToolResult(output=output, success=True, tool_name="retrieve_chunks")
    except Exception as e:
        logger.error(f"[tool:retrieve_chunks] FAILED: {e}")
        return ToolResult(
            output=None,
            success=False,
            error=f"Retrieval failed: {str(e)}",
            tool_name="retrieve_chunks",
        )


def _fetch_enriched_metadata(chunk_ids: list) -> dict:
    """Fetch enriched metadata for chunk_ids from PostgreSQL. Returns {chunk_id: {summary, keywords, ...}}."""
    if not chunk_ids:
        return {}
    try:
        import psycopg2
        import json as _json
        from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
        conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT, database=POSTGRES_DB,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=5,
        )
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(chunk_ids))
            cur.execute(f"""
                SELECT chunk_id, summary, keywords, hypothetical_questions
                FROM chunks
                WHERE chunk_id IN ({placeholders}) AND enrichment_status = 'enriched'
            """, chunk_ids)
            rows = cur.fetchall()
        conn.close()
        
        result = {}
        for chunk_id, summary, keywords, questions in rows:
            meta = {}
            if summary:
                meta["summary"] = summary
            if keywords:
                try:
                    meta["keywords"] = _json.loads(keywords)
                except Exception:
                    meta["keywords"] = keywords
            if questions:
                try:
                    meta["hypothetical_questions"] = _json.loads(questions)
                except Exception:
                    meta["hypothetical_questions"] = questions
            if meta:
                result[chunk_id] = meta
        return result
    except Exception as e:
        logger.debug(f"[tool:retrieve_chunks] Enrichment lookup skipped: {e}")
        return {}


def search_web(query: str) -> ToolResult:
    """External web search — stub for future implementation.

    Returns:
        ToolResult indicating feature is not yet available.
    """
    return ToolResult(
        output=None,
        success=False,
        error="Web search is not yet implemented. Please rely on document-based retrieval.",
        tool_name="search_web",
    )


def calculate(expression: str) -> ToolResult:
    """Safely evaluate a mathematical expression.

    Uses AST-based evaluation — NEVER eval().

    Args:
        expression: Math expression (e.g., "42 * 3.14 + 100").

    Returns:
        ToolResult with numeric result.
    """
    try:
        result = _calculator.evaluate(expression)
        return ToolResult(output=result, success=True, tool_name="calculate")
    except ValueError as e:
        return ToolResult(
            output=None,
            success=False,
            error=f"Math error: {str(e)}. Please provide a valid arithmetic expression.",
            tool_name="calculate",
        )
    except Exception as e:
        return ToolResult(
            output=None,
            success=False,
            error=f"Calculation failed: {str(e)}",
            tool_name="calculate",
        )


def summarize_document(doc_id: str) -> ToolResult:
    """Return cached or LLM-generated document summary.

    Checks the document_summaries table first. On miss,
    retrieves chunk texts and generates a summary via LLM.

    Args:
        doc_id: Document identifier.

    Returns:
        ToolResult with summary text.
    """
    try:
        # 1. Check cache in PostgreSQL
        conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            database=POSTGRES_DB, user=POSTGRES_USER,
            password=POSTGRES_PASSWORD, connect_timeout=10,
        )

        with conn.cursor() as cursor:
            # Try cached summary first
            cursor.execute(
                "SELECT summary FROM document_summaries WHERE document_id = %s",
                (doc_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                conn.close()
                return ToolResult(output=row[0], success=True, tool_name="summarize_document")

            # No cache — fetch chunk texts for this document
            cursor.execute(
                "SELECT text FROM chunks WHERE document_id = %s ORDER BY chunk_index ASC LIMIT 20",
                (doc_id,)
            )
            chunk_rows = cursor.fetchall()

        if not chunk_rows:
            conn.close()
            return ToolResult(
                output=None,
                success=False,
                error=f"No chunks found for document {doc_id}",
                tool_name="summarize_document",
            )

        # 2. Generate summary via LLM
        llm = get_shared_llm()
        if not llm:
            conn.close()
            return ToolResult(
                output=None,
                success=False,
                error="LLM not available for summary generation",
                tool_name="summarize_document",
            )

        combined_text = "\n\n".join(row[0] for row in chunk_rows)
        # Truncate to fit context window
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "\n\n[...truncated...]"

        prompt = (
            "Summarize the following document content in 3-5 sentences. "
            "Be comprehensive but concise.\n\n"
            f"Content:\n{combined_text}\n\n"
            "Summary:"
        )
        response = llm.generate(prompt, max_tokens=300, temperature=0.3)
        summary = response.response.strip() if response.response else None

        if summary:
            # 3. Cache the summary
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO document_summaries (document_id, summary)
                    VALUES (%s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET summary = EXCLUDED.summary
                """, (doc_id, summary))
            conn.commit()

        conn.close()
        return ToolResult(output=summary, success=True, tool_name="summarize_document")

    except Exception as e:
        return ToolResult(
            output=None,
            success=False,
            error=f"Document summarization failed: {str(e)}",
            tool_name="summarize_document",
        )


# ── Tool Registry ──────────────────────────────────────────────────────

class ToolRegistry:
    """Registry of tools available to agents.

    Each tool call is wrapped in try/except — errors are returned as text
    to the calling agent so it can self-correct without crashing.
    """

    TOOL_MAP: Dict[str, Callable] = {
        "retrieve_chunks": retrieve_chunks,
        "search_web": search_web,
        "calculate": calculate,
        "summarize_document": summarize_document,
    }

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with error wrapping.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Tool-specific arguments.

        Returns:
            ToolResult (always — never raises).
        """
        if tool_name not in self.TOOL_MAP:
            return ToolResult(
                output=None,
                success=False,
                error=f"Unknown tool '{tool_name}'. Available: {list(self.TOOL_MAP.keys())}",
                tool_name=tool_name,
            )

        try:
            logger.debug(f"[tool_registry] Executing tool: {tool_name}")
            result = self.TOOL_MAP[tool_name](**kwargs)
            if not result.success:
                logger.warning(f"[tool_registry] Tool '{tool_name}' returned error: {result.error}")
            return result
        except TypeError as e:
            logger.error(f"[tool_registry] Tool '{tool_name}' bad args: {e}")
            return ToolResult(
                output=None,
                success=False,
                error=f"Tool '{tool_name}' invalid arguments: {str(e)}. Check parameter names.",
                tool_name=tool_name,
            )
        except Exception as e:
            logger.error(f"[tool_registry] Tool '{tool_name}' unexpected error: {e}", exc_info=True)
            return ToolResult(
                output=None,
                success=False,
                error=f"Tool '{tool_name}' failed: {str(e)}",
                tool_name=tool_name,
            )

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.TOOL_MAP.keys())
