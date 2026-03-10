"""
Base Agent — Abstract base class for all specialized agents.

Provides:
  • AgentTask / AgentResult dataclasses
  • Shared tool calling with per-agent hard limits
  • Timeout enforcement
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from src.agents.Tools import ToolRegistry, ToolResult
from src.utils.Logger import get_logger

logger = get_logger(__name__)


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class AgentTask:
    """Input to an agent's run() method."""
    query: str
    context: str = ""
    tools_available: List[str] = field(default_factory=list)
    previous_results: List['AgentResult'] = field(default_factory=list)
    top_k: int = 5
    session_id: str = ""
    retriever: Any = None  # RAGRetriever instance, injected by orchestrator
    retrieval_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Output from an agent's run() method."""
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    grounded: Optional[bool] = None   # Set by ValidatorAgent
    retrieved_chunks: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    metadata: List[dict] = field(default_factory=list)
    error: Optional[str] = None


# ── BaseAgent ───────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base class for all agents.

    Guardrails:
      • MAX_TOOL_CALLS = 3 per agent (prevents infinite tool loops)
      • TIMEOUT_SECONDS = 15 per agent (prevents hangs)
    """

    MAX_TOOL_CALLS: int = 3
    TIMEOUT_SECONDS: float = 15.0

    def __init__(self, agent_id: str, role: str, tools: Optional[List[str]] = None):
        self.agent_id = agent_id
        self.role = role
        self.tools = tools or []
        self._tool_call_count = 0
        self._start_time: Optional[float] = None
        self._tool_registry = ToolRegistry()

    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult:
        """Execute the agent's task. Must be implemented by subclasses."""
        ...

    def _call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Call a tool with guardrails (count limit + timeout check).

        Returns:
            ToolResult — always returns, never raises.
        """
        logger.debug(f"[{self.agent_id}] Tool call #{self._tool_call_count + 1}: {tool_name}")

        # Check tool call limit
        self._tool_call_count += 1
        if self._tool_call_count > self.MAX_TOOL_CALLS:
            msg = (
                f"Agent {self.agent_id} exceeded max tool calls "
                f"({self.MAX_TOOL_CALLS}). Stopping."
            )
            logger.warning(msg)
            return ToolResult(output=None, success=False, error=msg, tool_name=tool_name)

        # Check timeout
        if self._start_time and (time.time() - self._start_time) > self.TIMEOUT_SECONDS:
            elapsed = time.time() - self._start_time
            msg = (
                f"Agent {self.agent_id} exceeded timeout "
                f"({elapsed:.1f}s > {self.TIMEOUT_SECONDS}s). Stopping."
            )
            logger.warning(msg)
            return ToolResult(output=None, success=False, error=msg, tool_name=tool_name)

        # Check tool is in allowed list
        if tool_name not in self.tools:
            logger.warning(f"[{self.agent_id}] Disallowed tool '{tool_name}' requested. Allowed: {self.tools}")
            return ToolResult(
                output=None,
                success=False,
                error=f"Tool '{tool_name}' not available to agent {self.agent_id}. Available: {self.tools}",
                tool_name=tool_name,
            )

        # Execute tool
        result = self._tool_registry.execute_tool(tool_name, **kwargs)
        logger.debug(f"[{self.agent_id}] Tool '{tool_name}' result: success={result.success}")
        return result

    def _start_timer(self):
        """Start the agent's timeout timer."""
        self._start_time = time.time()
        self._tool_call_count = 0
        logger.debug(f"[{self.agent_id}] Agent started (timeout={self.TIMEOUT_SECONDS}s, max_tools={self.MAX_TOOL_CALLS})")

    def _check_timeout(self) -> bool:
        """Check if the agent has exceeded its timeout."""
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) > self.TIMEOUT_SECONDS
