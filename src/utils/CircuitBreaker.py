"""Simple thread-safe circuit breaker for external API calls."""

from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitBreakerOpenError(RuntimeError):
    """Raised when circuit breaker is OPEN and calls are blocked."""


class CircuitBreaker:
    """Thread-safe circuit breaker with CLOSED -> OPEN -> HALF_OPEN states."""

    def __init__(self, name: str, failure_threshold: int = 10, recovery_seconds: int = 120):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_seconds = max(1, int(recovery_seconds))

        self._state = "CLOSED"
        self._failure_count = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def _allow_call(self) -> bool:
        now = time.time()
        with self._lock:
            if self._state == "CLOSED":
                return True
            if self._state == "OPEN":
                if now - self._opened_at >= self.recovery_seconds:
                    self._state = "HALF_OPEN"
                    return True
                return False
            # HALF_OPEN allows a probe request
            return True

    def _on_success(self) -> None:
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._opened_at = 0.0

    def _on_failure(self) -> None:
        now = time.time()
        with self._lock:
            self._failure_count += 1
            if self._state == "HALF_OPEN" or self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = now

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        if not self._allow_call():
            raise CircuitBreakerOpenError(
                f"Circuit '{self.name}' is OPEN. Retry after {self.recovery_seconds}s."
            )

        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise

        self._on_success()
        return result
