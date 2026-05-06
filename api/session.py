"""Session store: Protocol definition + thread-safe in-memory implementation."""
from __future__ import annotations

import threading
import time
from typing import Protocol


class SessionStore(Protocol):
    def get(self, conversation_id: str) -> dict | None: ...
    def set(self, conversation_id: str, state: dict) -> None: ...
    def delete(self, conversation_id: str) -> None: ...
    def list_ids(self) -> list[str]: ...  # debug only


class InMemorySessionStore:
    """Thread-safe dict-backed store with TTL eviction.

    Sessions are evicted lazily on access once they exceed TTL.  Phase 2 will
    replace this with PostgresSessionStore implementing the same Protocol.
    """

    TTL: float = 3600.0  # 1 hour

    def __init__(self) -> None:
        # Maps conversation_id → (state_dict, last_accessed_epoch)
        self._store: dict[str, tuple[dict, float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove sessions older than TTL.  Must be called under self._lock."""
        cutoff = time.time() - self.TTL
        expired = [k for k, (_, ts) in self._store.items() if ts < cutoff]
        for k in expired:
            del self._store[k]

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def get(self, conversation_id: str) -> dict | None:
        with self._lock:
            self._evict_expired()
            entry = self._store.get(conversation_id)
            if entry is None:
                return None
            state, _ = entry
            # Refresh last-accessed timestamp.
            self._store[conversation_id] = (state, time.time())
            return state

    def set(self, conversation_id: str, state: dict) -> None:
        with self._lock:
            self._store[conversation_id] = (state, time.time())

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._store.pop(conversation_id, None)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())
