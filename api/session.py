"""Session store: Protocol definition + thread-safe in-memory implementation."""
from __future__ import annotations

import threading
import time
from typing import Protocol


class SessionStore(Protocol):
    def get(self, conversation_id: str, user_id: str) -> dict | None: ...
    def set(self, conversation_id: str, state: dict, user_id: str) -> None: ...
    def delete(self, conversation_id: str, user_id: str) -> None: ...
    def list_ids(self, user_id: str) -> list[str]: ...


class InMemorySessionStore:
    """Thread-safe dict-backed store with TTL eviction.

    user_id is accepted on every method to match the SessionStore protocol but
    is not used for scoping — the in-memory store is single-process and
    conversation_ids are UUIDs, so collisions across users cannot occur in
    practice.  Phase 2 prompt 2 replaces this with PostgresSessionStore where
    user_id is enforced at the DB level.
    """

    TTL: float = 3600.0  # 1 hour

    def __init__(self) -> None:
        self._store: dict[str, tuple[dict, float]] = {}
        self._lock = threading.Lock()

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.TTL
        expired = [k for k, (_, ts) in self._store.items() if ts < cutoff]
        for k in expired:
            del self._store[k]

    def get(self, conversation_id: str, user_id: str) -> dict | None:
        with self._lock:
            self._evict_expired()
            entry = self._store.get(conversation_id)
            if entry is None:
                return None
            state, _ = entry
            self._store[conversation_id] = (state, time.time())
            return state

    def set(self, conversation_id: str, state: dict, user_id: str) -> None:
        with self._lock:
            self._store[conversation_id] = (state, time.time())

    def delete(self, conversation_id: str, user_id: str) -> None:
        with self._lock:
            self._store.pop(conversation_id, None)

    def list_ids(self, user_id: str) -> list[str]:
        with self._lock:
            return list(self._store.keys())
