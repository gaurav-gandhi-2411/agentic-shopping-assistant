"""Dependency-injection providers for FastAPI routes.

Singletons (retriever, LLM, catalogue) are loaded once during the lifespan
context in main.py and stored here as module-level variables.  Route handlers
access them via FastAPI's Depends() or by calling the getters directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from api.session import InMemorySessionStore, SessionStore

if TYPE_CHECKING:
    from src.retrieval.hybrid_search import HybridRetriever

# ---------------------------------------------------------------------------
# Module-level singletons — set by main.py lifespan, never mutated after that.
# ---------------------------------------------------------------------------

_retriever: Any = None
_catalogue_df: pd.DataFrame | None = None
_llm: Any = None
_config: dict | None = None
_session_store: SessionStore = InMemorySessionStore()

# ---------------------------------------------------------------------------
# Auth placeholder — Phase 2 prompt 2 replaces get_current_user_id() body
# with JWT extraction; the call chain in chat.py stays unchanged.
# ---------------------------------------------------------------------------

DEV_USER_ID: str = "00000000-0000-0000-0000-000000000001"


def get_current_user_id() -> str:
    """Return the current user's ID.

    Hardcoded to DEV_USER_ID until JWT middleware is wired in Phase 2 prompt 2.
    At that point this function will extract the sub claim from the Bearer token
    and return it; the rest of the call chain (chat.py → store.get/set) stays
    identical.
    """
    return DEV_USER_ID


# ---------------------------------------------------------------------------
# Setters called by lifespan
# ---------------------------------------------------------------------------

def _init(
    retriever: Any,
    catalogue_df: pd.DataFrame,
    llm: Any,
    config: dict,
    session_store: SessionStore | None = None,
) -> None:
    global _retriever, _catalogue_df, _llm, _config, _session_store
    _retriever = retriever
    _catalogue_df = catalogue_df
    _llm = llm
    _config = config
    if session_store is not None:
        _session_store = session_store


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------

def get_retriever() -> "HybridRetriever":
    assert _retriever is not None, "retriever not initialised"
    return _retriever


def get_catalogue_df() -> pd.DataFrame:
    assert _catalogue_df is not None, "catalogue_df not initialised"
    return _catalogue_df


def get_llm() -> Any:
    assert _llm is not None, "llm not initialised"
    return _llm


def get_config() -> dict:
    assert _config is not None, "config not initialised"
    return _config


def get_session_store() -> SessionStore:
    return _session_store


def get_agent_factory() -> Callable[..., Any]:
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever = get_retriever()
    df = get_catalogue_df()
    llm = get_llm()
    config = get_config()

    def factory(memory: ConversationMemory, streaming: bool = False) -> Any:
        return build_graph(retriever, df, llm, memory, config, streaming_mode=streaming)

    return factory
