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

_retriever: Any = None          # HybridRetriever
_catalogue_df: pd.DataFrame | None = None
_llm: Any = None                # LLMClient
_config: dict | None = None
_session_store: SessionStore = InMemorySessionStore()


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
# Getters — used directly in route handlers and in tests via override
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
    """Return a factory that builds a fresh compiled agent for each conversation.

    The factory captures the current singletons, so it must be called after
    lifespan startup.  Do NOT cache the returned agent — ConversationMemory is
    stateful and must not be shared across sessions.
    """
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever = get_retriever()
    df = get_catalogue_df()
    llm = get_llm()
    config = get_config()

    def factory(memory: ConversationMemory, streaming: bool = False) -> Any:
        return build_graph(retriever, df, llm, memory, config, streaming_mode=streaming)

    return factory
