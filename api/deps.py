"""Dependency-injection providers for FastAPI routes.

Singletons (retriever, LLM, catalogue, compiled agent graphs) are loaded once
during the lifespan context in main.py and stored here as module-level
variables.  Route handlers access them via FastAPI's Depends() or by calling
the getters directly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from api.session import InMemorySessionStore, SessionStore

if TYPE_CHECKING:
    from src.retrieval.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — set by main.py lifespan, never mutated after that.
# ---------------------------------------------------------------------------

_retriever: Any = None
_catalogue_df: pd.DataFrame | None = None
_llm: Any = None
_config: dict | None = None
_session_store: SessionStore = InMemorySessionStore()

# Compiled LangGraph agent graphs.  Both variants are compiled once at startup
# so build_graph() / builder.compile() are never called on the hot path.
# ConversationMemory is passed through AgentState._memory at invoke time.
_agent_sync: Any = None      # streaming_mode=False
_agent_streaming: Any = None  # streaming_mode=True

# ---------------------------------------------------------------------------
# Auth — get_current_user_id lives in api.auth (Phase 2 prompt 2).
# DEV_USER_ID is kept here for tests and for JWT_VERIFICATION_DISABLED mode.
# Production code uses api.auth.get_current_user_id as a FastAPI Depends.
# ---------------------------------------------------------------------------

DEV_USER_ID: str = "00000000-0000-0000-0000-000000000001"


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
    global _agent_sync, _agent_streaming
    _retriever = retriever
    _catalogue_df = catalogue_df
    _llm = llm
    _config = config
    if session_store is not None:
        _session_store = session_store

    # Compile both graph variants once — this is the only call to build_graph()
    # and builder.compile() in the process lifetime.
    from src.agents.graph import build_graph
    logger.info("Compiling agent graphs (sync + streaming)…")
    _agent_sync = build_graph(retriever, catalogue_df, llm, config, streaming_mode=False)
    _agent_streaming = build_graph(retriever, catalogue_df, llm, config, streaming_mode=True)
    logger.info("Agent graphs compiled")


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
    """Return a factory that maps (memory, streaming) → compiled agent singleton.

    The memory argument is accepted for API compatibility but is ignored here —
    ConversationMemory is injected into AgentState._memory at invoke time by
    _build_invoke_state() in the route handlers.
    """
    from src.memory.conversation import ConversationMemory  # noqa: F401 — kept for type hint

    def factory(memory: Any, streaming: bool = False) -> Any:
        return _agent_streaming if streaming else _agent_sync

    return factory
