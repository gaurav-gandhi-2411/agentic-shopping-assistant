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
from src.retrieval.snapshot import LEGACY_BUILD_ID, ClipEntry, IndexSnapshot, public_catalogue

if TYPE_CHECKING:
    from src.retrieval.build_manifest import BuildManifest
    from src.retrieval.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — set by main.py lifespan.
#
# Everything that depends on the loaded index (retriever, catalogue DataFrame, CLIP index, the
# compiled agent graphs, the build manifest) lives in ONE immutable IndexSnapshot held in
# ``_snapshot``. The hot-reload thread replaces that single reference (install_snapshot); route
# handlers take it once per turn (get_snapshot) and use it throughout.
# ---------------------------------------------------------------------------

_snapshot: IndexSnapshot | None = None
# Index hot-reload controller (api/main.py lifespan), None when hot reload is off.
_reloader: Any = None

# Legacy fall-backs used ONLY when no snapshot is installed (tests that inject a bare retriever
# or DataFrame with monkeypatch). _init() never sets them: holding them would pin the old
# index in memory after a swap.
_retriever: Any = None
_catalogue_df: pd.DataFrame | None = None
_llm: Any = None
_config: dict | None = None
_session_store: SessionStore = InMemorySessionStore()

# Compiled LangGraph agent graphs.  Both variants are compiled per snapshot (at startup and on
# each reload, never on the request path) and live in the IndexSnapshot; the two names below are
# the same legacy test fall-backs as above.
# ConversationMemory is passed through AgentState._memory at invoke time.
_agent_sync: Any = None      # streaming_mode=False
_agent_streaming: Any = None  # streaming_mode=True
_db_engine: Any = None

# ---------------------------------------------------------------------------
# Auth — get_current_user_id lives in api.auth (Phase 2 prompt 2).
# DEV_USER_ID is kept here for tests and for JWT_VERIFICATION_DISABLED mode.
# Production code uses api.auth.get_current_user_id as a FastAPI Depends.
# ---------------------------------------------------------------------------

DEV_USER_ID: str = "00000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# Setters called by lifespan
# ---------------------------------------------------------------------------

def make_snapshot(
    retriever: Any,
    catalogue_df: pd.DataFrame,
    llm: Any,
    config: dict,
    *,
    manifest: BuildManifest | None = None,
    clip: ClipEntry | None = None,
    source: str = "flat",
) -> IndexSnapshot:
    """Build an IndexSnapshot: public view of the catalogue + both agent graphs compiled around
    THIS retriever and DataFrame (``build_graph`` captures them in closures, so the graphs must
    be replaced together with the index). Compiled before any swap, never on the request path.
    """
    from src.agents.graph import build_graph

    df = public_catalogue(catalogue_df)
    logger.info("Compiling agent graphs (sync + streaming)…")
    agent_sync = build_graph(retriever, df, llm, config, streaming_mode=False)
    agent_streaming = build_graph(retriever, df, llm, config, streaming_mode=True)
    logger.info("Agent graphs compiled")
    return IndexSnapshot(
        retriever=retriever,
        catalogue_df=df,
        manifest=manifest,
        build_id=manifest.build_id if manifest is not None else LEGACY_BUILD_ID,
        clip=clip,
        agent_sync=agent_sync,
        agent_streaming=agent_streaming,
        source=source,
    )


def install_snapshot(snapshot: IndexSnapshot) -> IndexSnapshot | None:
    """Publish ``snapshot`` as the served index; return the previous one (or None).

    The swap is ONE reference assignment (atomic under the GIL): a request sees either the whole
    old snapshot or the whole new one. The caller must drop its reference to the returned old
    snapshot so it can be freed once in-flight turns finish.
    """
    global _snapshot
    old = _snapshot
    _snapshot = snapshot
    return old


def _init(
    retriever: Any,
    catalogue_df: pd.DataFrame,
    llm: Any,
    config: dict,
    session_store: SessionStore | None = None,
    db_engine: Any = None,
    *,
    manifest: BuildManifest | None = None,
    clip: ClipEntry | None = None,
    source: str = "flat",
) -> None:
    global _llm, _config, _session_store, _db_engine
    _llm = llm
    _config = config
    if session_store is not None:
        _session_store = session_store
    if db_engine is not None:
        _db_engine = db_engine

    # Compile both graph variants once per snapshot — never on the request path.
    install_snapshot(
        make_snapshot(
            retriever, catalogue_df, llm, config, manifest=manifest, clip=clip, source=source
        )
    )


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------

def get_snapshot() -> IndexSnapshot:
    """The served index snapshot. Call ONCE per turn/request and use the result throughout."""
    snap = _snapshot
    assert snap is not None, "index snapshot not initialised"
    return snap


def get_snapshot_or_none() -> IndexSnapshot | None:
    """Like get_snapshot() but None before startup finished (readiness probes)."""
    return _snapshot


def get_reloader() -> Any:
    """The index hot-reload controller, or None when hot reload is off."""
    return _reloader


def get_retriever() -> "HybridRetriever":
    snap = _snapshot
    if snap is not None:
        return snap.retriever
    assert _retriever is not None, "retriever not initialised"
    return _retriever


def get_catalogue_df() -> pd.DataFrame:
    snap = _snapshot
    if snap is not None:
        return snap.catalogue_df
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


def get_db_engine() -> Any:
    return _db_engine


def get_agent_factory() -> Callable[..., Any]:
    """Return a factory that maps (memory, streaming) → compiled agent singleton.

    The memory argument is accepted for API compatibility but is ignored here —
    ConversationMemory is injected into AgentState._memory at invoke time by
    _build_invoke_state() in the route handlers.
    """
    from src.memory.conversation import ConversationMemory  # noqa: F401 — kept for type hint

    # Bind the snapshot NOW (once per turn: chat.py calls get_agent_factory() once per turn), so
    # the agent the turn runs on belongs to the same snapshot as everything else it reads even
    # if a reload swaps the served snapshot while the turn is in flight.
    snap = _snapshot

    def factory(memory: Any, streaming: bool = False) -> Any:
        if snap is not None:
            return snap.agent_streaming if streaming else snap.agent_sync
        return _agent_streaming if streaming else _agent_sync

    return factory
