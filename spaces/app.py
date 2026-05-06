"""Streamlit thin client for the Shopping Assistant API.

Connects to the API via WebSocket (/chat/stream) for agent turns and via
HTTP (GET /catalogue/{id}/similar) for "More like this".  No src.* imports —
all agent and retrieval logic lives behind the API.

Configuration
-------------
BACKEND_URL : str
    WebSocket base URL of the API.  Default: ws://localhost:8000
    Production example: wss://your-app.fly.dev
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

import spaces.ws_client as ws_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL: str = os.environ.get("BACKEND_URL", "ws://localhost:8000")

_HTTP_BASE = BACKEND_URL.replace("wss://", "https://").replace("ws://", "http://")

_SUGGESTIONS = [
    "Show me something for the beach",
    "Outfits for a date night",
    "Minimalist winter essentials",
    "Cosy loungewear in neutral tones",
]

_TOOL_LABELS: dict[str, str] = {
    "search": "Searching the catalogue…",
    "filter": "Applying filters…",
    "search_ooc": "Checking catalogue scope…",
    "outfit": "Building outfit…",
    "clarify": "Clarifying your request…",
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None  # filled after first session frame
    if "history" not in st.session_state:
        st.session_state.history: list[dict] = []
    if "last_filters" not in st.session_state:
        st.session_state.last_filters: dict = {}


def _reset_conversation() -> None:
    st.session_state.conversation_id = None
    st.session_state.history = []
    st.session_state.last_filters = {}


def _set_pending_query(query: str) -> None:
    st.session_state.pending_query = query


def _set_more_like_seed(article_id: str) -> None:
    st.session_state.more_like_seed_id = article_id


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _item_image_url(item: dict) -> str | None:
    iu = item.get("image_url")
    if not iu:
        return None
    if iu.startswith(("http://", "https://")):
        return iu
    # Relative path served via the API's /images static mount
    return f"{_HTTP_BASE.rstrip('/')}/{iu.lstrip('/')}"


def _more_like_this(seed_id: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Call GET /catalogue/{id}/similar and return (seed_display_name, items)."""
    try:
        resp = httpx.get(
            f"{_HTTP_BASE}/catalogue/{seed_id}/similar",
            params={"k": top_k},
            timeout=10.0,
        )
        resp.raise_for_status()
        items = resp.json()
        # Fetch seed display name for the response header
        seed_resp = httpx.get(f"{_HTTP_BASE}/catalogue/{seed_id}", timeout=5.0)
        seed_name = seed_resp.json().get("display_name", seed_id) if seed_resp.is_success else seed_id
        return seed_name, items
    except Exception as exc:
        st.error(f"Could not load similar items: {exc}")
        return seed_id, []


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_card(col, item: dict, turn_index: int = 0) -> None:
    with col:
        img_url = _item_image_url(item)
        if img_url:
            try:
                st.image(img_url, width=150)
            except Exception:
                pass  # 404 or connection error — skip image gracefully

        st.markdown(f"**{item.get('display_name', item.get('prod_name', ''))}**")
        meta_parts = [
            item.get("colour", ""),
            item.get("product_type", ""),
            item.get("department", ""),
        ]
        meta = " · ".join(p for p in meta_parts if p)
        if meta:
            st.caption(meta)

        desc = item.get("detail_desc", "")
        if desc:
            with st.expander("Details", expanded=False):
                st.write(desc[:300] + ("…" if len(desc) > 300 else ""))

        aid = item.get("article_id", "")
        col_a, col_b = st.columns(2)
        col_a.button(
            "🔍 More like this",
            key=f"more_{aid}_{turn_index}",
            on_click=_set_more_like_seed,
            args=(aid,),
            use_container_width=True,
        )
        col_b.button(
            "✨ Style this",
            key=f"outfit_{aid}_{turn_index}",
            on_click=_set_pending_query,
            args=(f"build an outfit around {item.get('display_name', aid)}",),
            use_container_width=True,
        )


def _show_items(items: list[dict], turn_index: int = 0) -> None:
    if not items:
        return
    st.markdown("---")
    show = items[:5]
    n_cols = min(len(show), 3)
    cols = st.columns(n_cols)
    for i, item in enumerate(show):
        _render_card(cols[i % n_cols], item, turn_index)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Shopping Assistant",
    page_icon="🛍️",
    layout="centered",
)

_init_session()

with st.sidebar:
    st.markdown("## 🛍️ Shopping Assistant")
    st.markdown(
        "Ask natural-language questions about the H&M fashion catalogue. "
        "The assistant searches, compares, and filters items for you."
    )
    st.divider()
    st.info("🤖 Groq Llama 3.1 8B — 1–2 s latency")
    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        _reset_conversation()
        st.rerun()
    st.divider()
    st.caption("**System info**")
    st.caption(f"Backend: `{_HTTP_BASE}`")
    if st.session_state.conversation_id:
        st.caption(f"Session: `{st.session_state.conversation_id[:8]}…`")

st.title("🛍️ Agentic Shopping Assistant")

# Prompt chips — shown only on fresh conversations
if not st.session_state.history:
    st.markdown("**Try asking:**")
    chip_cols = st.columns(len(_SUGGESTIONS))
    for col, prompt in zip(chip_cols, _SUGGESTIONS):
        col.button(
            prompt,
            on_click=_set_pending_query,
            args=(prompt,),
            use_container_width=True,
        )

# Chat history
for _turn_i, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("items"):
            _show_items(msg["items"], turn_index=_turn_i)

# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

_pending = st.session_state.pop("pending_query", None)
_seed_id = st.session_state.pop("more_like_seed_id", None)
_typed = st.chat_input("Ask about anything in the store…")

# ---------------------------------------------------------------------------
# "More like this" — calls GET /catalogue/{id}/similar (no agent needed)
# ---------------------------------------------------------------------------

if _seed_id:
    seed_name, similar_items = _more_like_this(_seed_id)
    user_text = f"More like {seed_name}"
    response_text = f"Here are items similar to **{seed_name}**:"

    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        st.markdown(response_text)
        _show_items(similar_items, turn_index=len(st.session_state.history))

    st.session_state.history.append({"role": "user", "content": user_text})
    st.session_state.history.append({
        "role": "assistant",
        "content": response_text,
        "items": list(similar_items),
    })

# ---------------------------------------------------------------------------
# Agent turn — streams frames from /chat/stream
# ---------------------------------------------------------------------------

user_input: str | None = None if _seed_id else (_pending or _typed)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_ph = st.empty()
        items_container = st.container()   # items rendered here when frame arrives
        text_ph = st.empty()               # accumulates streaming tokens

        status_ph.caption("🔎 Connecting…")

        routing_decision: dict = {}
        items_list: list[dict] = []
        accumulated_text = ""
        final_state: dict = {}
        had_error = False

        for frame in ws_client.iter_frames(
            BACKEND_URL,
            st.session_state.conversation_id,
            user_input,
        ):
            ft = frame.get("type")

            if ft == "session":
                st.session_state.conversation_id = frame["conversation_id"]

            elif ft == "routing":
                routing_decision = frame.get("decision", {})

            elif ft == "tool_start":
                tool = frame.get("tool", "")
                label = _TOOL_LABELS.get(tool, f"Using {tool}…")
                status_ph.caption(f"🔎 {label}")

            elif ft == "items":
                status_ph.empty()
                items_list = frame.get("items", [])
                with items_container:
                    _show_items(items_list, turn_index=len(st.session_state.history))

            elif ft == "token":
                accumulated_text += frame.get("text", "")
                text_ph.markdown(accumulated_text + " ▌")

            elif ft == "done":
                final_state = frame.get("final_state", {})
                status_ph.empty()
                text_ph.markdown(accumulated_text)
                st.session_state.last_filters = final_state.get("filters", {})
                break

            elif ft == "cancelled":
                status_ph.empty()
                text_ph.markdown("*Turn cancelled.*")
                accumulated_text = "*Turn cancelled.*"
                break

            elif ft == "error":
                status_ph.empty()
                msg = frame.get("message", "Unknown error")
                st.error(f"Error from API: {msg}")
                had_error = True
                break

        # Routing debug expander (collapsed by default)
        if routing_decision and not had_error:
            with st.expander("How was this routed?", expanded=False):
                st.markdown(f"**Action:** `{routing_decision.get('action', '?')}`")
                query = routing_decision.get("query")
                if query:
                    st.markdown(f"**Query:** {query}")

    if not had_error:
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({
            "role": "assistant",
            "content": accumulated_text,
            "items": items_list if final_state.get("new_items_this_turn") else [],
        })
