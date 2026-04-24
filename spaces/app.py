"""
HuggingFace Spaces entry point — folded single-process Streamlit app.

The agent runs directly in the Streamlit process (no FastAPI / SSE layer).
LLM: Groq (set GROQ_API_KEY as a Space secret).
Data:  data/processed/{dense.faiss, dense_article_ids.npy,
                        bm25.pkl, bm25_article_ids.npy, catalogue.parquet,
                        images/}
       Upload once with: python spaces/upload_artifacts.py --repo <user>/<space> --space
"""

import json
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

# Local dev: app.py lives in spaces/, src/ is one level up (repo root).
# HF Space:  app.py is at container root alongside src/ (bundled at deploy time).
_SPACES_DIR = Path(__file__).parent
_REPO_ROOT = _SPACES_DIR.parent if (_SPACES_DIR.parent / "src").exists() else _SPACES_DIR
sys.path.insert(0, str(_REPO_ROOT))

from src.agents.grounding import validate_response
from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.llm.client import get_llm_client
from src.memory.conversation import ConversationMemory
from src.agents.graph import build_graph

# ---------------------------------------------------------------------------
# Config — override provider to groq for Spaces
# ---------------------------------------------------------------------------
_CONFIG_PATH = str(_REPO_ROOT / "config.yaml")
_DATA_DIR = _REPO_ROOT / "data" / "processed"

config = load_config(_CONFIG_PATH)
config["llm"]["provider"] = os.environ.get("LLM_PROVIDER", "groq")

LLM_MODEL = config["llm"].get("groq_model", "llama-3.1-8b-instant")

_SUGGESTIONS = [
    "Show me something for the beach",
    "Outfits for a date night",
    "Minimalist winter essentials",
    "Cosy loungewear in neutral tones",
]

# ---------------------------------------------------------------------------
# Heavy components — loaded once, shared across all sessions
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading retrieval indices...")
def _load_retrieval():
    import pandas as pd
    df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, df, config)
    return retriever, df


# ---------------------------------------------------------------------------
# Per-session agent (memory is stateful — must not be shared across users)
# ---------------------------------------------------------------------------

def _init_session():
    if "agent" not in st.session_state:
        retriever, df = _load_retrieval()
        llm = get_llm_client(config)
        memory = ConversationMemory(llm, config)
        st.session_state.agent = build_graph(
            retriever, df, llm, memory, config, streaming_mode=True
        )
        st.session_state.llm = llm
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history: list[dict] = []
    if "conv_state" not in st.session_state:
        st.session_state.conv_state = {
            "messages": [], "filters": {}, "retrieved_items": [],
        }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _set_pending_query(query: str) -> None:
    st.session_state.pending_query = query


def _render_card(col, item: dict, turn_index: int = 0) -> None:
    with col:
        img_path = _DATA_DIR / item["image_url"] if item.get("image_url") else None
        if img_path and img_path.exists():
            st.image(str(img_path), width=150)
        st.markdown(f"**{item['display_name']}**")
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
                st.write(desc[:300] + ("..." if len(desc) > 300 else ""))
        aid = item.get("article_id", "")
        col_a, col_b = st.columns(2)
        col_a.button(
            "🔍 More like this",
            key=f"more_like_{aid}_{turn_index}",
            on_click=_set_pending_query,
            args=(f"find more items similar to {item['display_name']}",),
            use_container_width=True,
        )
        col_b.button(
            "✨ Style this",
            key=f"outfit_{aid}_{turn_index}",
            on_click=_set_pending_query,
            args=(f"build an outfit around {item['display_name']}",),
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

# Sidebar
with st.sidebar:
    st.markdown("## 🛍️ Shopping Assistant")
    st.markdown(
        "Ask natural-language questions about the H&M fashion catalogue. "
        "The assistant searches, compares, and filters items for you."
    )
    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.history = []
        st.session_state.conv_state = {
            "messages": [], "filters": {}, "retrieved_items": [],
        }
        st.rerun()
    st.divider()
    st.caption("**System info**")
    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption("Retrieval: Hybrid (dense + BM25 with RRF)")
    _, _cat_df = _load_retrieval()
    st.caption(f"Corpus: {len(_cat_df):,} items")

st.title("🛍️ Agentic Shopping Assistant")

# Prompt chips — shown only on fresh conversation
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
# Chat input — accepts both typed queries and chip button presses
# ---------------------------------------------------------------------------

_pending = st.session_state.pop("pending_query", None)
_typed = st.chat_input("Ask about anything in the store...")
user_input = _pending or _typed

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    conv = st.session_state.conv_state
    initial_state = {
        "messages": conv["messages"] + [{"role": "user", "content": user_input}],
        "user_query": user_input,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": conv["retrieved_items"],
        "filters": conv["filters"],
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
    }

    with st.chat_message("assistant"):
        status_ph = st.empty()
        status_ph.caption("🔎 Searching the catalogue...")

        # Phase 1 — run the graph (routing + tool calls, no LLM respond yet)
        try:
            result = st.session_state.agent.invoke(initial_state)
        except Exception as exc:
            status_ph.empty()
            st.warning("Something went wrong — please try again in a moment.")
            print(f"[agent] invoke error: {exc!r} query={user_input!r}")
            st.stop()

        plan = json.loads(result.get("current_plan") or "{}")
        action = plan.get("action", "")
        items = result.get("retrieved_items", [])
        new_items = result.get("new_items_this_turn", False)

        # Phase 2 — stream the LLM response
        if action == "pending_respond":
            prompt = plan["prompt"]
            status_ph.empty()
            try:
                response_text = st.write_stream(
                    st.session_state.llm.generate_stream(prompt)
                )
            except Exception as exc:
                response_text = "I'm having trouble generating a response right now — please try again."
                st.markdown(response_text)
                print(f"[groq] stream error: {exc!r} query={user_input!r}")
            cleaned, flags = validate_response(response_text or "", items)
            if flags:
                print(f"[grounding] flags={flags} query={user_input!r}")
            response_text = cleaned
        elif action == "pending_answer":
            text = plan.get("text", "")
            status_ph.empty()
            st.markdown(text)
            response_text = text
        else:
            status_ph.empty()
            response_text = result.get("final_answer", "")
            st.markdown(response_text)

        if new_items and items:
            _show_items(items, turn_index=len(st.session_state.history))

    # Persist conversation state for next turn
    new_messages = conv["messages"] + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response_text or ""},
    ]
    st.session_state.conv_state = {
        "messages": new_messages,
        "filters": result.get("filters", conv["filters"]),
        "retrieved_items": items,
    }

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({
        "role": "assistant",
        "content": response_text or "",
        "items": list(items) if new_items else [],
    })
