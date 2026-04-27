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
from src.retrieval.hybrid_search import HybridRetriever, normalize_prod_name
from src.llm.client import get_llm_client
from src.memory.conversation import ConversationMemory
from src.agents.graph import build_graph, _detect_ooc

# ---------------------------------------------------------------------------
# Config — override provider to groq for Spaces
# ---------------------------------------------------------------------------
_CONFIG_PATH = str(_REPO_ROOT / "config.yaml")
_DATA_DIR = _REPO_ROOT / "data" / "processed"

config = load_config(_CONFIG_PATH)
config["llm"]["provider"] = os.environ.get("LLM_PROVIDER", "groq")

LLM_PROVIDER = config["llm"]["provider"]
_llm_model_key = "model" if LLM_PROVIDER == "ollama" else f"{LLM_PROVIDER}_model"
LLM_MODEL = config["llm"].get(_llm_model_key, "llama-3.1-8b-instant")
LLM_LABEL = f"{LLM_PROVIDER.capitalize()} · {LLM_MODEL}"

_DB_MODEL_PATH = _REPO_ROOT / config.get("router", {}).get("distilbert_model_path", "models/distilbert_router")
_DB_AVAILABLE = _DB_MODEL_PATH.exists()

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


@st.cache_resource(show_spinner="Loading DistilBERT router...")
def _load_distilbert_backend():
    from src.agents.router import DistilBERTRouterBackend
    _, df = _load_retrieval()
    return DistilBERTRouterBackend(_DB_MODEL_PATH, df)


# ---------------------------------------------------------------------------
# Per-session agent (memory is stateful — must not be shared across users)
# ---------------------------------------------------------------------------

def _init_session():
    # Rebuild agent if the router provider changed mid-session.
    selected_router = st.session_state.get("router_provider", "llm")
    if st.session_state.get("_active_router") != selected_router:
        st.session_state.pop("agent", None)
        st.session_state["_active_router"] = selected_router
        # Also reset conversation so history matches the new router's behaviour.
        st.session_state.pop("conversation_id", None)
        st.session_state.pop("history", None)
        st.session_state.pop("conv_state", None)

    if "agent" not in st.session_state:
        retriever, df = _load_retrieval()
        llm = get_llm_client(config)
        memory = ConversationMemory(llm, config)

        router_backend = None
        if selected_router == "distilbert" and _DB_AVAILABLE:
            router_backend = _load_distilbert_backend()

        st.session_state.agent = build_graph(
            retriever, df, llm, memory, config,
            streaming_mode=True, router_backend=router_backend,
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


def _set_more_like_seed(article_id: str) -> None:
    st.session_state.more_like_seed_id = article_id


_NEUTRAL_COLOURS = frozenset({
    "black", "white", "grey", "dark grey", "light grey",
    "beige", "light beige", "dark beige",
})

# Tonal-style queries trigger CIELAB post-filter: (palette, max_delta_e)
TONAL_STYLE_PALETTES: dict[str, tuple[list[str] | None, float]] = {
    "minimalist": (
        ["Black", "White", "Off White", "Grey", "Dark Grey", "Light Grey", "Beige", "Light Beige"],
        25,
    ),
    "neutral": (
        ["Black", "White", "Off White", "Grey", "Dark Grey", "Light Grey", "Beige", "Light Beige"],
        25,
    ),
    "earth tone": (["Beige", "Light Beige", "Dark Beige", "Greyish Beige", "Yellowish Brown"], 25),
    "earth tones": (["Beige", "Light Beige", "Dark Beige", "Greyish Beige", "Yellowish Brown"], 25),
    "monochrome": (None, 15),
}


def _get_tonal_spec(query: str) -> tuple[list[str] | None, float] | None:
    """Return (palette, max_delta_e) if query contains a tonal keyword, else None."""
    q_lower = query.lower()
    for keyword, spec in TONAL_STYLE_PALETTES.items():
        if keyword in q_lower:
            return spec
    return None


def _filter_by_tonal_compatibility(
    items: list[dict],
    palette: list[str],
    max_delta_e: float,
) -> list[dict]:
    from src.utils.colour_lab import COLOUR_TO_LAB, delta_e_2000
    palette_labs = [COLOUR_TO_LAB[c] for c in palette if c in COLOUR_TO_LAB]
    if not palette_labs:
        return items
    result = []
    for item in items:
        item_lab = COLOUR_TO_LAB.get(item.get("colour", ""))
        if item_lab is None:
            result.append(item)  # unknown colour: include
            continue
        if min(delta_e_2000(item_lab, p_lab) for p_lab in palette_labs) <= max_delta_e:
            result.append(item)
    return result


def _tonal_backfill(
    filtered: list[dict],
    user_query: str,
    palette: list[str],
    max_delta_e: float,
    target: int = 5,
) -> list[dict]:
    """Re-retrieve up to `target` tonal items if filtered pool is too small."""
    from src.utils.colour_lab import COLOUR_TO_LAB, delta_e_2000
    retriever_obj, _ = _load_retrieval()
    candidates = retriever_obj.search(user_query, top_k=25)
    palette_labs = [COLOUR_TO_LAB[c] for c in palette if c in COLOUR_TO_LAB]
    seen_ids = {it["article_id"] for it in filtered}
    seen_keys = {
        (normalize_prod_name(it.get("prod_name", it.get("display_name", ""))), it.get("colour", "").lower())
        for it in filtered
    }
    for cand in candidates:
        if len(filtered) >= target:
            break
        if cand["article_id"] in seen_ids:
            continue
        key = (normalize_prod_name(cand.get("prod_name", cand.get("display_name", ""))), cand.get("colour", "").lower())
        if key in seen_keys:
            continue
        cand_lab = COLOUR_TO_LAB.get(cand.get("colour", ""))
        if cand_lab is None or (palette_labs and min(delta_e_2000(cand_lab, p) for p in palette_labs) <= max_delta_e):
            filtered.append(cand)
            seen_ids.add(cand["article_id"])
            seen_keys.add(key)
    return filtered
# Dark tones that work together; excludes dark green/red (too colourful to count as "dark neutral")
_COMPATIBLE_DARKS = frozenset({"black", "dark blue", "dark grey"})
_COMPATIBLE_LIGHTS = frozenset({"white", "light grey", "light beige", "beige", "light pink", "light blue"})


def _colour_compatible(seed_colour: str, item_colour: str) -> bool:
    """True if item_colour is the same as seed or palette-compatible.

    A neutral ITEM goes with any seed, but a neutral SEED does not mean we accept
    any item — we still require same colour, neutral item, or same-family dark/light.
    """
    s, i = seed_colour.lower(), item_colour.lower()
    if s == i:
        return True
    if i in _NEUTRAL_COLOURS:  # neutral item pairs with any seed
        return True
    if s in _COMPATIBLE_DARKS and i in _COMPATIBLE_DARKS:
        return True
    if s in _COMPATIBLE_LIGHTS and i in _COMPATIBLE_LIGHTS:
        return True
    return False


def _more_like_this_items(seed_id: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Return (seed_name, similar_items) using stored FAISS embedding with colour filter."""
    retriever, _ = _load_retrieval()
    cat = retriever.catalogue_df  # indexed by article_id

    if seed_id not in cat.index:
        return seed_id, []
    seed_row = cat.loc[seed_id]
    seed_facets = seed_row["facets"] if isinstance(seed_row["facets"], dict) else {}
    seed_name = seed_row["display_name"]
    seed_colour = seed_facets.get("colour_group_name", "").lower()

    # Fetch generously so colour filtering still yields top_k results
    hits = retriever.dense.search_by_id(seed_id, top_k=top_k * 4)

    seen_prod_colour: set[tuple[str, str]] = set()

    def _make_item(aid: str, score: float) -> dict | None:
        if aid == seed_id or aid not in cat.index:
            return None
        row = cat.loc[aid]
        facets = row["facets"] if isinstance(row["facets"], dict) else {}
        return {
            "article_id": aid,
            "prod_name": row.get("prod_name", ""),
            "display_name": row["display_name"],
            "colour": facets.get("colour_group_name", ""),
            "product_type": facets.get("product_type_name", ""),
            "department": facets.get("department_name", ""),
            "detail_desc": row.get("detail_desc", ""),
            "image_url": row.get("image_url", ""),
            "score": score,
        }

    def _is_duplicate(it: dict) -> bool:
        key = (normalize_prod_name(it.get("prod_name", it["display_name"])), it["colour"].lower())
        if key in seen_prod_colour:
            return True
        seen_prod_colour.add(key)
        return False

    # Pass 1: same colour or palette-compatible (dedup by prod_name+colour)
    items: list[dict] = []
    all_items: list[dict] = []
    for aid, score in hits:
        it = _make_item(aid, score)
        if it is None:
            continue
        all_items.append(it)
        if not _is_duplicate(it) and seed_colour and _colour_compatible(seed_colour, it["colour"].lower()):
            items.append(it)
        if len(items) >= top_k:
            break

    # Pass 2: backfill with neutral-coloured items only (palette-safe, deduped)
    if len(items) < top_k:
        seen_ids = {it["article_id"] for it in items}
        for it in all_items:
            if it["article_id"] in seen_ids:
                continue
            if it["colour"].lower() in _NEUTRAL_COLOURS and not _is_duplicate(it):
                items.append(it)
                seen_ids.add(it["article_id"])
            if len(items) >= top_k:
                break

    # Pass 3: if still short, fill with any remaining by similarity order (deduped)
    if len(items) < top_k:
        seen_ids = {it["article_id"] for it in items}
        for it in all_items:
            if it["article_id"] not in seen_ids and not _is_duplicate(it):
                items.append(it)
                seen_ids.add(it["article_id"])
            if len(items) >= top_k:
                break

    return seed_name, items[:top_k]


def _render_card(col, item: dict, turn_index: int = 0) -> None:
    with col:
        role = item.get("_role", "")
        if role == "seed":
            st.caption("**Starting item**")
        elif role == "complement":
            st.caption("Pair with")
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
            on_click=_set_more_like_seed,
            args=(aid,),
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

    # Router selector
    st.markdown("**Router**")
    _router_options = ["LLM (Groq)", "Classifier (DistilBERT)"]
    if not _DB_AVAILABLE:
        _router_options = ["LLM (Groq)"]
    _router_idx = 1 if st.session_state.get("router_provider") == "distilbert" and _DB_AVAILABLE else 0
    _router_choice = st.selectbox(
        "Routing engine",
        _router_options,
        index=_router_idx,
        label_visibility="collapsed",
    )
    _new_router = "distilbert" if _router_choice == "Classifier (DistilBERT)" else "llm"
    if _new_router != st.session_state.get("router_provider", "llm"):
        st.session_state.router_provider = _new_router
        st.rerun()

    if st.session_state.get("router_provider") == "distilbert":
        st.success("⚡ Local classifier — ~31ms latency")
    else:
        st.info("🤖 Groq llama-3.1-8b-instant — 1–2s latency")

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
    st.caption(f"LLM: `{LLM_LABEL}`")
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
_seed_id = st.session_state.pop("more_like_seed_id", None)
_typed = st.chat_input("Ask about anything in the store...")

# ---------------------------------------------------------------------------
# "More like this" — pure FAISS vector similarity, bypasses the agent entirely
# ---------------------------------------------------------------------------
if _seed_id:
    seed_name, similar_items = _more_like_this_items(_seed_id)
    user_text = f"More like {seed_name}"
    response_text = f"Here are items similar to **{seed_name}**:"

    conv = st.session_state.conv_state
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if similar_items:
            _show_items(similar_items, turn_index=len(st.session_state.history))

    new_messages = conv["messages"] + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": response_text},
    ]
    st.session_state.conv_state = {
        "messages": new_messages,
        "filters": conv["filters"],
        "retrieved_items": similar_items,
    }
    st.session_state.history.append({"role": "user", "content": user_text})
    st.session_state.history.append({
        "role": "assistant",
        "content": response_text,
        "items": list(similar_items),
    })

user_input = None if _seed_id else (_pending or _typed)

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
        "out_of_catalogue": False,
    }

    with st.chat_message("assistant"):
        status_ph = st.empty()
        status_ph.caption("🔎 Searching the catalogue...")

        # Phase 1 — run the graph (routing + tool calls, no LLM respond yet)
        try:
            result = st.session_state.agent.invoke(initial_state)
        except Exception as exc:
            status_ph.empty()
            exc_str = str(exc)
            if "rate_limit_exceeded" in exc_str or "tokens per day" in exc_str.lower() or "RateLimitError" in type(exc).__name__:
                st.warning("⏳ The AI service is temporarily unavailable due to high usage. Please try again in a few hours.")
            elif "ConnectionError" in type(exc).__name__ or "Timeout" in exc_str:
                st.warning("🔌 Connection issue with the AI service. Please retry in a moment.")
            else:
                st.error(f"Something went wrong: {type(exc).__name__}. Please try again or report if this persists.")
            print(f"[agent] invoke error: {exc!r} query={user_input!r}")
            st.stop()

        plan = json.loads(result.get("current_plan") or "{}")
        action = plan.get("action", "")
        items = result.get("retrieved_items", [])
        new_items = result.get("new_items_this_turn", False)

        # Post-reranker tonal filter — only for queries with tonal style keywords
        if new_items and items:
            tonal_spec = _get_tonal_spec(user_input)
            if tonal_spec is not None:
                palette, max_de = tonal_spec
                if palette is not None:
                    filtered = _filter_by_tonal_compatibility(items, palette, max_de)
                    if len(filtered) < 3:
                        filtered = _tonal_backfill(filtered, user_input, palette, max_de)
                    if len(filtered) >= 3:
                        items = filtered[:5]

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
