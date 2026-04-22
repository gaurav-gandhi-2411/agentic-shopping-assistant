import json
import sys
import uuid
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.catalogue.loader import load_config

config = load_config()
API_URL = f"http://{config['api']['host']}:{config['api']['port']}"
LLM_MODEL = config["llm"]["model"]
CORPUS_SIZE = config["catalogue"]["sample_num_items"]

st.set_page_config(
    page_title="Agentic Shopping Assistant",
    page_icon="🛍️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []   # {role, content, items}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _show_items(items: list[dict]) -> None:
    """Display retrieved items as compact cards below an assistant message."""
    if not items:
        return
    st.markdown("---")
    n = min(len(items), 5)
    cols = st.columns(n)
    for col, item in zip(cols, items[:n]):
        with col:
            st.markdown(f"**{item['display_name']}**")
            st.caption(f"{item.get('colour', '')} · {item.get('product_type', '')}")
            desc = item.get("detail_desc", "")
            if desc:
                with st.expander("Details", expanded=False):
                    st.write(desc[:200] + ("…" if len(desc) > 200 else ""))


def _api_ok() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
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
        st.rerun()

    st.divider()

    # System info footer
    st.caption("**System info**")
    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption("Retrieval: Hybrid (dense + BM25 with RRF)")
    st.caption(f"Corpus: {CORPUS_SIZE:,} items")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🛍️ Agentic Shopping Assistant")

if not _api_ok():
    st.error(
        f"Cannot reach API at `{API_URL}`. "
        "Start it with:  \n```\nuvicorn api.main:app --host 127.0.0.1 --port 8080\n```"
    )
    st.stop()

# Render chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("items"):
            _show_items(msg["items"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if user_input := st.chat_input("Ask about anything in the store…"):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    items_buf: list[dict] = []

    def _stream():
        """Yields token strings; collects items/tool events as side-effects."""
        payload = {
            "message": user_input,
            "conversation_id": st.session_state.conversation_id,
        }
        try:
            with httpx.Client(timeout=180) as client:
                with client.stream(
                    "POST", f"{API_URL}/chat/stream", json=payload
                ) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines():
                        if not raw.startswith("data: "):
                            continue
                        try:
                            data = json.loads(raw[6:])
                        except json.JSONDecodeError:
                            continue

                        t = data.get("type")
                        if t == "items":
                            items_buf.extend(data.get("items", []))
                        elif t == "token":
                            yield data["content"]
                        elif t == "error":
                            yield f"\n\n_Error from agent: {data.get('message', '')}_"
                            return
                        elif t == "done":
                            return
        except httpx.HTTPError as exc:
            yield f"\n\n_HTTP error: {exc}_"
        except Exception as exc:
            yield f"\n\n_Unexpected error: {exc}_"

    with st.chat_message("assistant"):
        # Status indicator — visible while the routing phase runs (before first token)
        status_ph = st.empty()
        status_ph.caption("🔎 Searching the catalogue…")

        response_text = st.write_stream(_stream())
        status_ph.empty()

        if items_buf:
            _show_items(items_buf)

    # Persist in session history
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({
        "role": "assistant",
        "content": response_text or "",
        "items": list(items_buf),
    })
