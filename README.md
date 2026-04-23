# Agentic Shopping Assistant

> A multi-turn conversational shopping assistant over a fashion catalogue. Combines hybrid retrieval (dense + BM25 with Reciprocal Rank Fusion) with a LangGraph agent that orchestrates search, compare, and filter tools — all powered by a local Llama 3.1 8B.

🔗 **[Live Demo](https://huggingface.co/spaces/gaurav-gandhi-2411/agentic-shopping-assistant)** · 📸 **[Demo](#demo)**

---

## Motivation

Faceted search is efficient but unnatural. Customers actually ask things like *"show me something for a beach wedding, not too formal"* — a query that needs semantic understanding, attribute filtering, and conversation memory. This project explores a practical agentic approach to that problem using a real fashion dataset.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Agent                    │
│                                                     │
│  ┌──────────┐     ┌────────┐  query+filters         │
│  │  router  │────▶│ search │──────────────────────▶ HybridRetriever
│  │  (LLM)   │     └────────┘                        │ FAISS (dense, MiniLM)
│  │          │     ┌─────────┐                       │  ⊕
│  │          │────▶│ compare │  article_ids          │ BM25Okapi (sparse)
│  │          │     └─────────┘                       │  ↓ Reciprocal Rank Fusion
│  │          │     ┌────────┐                        │ top-K results
│  │          │────▶│ filter │  update facets         │
│  │          │     └────────┘                        │
│  │          │     ┌─────────┐                       │
│  │          │────▶│ clarify │──▶ END (ask user)     │
│  │          │     └─────────┘                       │
│  │          │     ┌─────────┐                       │
│  │          │────▶│ respond │──▶ LLM synthesis      │
│  └──────────┘     └─────────┘         │             │
│        ▲               │              ▼             │
│        └───────────────┘       Streamed tokens      │
│    (loop until respond/clarify)  (SSE → Streamlit)  │
└─────────────────────────────────────────────────────┘
         │
    ConversationMemory
    (6-turn window + LLM summary of older turns)
```

### Design decisions

| Decision | Rationale |
|---|---|
| **Hybrid retrieval (dense + sparse + RRF)** | Fashion queries mix semantic intent ("comfy summer vibe") with lexical signals ("linen"). Neither retriever dominates; RRF fusion is parameter-free and robust. |
| **Explicit LangGraph state machine** | Deterministic routing, inspectable state, and a hard iteration cap — avoids runaway agent loops. |
| **Filter validation at the node level** | The 8B model invents filter values (e.g. `colour: "Lightweight"`) that silently zero out searches. Nodes reject values absent from the actual catalogue. |
| **Ollama locally / Groq on HF Spaces** | Local dev is free and private. Groq's free tier provides fast inference for the hosted demo without a GPU. |
| **Streaming via SSE** | Users see tokens within ~200 ms of the LLM starting. The routing phase (search/compare/filter) runs first; the final LLM response streams token-by-token. |
| **Single-process Streamlit on Spaces** | FastAPI + Streamlit as two processes is fragile on free-tier Spaces. The hosted demo folds the agent directly into the Streamlit app. |

---

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 20,000-article sample (text only; no image embeddings in this project). 107 product types, 50 colour groups.

---

## Tech stack

`LangGraph` · `LangChain Core` · `Ollama` · `Groq` · `FAISS` · `rank-bm25` · `sentence-transformers` · `FastAPI` (SSE) · `Streamlit` · `HuggingFace Hub`

---

## Running locally

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) with `llama3.1:8b` pulled.

```bash
# 1. Clone and create venv
git clone https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant
cd agentic-shopping-assistant
python -m venv .venv && .venv\Scripts\activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the H&M dataset
# Download articles.csv from Kaggle and place at data/hm/articles.csv
# (or symlink to an existing copy)

# 4. Build retrieval indices (one-time, ~2 min on CPU)
python scripts/01_build_retrieval.py

# 5. Start the API
uvicorn api.main:app --host 127.0.0.1 --port 8080

# 6. Start the UI (new terminal)
streamlit run app/streamlit_chat.py
```

Open `http://localhost:8501`.

**Run tests:**
```bash
pytest tests/
```

**Run the multi-turn smoke test:**
```bash
python scripts/02_smoke_test.py
```

---

## Retrieval sanity check

Hybrid retrieval results on 4 representative queries (top-1 shown):

| Query | Top result | Score |
|---|---|---|
| `"lightweight black summer jacket"` | Go light jacket (Black Jacket) | 0.0315 |
| `"comfy hoodie for winter"` | COZY POP OVER HOODIE (Grey Hoodie) | 0.0263 |
| `"elegant red dress"` | Rosa dress (Red Dress) | 0.0313 |
| `"kids blue t-shirt"` | SIBLINGS tee (Dark Blue T-shirt) | 0.0311 |

---

## Smoke test — multi-turn conversation

4-turn conversation showing search, semantic retrieval, compare, and filter:

| Turn | Query | Agent tools | Response summary |
|---|---|---|---|
| 1 | *"show me some black jackets"* | `search ×2` | CS Whoa Jacket, Jacket Slim, Jennifer fancy Blazer |
| 2 | *"something for summer, light and breathable"* | `search`, `filter_rejected`, `search` | Summer Selma blouse, SECTION MNY SPEED SUMMER SKIRT |
| 3 | *"compare the first two you showed me"* | `compare ×1` | Side-by-side: Red strap dress vs Black sleeveless dress |
| 4 | *"anything in blue instead?"* | `filter`, `search ×3` | Cat Tee, Suzie linnen Tee, RAVEN pocket tee (all blue) |

---

## Demo

> _Screenshot / GIF to be added after recording._

<!-- Add demo.gif here: ![Demo](assets/demo.gif) -->

---

## What I learned

- **JSON parsers need brace-depth awareness.** The router prompt spec includes `"filters": {}` as part of the action JSON. A simple regex `[^{}]*` matched the inner empty object rather than the outer one, causing every filter action to fall back to search. The fix — walking character-by-character and tracking brace depth — is one of those bugs that's obvious in retrospect but only surfaces with real LLM output.

- **Small models invent plausible-but-wrong attribute values.** `llama3.1:8b` would apply `colour_group_name: "Lightweight"`, a value that reads naturally but doesn't exist in the catalogue. Every subsequent filtered search returned zero results silently. The fix was to build a set of valid facet values at graph-construction time and reject invented values at the filter node — defence-in-depth rather than trusting the LLM to stay in-distribution.

- **State-machine rules in the prompt beat abstract instructions.** "Use respond when you have enough context" was ignored half the time; "if `last_action` is `search` and `items_retrieved > 0`, output respond" was followed reliably. Exposing concrete state (last action, item count) and writing if-then rules rather than natural-language principles made the 8B model converge in 2–3 tool calls instead of 6.

- **SSE architecture on Windows has a non-obvious footgun.** The Ollama Python client's `stream=False` path triggers an internal runner crash on the Windows build (HTTP 500). There's no error in the Ollama logs — it just silently terminates. The fix is to always use the streaming path and collect chunks: `"".join(client.chat_stream(...))`. Equally reliable, and now the streaming and non-streaming code paths are unified.

- **Free-tier deployment forces architectural simplicity.** The local design (FastAPI + Streamlit, two processes, SSE) is clean for development but fragile on a single-container free Space. Folding the agent directly into the Streamlit app removes the HTTP layer, halves the moving parts, and actually makes the code easier to read — a reminder that the deployment target should shape the design early.

---

## License

MIT
