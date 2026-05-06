# Shopping Assistant API

FastAPI backend for the Agentic Shopping Assistant.  Exposes three route groups:

| Prefix | Description |
|---|---|
| `GET /healthz` | Liveness probe (always 200) |
| `GET /readyz` | Readiness probe — 503 until retriever + LLM are loaded |
| `POST /chat` | Non-streaming chat (full agent round-trip) |
| `WS /chat/stream` | Streaming chat over WebSocket |
| `GET /catalogue/{id}` | Item metadata |
| `GET /catalogue/{id}/similar?k=5` | Top-k FAISS nearest-neighbour items |

---

## Running locally

```bash
# From repo root
pip install -r requirements.txt

uvicorn api.main:app --reload --port 8000
```

The lifespan loads `data/processed/catalogue.parquet`, `dense.faiss`, and `bm25.pkl` on startup (~5–10 s the first time).

---

## Environment variables

### Required in production

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for the Groq LLM provider |

### Optional / observability

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | *(from config.yaml)* | Override LLM provider at runtime (`groq`, `ollama`, `openrouter`, `gemini`) |
| `LANGSMITH_API_KEY` | *(unset — tracing disabled)* | LangSmith API key; set to enable LangGraph trace collection |
| `LANGCHAIN_TRACING_V2` | `true` *(auto-set when key present)* | Explicit toggle; set to `false` to disable even when the key is set |
| `LANGSMITH_PROJECT` | `agentic-shopping-assistant` | Project name in LangSmith UI |
| `LOG_LEVEL` | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated allowed origins for CORS |

### Setting secrets on Fly.io

```bash
fly secrets set GROQ_API_KEY=<your-key>
fly secrets set LANGSMITH_API_KEY=<your-key>
fly secrets set LANGSMITH_PROJECT=agentic-shopping-assistant
```

Do **not** commit `.env` files — they are gitignored.  For local development copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
# edit .env
```

---

## WebSocket protocol

Connect to `ws://localhost:8000/chat/stream`.

**Client → Server frames** (JSON):

```jsonc
// Start a turn
{"type": "user_message", "message": "show me blue jackets", "conversation_id": null}

// Cancel in-flight turn
{"type": "cancel"}
```

**Server → Client frames** (in order):

```jsonc
{"type": "session",    "conversation_id": "<uuid4>"}
{"type": "routing",    "decision": {"action": "search", "query": "blue jackets"}}
{"type": "tool_start", "tool": "search"}
{"type": "items",      "items": [{"article_id": "...", "prod_name": "...", ...}]}
{"type": "token",      "text": "Here are "}
{"type": "token",      "text": "some great jackets!"}
{"type": "done",       "final_state": {"filters": {}, "out_of_catalogue": false, "new_items_this_turn": true, "response": "Here are some great jackets!"}}

// On cancel
{"type": "cancelled"}

// On error
{"type": "error", "message": "...", "code": "internal_error"}
```

The `items` frame includes full `ItemSummary` payloads (article_id, prod_name, colour, image_url, score, etc.) so frontends can render product cards without N+1 catalogue fetches.

---

## LangSmith tracing

When `LANGSMITH_API_KEY` is set, every LangGraph agent invocation is traced automatically via LangChain's built-in LANGCHAIN_TRACING_V2 instrumentation.  Each trace is labelled with the project name from `LANGSMITH_PROJECT`.

To view traces, open [smith.langchain.com](https://smith.langchain.com) and navigate to the project.
