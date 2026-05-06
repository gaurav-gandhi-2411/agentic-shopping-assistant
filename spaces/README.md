---
title: Agentic Shopping Assistant
emoji: 🛍️
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# Agentic Shopping Assistant — Streamlit UI

Thin Streamlit client that connects to the Shopping Assistant API.
All agent, retrieval, and LLM logic runs in the API process; the UI
only handles rendering and WebSocket communication.

**Ask things like:**
- *"show me black summer jackets"*
- *"something light and breathable for the beach"*
- *"compare the first two you showed me"*
- *"anything in blue instead?"*

Built with LangGraph · FAISS + BM25 hybrid retrieval · Groq (Llama 3.1 8B)

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `ws://localhost:8000` | WebSocket base URL of the Shopping Assistant API. Use `wss://` for TLS (production). |

**Local development:**
```bash
# 1. Start the API (separate terminal)
uvicorn api.main:app --port 8000

# 2. Start the UI
BACKEND_URL=ws://localhost:8000 streamlit run spaces/app.py
```

**Pointing at a deployed API:**
```bash
BACKEND_URL=wss://your-app.fly.dev streamlit run spaces/app.py
```

---

## Dependencies

`spaces/requirements.txt` intentionally contains only three packages:

| Package | Purpose |
|---|---|
| `streamlit` | UI framework |
| `websockets` | WS client for `/chat/stream` |
| `httpx` | HTTP client for `GET /catalogue/{id}/similar` |

The UI has **no** dependency on FAISS, sentence-transformers, LangGraph, or any LLM SDK — those all live in the API container.
