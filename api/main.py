import asyncio
import json
import queue
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import requests as _requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.llm.client import get_llm_client
from src.memory.conversation import ConversationMemory
from src.agents.graph import build_graph


config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    save_dir = Path("data/processed")
    df = pd.read_parquet(save_dir / "catalogue.parquet")
    dense = DenseRetriever.load(config, save_dir)
    sparse = SparseRetriever.load(config, save_dir)
    retriever = HybridRetriever(dense, sparse, df, config)
    llm = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    # streaming_mode=True: respond/clarify nodes defer LLM call to the API
    # so the API can stream tokens directly from Ollama.
    agent = build_graph(retriever, df, llm, memory, config, streaming_mode=True)

    app.state.agent = agent
    app.state.llm = llm
    app.state.conversations = {}   # conversation_id → {messages, filters, retrieved_items}
    print("Agent ready.")
    yield


app = FastAPI(title="Agentic Shopping Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_update(state: dict, update: dict) -> dict:
    """Merge a LangGraph node update into the accumulated state."""
    result = dict(state)
    for key, val in update.items():
        if key == "messages":                     # Annotated[list, operator.add]
            result[key] = state.get(key, []) + val
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        r = _requests.get(f"{config['llm']['host']}/api/tags", timeout=3)
        ollama = "ok" if r.status_code == 200 else "error"
    except Exception:
        ollama = "unreachable"
    return {"status": "ok", "ollama": ollama}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    agent = app.state.agent
    llm = app.state.llm
    conversations: dict = app.state.conversations

    conv = conversations.get(req.conversation_id, {
        "messages": [], "filters": {}, "retrieved_items": [],
    })

    initial_state = {
        "messages": conv["messages"] + [{"role": "user", "content": req.message}],
        "user_query": req.message,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": conv["retrieved_items"],
        "filters": conv["filters"],
        "final_answer": None,
        "iteration": 0,
    }

    ev_queue: queue.Queue = queue.Queue()

    def _run():
        try:
            acc = dict(initial_state)

            # --- Phase 1: run the graph (routing + tool calls, no LLM respond) ---
            for chunk in agent.stream(initial_state, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                state_up = chunk[node_name]
                acc = _apply_update(acc, state_up)

                if node_name in ("search", "compare"):
                    tc_list = state_up.get("tool_calls", [])
                    args: dict = {}
                    for tc in reversed(tc_list):
                        k = list(tc.keys())[0]
                        if k in ("search", "compare"):
                            args = tc[k]
                            break
                    ev_queue.put(json.dumps({
                        "type": "tool_call", "tool": node_name, "args": args,
                    }))
                    items = state_up.get("retrieved_items")
                    if items:
                        ev_queue.put(json.dumps({"type": "items", "items": items}))

                elif node_name == "filter":
                    tc_list = state_up.get("tool_calls", [])
                    args = {}
                    for tc in reversed(tc_list):
                        k = list(tc.keys())[0]
                        if k == "filter":
                            args = tc[k]
                            break
                    ev_queue.put(json.dumps({
                        "type": "tool_call", "tool": "filter", "args": args,
                    }))

            # --- Phase 2: stream the LLM response ---
            plan = json.loads(acc.get("current_plan") or "{}")
            action = plan.get("action", "")
            full_answer = ""

            if action == "pending_respond":
                prompt = plan["prompt"]
                for tok in llm.generate_stream(prompt):
                    full_answer += tok
                    ev_queue.put(json.dumps({"type": "token", "content": tok}))

            elif action == "pending_answer":
                # clarify: short text, stream word-by-word
                text = plan.get("text", "")
                full_answer = text
                for word in text.split():
                    ev_queue.put(json.dumps({"type": "token", "content": word + " "}))

            # --- Persist conversation state ---
            new_msgs = acc.get("messages", []) + (
                [{"role": "assistant", "content": full_answer}] if full_answer else []
            )
            conversations[req.conversation_id] = {
                "messages": new_msgs,
                "filters": acc.get("filters", {}),
                "retrieved_items": acc.get("retrieved_items", []),
            }

            ev_queue.put(json.dumps({"type": "done"}))

        except Exception as exc:
            ev_queue.put(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            ev_queue.put(None)   # sentinel — tells event_gen to stop

    threading.Thread(target=_run, daemon=True).start()

    async def event_gen():
        loop = asyncio.get_event_loop()
        while True:
            data = await loop.run_in_executor(None, ev_queue.get)
            if data is None:
                break
            yield {"data": data}

    return EventSourceResponse(event_gen())
