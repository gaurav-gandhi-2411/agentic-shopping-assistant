"""Empty assistant replies must never reach the user silently.

Incident (W10 grounding harness, 2026-11-03): with groq/openai/gpt-oss-20b, 3 of 6
audited WS turns streamed 5 items but an empty answer. Mechanism, measured live:
gpt-oss is a reasoning model, hidden reasoning tokens count against max_tokens, and
at default effort the respond prompt with max_tokens=400 produced 0 content chars
and finish_reason=length. Two layers are covered here, all hermetic (no network):
  1. GroqClient passes reasoning_effort for gpt-oss models and flags empty streams.
  2. The WS path retries once, then serves a grounded fallback; REST respond_node too.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore
from src.agents.grounding import grounded_fallback_reply
from src.llm.client import GroqClient

# ---------------------------------------------------------------------------
# grounded_fallback_reply
# ---------------------------------------------------------------------------


def test_fallback_reply_names_items_and_prices_from_data_only() -> None:
    items = [
        {"display_name": "Riviera Dress", "price_inr": 2999},
        {"display_name": "Linen Shirt"},  # no price -> no invented price
    ]
    text = grounded_fallback_reply(items)
    assert "Riviera Dress (₹2,999)" in text
    assert "Linen Shirt" in text
    assert "Linen Shirt (" not in text


def test_fallback_reply_without_items_is_nonempty() -> None:
    assert grounded_fallback_reply([]).strip()
    assert grounded_fallback_reply([{"display_name": ""}]).strip()


# ---------------------------------------------------------------------------
# GroqClient: reasoning_effort + empty-stream diagnostics
# ---------------------------------------------------------------------------


def _chunk(content: str | None, finish: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)
        ],
        usage=None,
    )


def _groq(monkeypatch: pytest.MonkeyPatch, chunks: list, model: str = "openai/gpt-oss-20b",
          effort: str | None = "low") -> tuple[GroqClient, MagicMock]:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    create = MagicMock(return_value=iter(chunks))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("groq.Groq", lambda api_key, timeout: fake)
    llm_cfg: dict[str, Any] = {
        "temperature": 0.2, "max_tokens": 400, "timeout_seconds": 60, "groq_model": model,
    }
    if effort:
        llm_cfg["groq_reasoning_effort"] = effort
    return GroqClient({"llm": llm_cfg}), create


def test_groq_stream_sends_reasoning_effort_for_gpt_oss(monkeypatch: pytest.MonkeyPatch) -> None:
    client, create = _groq(monkeypatch, [_chunk("Hi", None), _chunk(None, "stop")])
    assert "".join(client.chat_stream([{"role": "user", "content": "x"}])) == "Hi"
    assert create.call_args.kwargs["reasoning_effort"] == "low"


def test_groq_non_gpt_oss_model_gets_no_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    client, create = _groq(monkeypatch, [_chunk("Hi", "stop")], model="llama-3.1-8b-instant")
    list(client.chat_stream([{"role": "user", "content": "x"}]))
    assert "reasoning_effort" not in create.call_args.kwargs


def test_groq_empty_stream_with_finish_length_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Reasoning-only stream: every delta has content=None, ends with finish_reason=length.
    client, _ = _groq(monkeypatch, [_chunk(None), _chunk(None), _chunk(None, "length")])
    with caplog.at_level(logging.WARNING):
        out = list(client.chat_stream([{"role": "user", "content": "x"}]))
    assert out == []
    assert client.last_finish_reason == "length"
    events = [json.loads(r.getMessage()) for r in caplog.records
              if "llm_empty_content" in r.getMessage()]
    assert events and events[0]["finish_reason"] == "length"
    assert events[0]["provider"] == "groq" and events[0]["model"] == "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# WS path
# ---------------------------------------------------------------------------

_ITEMS = [{
    "article_id": "1", "prod_name": "Test Jacket", "display_name": "Test Jacket (Blue)",
    "colour": "Blue", "product_type": "Jacket", "department": "Ladieswear",
    "image_url": "https://example.com/1.jpg", "detail_desc": "A jacket.", "score": 0.9,
    "price_inr": 1999,
}]
_RESULT: dict[str, Any] = {
    "retrieved_items": _ITEMS,
    "filters": {},
    "tool_calls": [
        {"router_decision": {"action": "search", "query": "jacket"}},
        {"search": {"query": "jacket", "filters": {}}},
    ],
    "final_answer": None,
    "current_plan": json.dumps({"action": "pending_respond", "prompt": "Describe."}),
    "iteration": 1,
    "new_items_this_turn": True,
    "out_of_catalogue": False,
    "excluded_colours": None,
}


class _Agent:
    def invoke(self, state: dict, **kwargs: Any) -> dict[str, Any]:
        return {**_RESULT, "messages": state.get("messages", [])}


class _EmptyStreamLLM:
    """Streams nothing (reasoning-only / finish_reason=length); optional non-stream retry text."""

    model = "openai/gpt-oss-20b"
    last_finish_reason = "length"

    def __init__(self, retry_text: str | None = "") -> None:
        self.retry_text = retry_text
        self.generate_calls = 0

    def generate_stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        return iter(())

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str | None:
        self.generate_calls += 1
        return self.retry_text


class _NormalLLM:
    model = "x"

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate_stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        yield from ("Lovely ", "jacket.")

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.generate_calls += 1
        return "unused"


@pytest.fixture(autouse=True)
def _deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_session_store", InMemorySessionStore())
    monkeypatch.setattr(deps, "_config", {
        "agent": {"max_iterations": 3},
        "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    })
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")

    def get_factory() -> Any:
        return lambda memory, streaming=False: _Agent()

    monkeypatch.setattr(deps, "get_agent_factory", get_factory)


def _run_turn(llm: Any) -> tuple[str, str]:
    deps._llm = llm
    client = TestClient(app, raise_server_exceptions=True)
    tokens: list[str] = []
    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": "show me jackets"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "token":
                tokens.append(msg["text"])
            elif msg["type"] == "done":
                return "".join(tokens), msg["final_state"]["response"]
            elif msg["type"] == "error":
                pytest.fail(f"unexpected error message: {msg}")


def test_ws_empty_stream_retries_non_streaming(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    llm = _EmptyStreamLLM(retry_text="A great blue jacket.")
    with caplog.at_level(logging.WARNING):
        streamed, final = _run_turn(llm)
    assert streamed == final == "A great blue jacket."
    assert llm.generate_calls == 1
    warn = [r.getMessage() for r in caplog.records if "ws empty reply after stream" in r.getMessage()]
    assert warn and "model=openai/gpt-oss-20b" in warn[0] and "finish_reason=length" in warn[0]


@pytest.mark.parametrize("retry_text", ["", None, "   "])
def test_ws_empty_stream_and_empty_retry_serves_grounded_fallback(
    retry_text: str | None,
) -> None:
    streamed, final = _run_turn(_EmptyStreamLLM(retry_text=retry_text))
    assert streamed == final
    assert "Test Jacket (Blue) (₹1,999)" in final


def test_ws_normal_stream_unaffected() -> None:
    llm = _NormalLLM()
    streamed, final = _run_turn(llm)
    assert streamed == final == "Lovely jacket."
    assert llm.generate_calls == 0


# ---------------------------------------------------------------------------
# REST respond_node (non-streaming)
# ---------------------------------------------------------------------------


class _NoneOnRespondLLM:
    """generate() returns None (Groq content=None on finish_reason=length) for the respond prompt."""

    model = "openai/gpt-oss-20b"

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str | None:
        return None if "fashion shopping assistant" in prompt else ""

    def generate_stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        return iter(())

    def chat(self, messages: list[dict], **kw: Any) -> str:
        return ""

    def chat_stream(self, messages: list[dict], **kw: Any) -> Iterator[str]:
        return iter(())


@pytest.mark.requires_index
def test_rest_respond_node_none_reply_uses_grounded_fallback() -> None:
    """Non-streaming respond_node must not return a blank answer when the LLM gives None."""
    import pandas as pd

    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever
    from tests.test_respond_history_and_stylist_reply import (
        _MINIMAL_CONFIG,
        UNIFIED_DIR,
        _blank_state,
    )

    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, df, _MINIMAL_CONFIG)
    llm = _NoneOnRespondLLM()
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, df, llm, _MINIMAL_CONFIG, streaming_mode=False)

    result = agent.invoke(_blank_state("red dress", memory))

    assert result.get("retrieved_items"), "precondition: search must return items"
    answer = result["final_answer"]
    assert answer and answer.strip()
    assert result["messages"][-1]["content"] == answer
