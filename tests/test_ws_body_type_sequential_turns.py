"""Wave 7 hang fix (turn 2) — a bare body-type STATEMENT ack turn (commit
5bf043f) followed IMMEDIATELY by a separate occasion-only turn must complete
normally through the REAL WS route, and a genuinely stalled downstream call
must surface an honest timeout error instead of hanging the connection
forever.

Live repro (2026-07-10): fresh session, turn 1 = "I have an inverted triangle
silhouette" (gets the new deterministic ack in ~3s, never touches the LLM —
see body_type_ack_message's docstring), turn 2 = "sangeet look under 8000"
(fresh occasion, no restated body type, first REAL LLM call of the session)
-> hung 180s+ live with zero reply, zero error.

Direct agent.invoke() reproduction of this exact 2-turn sequence with a mocked
LLM completes in well under 5s with a correct, non-empty outfit result (see
scratchpad investigation + tests/test_body_type_bare_statement.py's existing
same-message regression test) — ruling out a graph-logic hang or a body_type
history-reconstruction type mismatch. That isolates the cause to a downstream
dependency stall with no bounded wall-clock time (see GroqClient.chat's TPD
retry branch in src/llm/client.py, which has no ceiling on total retry time)
combined with api/routes/chat.py::ws_chat previously having NO deadline of its
own around the agent thread — so a stalled dependency left the WS connection
open indefinitely with no response and no error.

This module proves, via the REAL WS route (same demo-anon session path and
per-turn reconnect pattern as tests/test_ws_multiturn_gender.py):
1. The exact reported 2-turn sequence completes with a real outfit result.
2. The sanity case that already worked (body type + occasion in ONE message)
   still works — no regression.
3. A genuinely stalled agent turn is bounded by ws_chat's new turn deadline
   and surfaces a "turn_timeout" error frame instead of hanging.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

UNIFIED_DIR = Path("data/processed/unified")

_MINIMAL_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_dim": 384,
        "rrf_k": 60,
        "top_k": 50,
        "final_k": 10,
        "store_diversity": 0.2,
    },
}


class _MockLLM:
    """Cycles through canned responses; repeats the last one when exhausted.

    Same shape as test_ws_multiturn_gender.py's _MockLLM. The bare body-type
    ack turn never calls this (deterministic short-circuit — see
    body_type_ack_message's docstring), so the FIRST real call in these tests
    is always generate_rationales() during the occasion-outfit turn.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield self._next()


class _StallingLLM(_MockLLM):
    """Sleeps past the (monkeypatched, short) WS turn deadline on every call —
    simulates a downstream dependency stall (e.g. an LLM provider's unbounded
    rate-limit retry loop) so the timeout branch itself can be exercised
    without waiting for a real multi-minute hang.
    """

    def __init__(self, stall_seconds: float, responses: list[str]) -> None:
        super().__init__(responses)
        self._stall_seconds = stall_seconds

    def chat(self, messages: list[dict], **kwargs) -> str:
        time.sleep(self._stall_seconds)
        return super().chat(messages, **kwargs)


@pytest.fixture(scope="module")
def _unified_index() -> tuple[HybridRetriever, pd.DataFrame]:
    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _demo_client(
    monkeypatch: pytest.MonkeyPatch, _unified_index, llm
) -> TestClient:
    retriever, catalogue_df = _unified_index

    deps._init(
        retriever=retriever,
        catalogue_df=catalogue_df,
        llm=llm,
        config=_MINIMAL_CONFIG,
        session_store=InMemorySessionStore(),
    )
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "ws-body-type-sequential-test-secret")
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.setattr(deps, "_db_engine", None)  # skip daily-cap DB checks

    import api.routes.chat as chat_module

    monkeypatch.setattr(chat_module, "_DEMO_SESSIONS", {})

    return TestClient(app, raise_server_exceptions=True)


def _mint_ticket(client: TestClient, session_token: str) -> str:
    resp = client.post(
        "/auth/ws-ticket", headers={"Authorization": f"Bearer {session_token}"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["ticket"]


def _run_ws_turn(
    client: TestClient, ticket: str, message: str, conversation_id: str | None
) -> tuple[str, list[dict], list[dict]]:
    """Returns (conversation_id, items, all_frames) — all_frames lets callers
    assert on error frames (e.g. "turn_timeout") that carry no items.
    """
    with client.websocket_connect(f"/chat/stream?ticket={ticket}") as ws:
        payload: dict = {"type": "user_message", "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        ws.send_json(payload)

        cid: str | None = conversation_id
        items: list[dict] = []
        frames: list[dict] = []
        for _ in range(200):
            frame = ws.receive_json()
            frames.append(frame)
            ftype = frame.get("type")
            if ftype == "session":
                cid = frame["conversation_id"]
            elif ftype == "items":
                items = frame.get("items", [])
            elif ftype in ("done", "cancelled", "error"):
                break
        assert cid is not None
        return cid, items, frames


@pytest.mark.requires_index
def test_ws_body_type_ack_then_occasion_completes_with_look(
    monkeypatch: pytest.MonkeyPatch, _unified_index
) -> None:
    """The exact live repro sequence, turn by turn, through the real WS route:
    turn 1 = bare body-type statement (ack, no LLM call, no items), turn 2 =
    occasion-only (must reconstruct body_type from history and compose a real
    outfit) — must complete promptly with real items, never hang or error.
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    llm = _MockLLM(["A lovely outfit for the occasion."] * 10)
    client = _demo_client(monkeypatch, _unified_index, llm)

    resp = client.post("/demo/session")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    session_token = body["session_token"]
    first_ticket = body["ws_ticket"]

    t0 = time.monotonic()
    conversation_id, turn1_items, turn1_frames = _run_ws_turn(
        client, first_ticket, "I have an inverted triangle silhouette", None
    )
    t1 = time.monotonic()
    assert not turn1_items, "a bare body-type statement must never trigger a product search"
    assert not any(f.get("type") == "error" for f in turn1_frames), turn1_frames
    assert (t1 - t0) < 30, f"turn 1 (ack) took {t1 - t0:.1f}s — should be near-instant"

    ticket = _mint_ticket(client, session_token)
    t2 = time.monotonic()
    conversation_id, turn2_items, turn2_frames = _run_ws_turn(
        client, ticket, "sangeet look under 8000", conversation_id
    )
    t3 = time.monotonic()

    assert not any(f.get("type") == "error" for f in turn2_frames), (
        f"turn 2 must not error — frames={turn2_frames}"
    )
    assert turn2_items, (
        f"turn 2 must compose a real outfit (occasion-only, body_type reconstructed "
        f"from turn 1's history) — got zero items, frames={turn2_frames}"
    )
    assert (t3 - t2) < 60, f"turn 2 took {t3 - t2:.1f}s — should complete well under the deadline"


@pytest.mark.requires_index
def test_ws_body_type_same_message_as_occasion_still_composes(
    monkeypatch: pytest.MonkeyPatch, _unified_index
) -> None:
    """Regression guard: the ALREADY-working P3 flow (body type + occasion
    stated together in ONE message) must still compose a real outfit through
    the real WS route — the new sequential-turn fix must never change this
    single-message case's behaviour.
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    llm = _MockLLM(["A lovely outfit for the occasion."] * 10)
    client = _demo_client(monkeypatch, _unified_index, llm)

    resp = client.post("/demo/session")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    first_ticket = body["ws_ticket"]

    conversation_id, items, frames = _run_ws_turn(
        client, first_ticket, "I'm pear-shaped, sangeet look under 8000", None
    )
    assert not any(f.get("type") == "error" for f in frames), frames
    assert items, f"expected a composed outfit, got frames={frames}"


@pytest.mark.requires_index
def test_ws_turn_timeout_surfaces_honest_error_not_hang(
    monkeypatch: pytest.MonkeyPatch, _unified_index
) -> None:
    """Directly exercises ws_chat's new turn deadline: a downstream call that
    stalls past the deadline must surface a "turn_timeout" error frame within
    the deadline window, never hang the WS connection indefinitely.
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    monkeypatch.setenv("WS_TURN_DEADLINE_SECONDS", "1")
    llm = _StallingLLM(stall_seconds=5.0, responses=["Here you go."])
    client = _demo_client(monkeypatch, _unified_index, llm)

    resp = client.post("/demo/session")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    first_ticket = body["ws_ticket"]

    t0 = time.monotonic()
    _, _, frames = _run_ws_turn(
        client, first_ticket, "sangeet look under 8000", None
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0, (
        f"deadline should cut the turn off well before the 5s stall completes, "
        f"took {elapsed:.1f}s — frames={frames}"
    )
    error_frames = [f for f in frames if f.get("type") == "error"]
    assert error_frames, f"expected a turn_timeout error frame, got frames={frames}"
    assert error_frames[0].get("code") == "turn_timeout", error_frames[0]
