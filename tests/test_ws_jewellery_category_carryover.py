"""Full-stack WS /chat/stream reproduction of the footwear -> jewellery category
carryover bug.

Live-proven 2026-07-23: in a fresh session, turn 1 "juttis for a lehenga" (correctly
footwear) followed by turn 2 "gold jewellery for a wedding" incorrectly returned MORE
footwear (juttis) instead of jewellery — the assistant's own rationale said "I've
found some beautiful gold juttis... but the catalogue doesn't have an exact match".

Root cause: src/agents/intent_parser.py's _GARMENT_RULES had zero jewellery
vocabulary at all, so parse_intent("gold jewellery for a wedding").garment_type was
None. router_node's merge_with_context() then silently inherited garment_type from
session_context (the PRIOR turn's dominant product_type, "footwear"), which
hard-set _plan_filters["product_type_name"]="footwear" for a turn that never asked
for footwear at all — a genuine intent-classifier-over-trusts-history bug, not a
missing-inventory issue (the catalogue has thousands of real jewellery rows and a
FRESH "gold jewellery for a wedding" session resolves correctly on its own).

Mirrors tests/test_ws_multiturn_gender.py's fixture/harness exactly (real compiled
graph, unified cross-store index, real WS route, demo-anon session, per-turn
reconnect) — see that module's docstring for why the graph-level-only repro in
tests/test_agent.py isn't sufficient here (this bug is a router_node/search_node
session-carryover defect only observable through the real per-turn state
round-trip, same reasoning as the gender-carryover bug that file covers).
"""
from __future__ import annotations

import os
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

    IntentParser handles routing deterministically for plain product queries (no
    LLM call), so these responses only need to satisfy respond_node's streamed
    token generation (generate_stream) — the actual text content is irrelevant to
    this test, which asserts on retrieved items / filters, not response prose.
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


@pytest.fixture(scope="module")
def _unified_index() -> tuple[HybridRetriever, pd.DataFrame]:
    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


@pytest.fixture
def demo_client(monkeypatch: pytest.MonkeyPatch, _unified_index) -> TestClient:
    """Wires the real compiled agent graphs (unified index) into api.deps and
    enables the demo-anon auth path, exactly like the deployed service.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM(["Here you go."] * 40)

    deps._init(
        retriever=retriever,
        catalogue_df=catalogue_df,
        llm=llm,
        config=_MINIMAL_CONFIG,
        session_store=InMemorySessionStore(),
    )
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "ws-jewellery-carryover-test-secret")
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.setattr(deps, "_db_engine", None)  # skip daily-cap DB checks

    # api.routes.chat._DEMO_SESSIONS is a module-level dict keyed by conversation_id;
    # reset it so tests never see state left behind by another test/run.
    import api.routes.chat as chat_module

    monkeypatch.setattr(chat_module, "_DEMO_SESSIONS", {})

    return TestClient(app, raise_server_exceptions=True)


def _mint_ticket(client: TestClient, session_token: str) -> str:
    """POST /auth/ws-ticket with the demo session token — mints a fresh, single-use
    60s ticket, exactly as the frontend does before opening each new WS connection.
    """
    resp = client.post(
        "/auth/ws-ticket", headers={"Authorization": f"Bearer {session_token}"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["ticket"]


def _run_ws_turn(
    client: TestClient, ticket: str, message: str, conversation_id: str | None
) -> tuple[str, list[dict]]:
    """Open one fresh WS connection (per-turn reconnect, matching the browser),
    send one user message, and collect (conversation_id, items) from the response
    frames. Blocks until the "done" frame or an "error" frame.
    """
    with client.websocket_connect(f"/chat/stream?ticket={ticket}") as ws:
        payload: dict = {"type": "user_message", "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        ws.send_json(payload)

        cid: str | None = conversation_id
        items: list[dict] = []
        for _ in range(200):
            frame = ws.receive_json()
            ftype = frame.get("type")
            if ftype == "session":
                cid = frame["conversation_id"]
            elif ftype == "items":
                items = frame.get("items", [])
            elif ftype == "error":
                raise AssertionError(f"WS error frame: {frame}")
            elif ftype in ("done", "cancelled"):
                break
        assert cid is not None
        return cid, items


@pytest.mark.requires_index
def test_ws_jewellery_query_after_footwear_turn_returns_jewellery_not_footwear(
    demo_client: TestClient,
) -> None:
    """Reproduces the live browser bug through the REAL WS route, demo-anon session
    path, and per-turn reconnect (fresh ticket + fresh WS connection per turn, same
    conversation_id) — exactly like the frontend drives /chat/stream.

    2-turn conversation: "juttis for a lehenga" (footwear) -> "gold jewellery for a
    wedding". Turn 2 must return jewellery items, not more juttis/footwear.
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"

    resp = demo_client.post("/demo/session")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    session_token = body["session_token"]
    first_ticket = body["ws_ticket"]

    conversation_id: str | None = None
    conversation_id, turn1_items = _run_ws_turn(
        demo_client, first_ticket, "juttis for a lehenga", conversation_id
    )

    # Precondition: turn 1 must actually land on footwear, matching the live report.
    turn1_types = [it.get("product_type", "").lower() for it in turn1_items]
    assert turn1_types and all(t == "footwear" for t in turn1_types), (
        f"Precondition failed: turn 1 should be all footwear, got {turn1_types}"
    )

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id, turn2_items = _run_ws_turn(
        demo_client, ticket, "gold jewellery for a wedding", conversation_id
    )

    turn2_types = [it.get("product_type", "").lower() for it in turn2_items]
    assert turn2_types, "Turn 2 returned no items at all"
    assert all(t != "footwear" for t in turn2_types), (
        f"Turn 2 ('gold jewellery for a wedding') incorrectly returned footwear "
        f"items carried over from turn 1, got {turn2_types} "
        f"(display_names={[it.get('display_name') for it in turn2_items]})"
    )
    assert all(t == "jewellery" for t in turn2_types), (
        f"Expected turn 2 items to all be jewellery, got {turn2_types} "
        f"(display_names={[it.get('display_name') for it in turn2_items]})"
    )
