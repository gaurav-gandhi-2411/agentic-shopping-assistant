"""Wave 7 P2 — couple-from-scratch board serialization (API layer).

Covers the api/routes/chat.py serialization added on top of
src/agents/graph.py's `_compose_couple_from_scratch` (which sets the
PRIMARY board's look_role to "couple_primary" and a parallel set of
`partner_*` AgentState fields — partner_retrieved_items, partner_look_id,
partner_occasion, partner_look_gender, partner_outfit_rationale,
partner_budget_total_inr, partner_suppressed_slots, partner_look_role
("couple_partner"), partner_look_title, partner_coordinated_with):

  1. POST /chat: a couple-from-scratch agent result serializes a SECOND
     `partner_look` payload (items, look_role="partner", look_title,
     coordinated_with) alongside the primary look, and the primary board's
     own `look_role` is normalized from "couple_primary" to None (not a
     "partner" board itself).
  2. WS /chat/stream: the same fields land in the terminal "done" frame's
     `final_state.partner_look`.
  3. Regression: a turn with NO partner_* fields (ordinary primary turn, or
     the pre-existing single-partner-turn flow with look_role="partner")
     must NOT get a partner_look payload, and its own look_role passes
     through unchanged.

Uses the same mocked-agent pattern as tests/test_partner_styling.py's
TestChatResponsePartnerFields and tests/test_api_ws.py — no real index or
live LLM required. Deliberately a SEPARATE file (not appended to
tests/test_partner_styling.py) to avoid touching a file the backend
orchestration work is actively editing in parallel.
"""
from __future__ import annotations

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore

_MINIMAL_CONFIG = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}


class _MockLLM:
    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        return "ok"

    def generate_stream(self, prompt: str, system: str | None = None, **kwargs: Any) -> Iterator[str]:
        yield "ok"

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        return "ok"

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        yield "ok"


class _MockAgent:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def invoke(self, state: dict, **kwargs: Any) -> dict[str, Any]:
        result = dict(self._result)
        result.setdefault("messages", state.get("messages", []))
        return result


def _make_factory(agent: Any) -> Any:
    def get_factory() -> Any:
        def factory(memory: Any, streaming: bool = False) -> Any:
            return agent

        return factory

    return get_factory


_COUPLE_RESULT: dict[str, Any] = {
    # Primary board — "Her look"
    "retrieved_items": [
        {
            "article_id": "WOMEN1",
            "prod_name": "Rust Embellished Lehenga",
            "display_name": "Rust Embellished Lehenga",
            "colour": "rust",
            "product_type": "Lehenga",
            "department": "Women",
            "image_url": "https://example.com/w1.jpg",
            "gender": "women",
            "_role": "seed",
        }
    ],
    "filters": {},
    "tool_calls": [{"router_decision": {"action": "outfit"}}],
    "final_answer": "**Her look**\n\n...\n\n**His look**\n\n...",
    "iteration": 1,
    "new_items_this_turn": True,
    "out_of_catalogue": False,
    "excluded_colours": None,
    "look_id": "primary-look-id",
    "occasion": "reception",
    "look_gender": "women",
    "outfit_rationale": "A rust lehenga perfect for a reception.",
    "budget_total_inr": 11000.0,
    "suppressed_slots": None,
    "look_role": "couple_primary",
    "look_title": "Her look",
    "coordinated_with": None,
    # Partner board — "His look"
    "partner_retrieved_items": [
        {
            "article_id": "MEN1",
            "prod_name": "Navy Blue Sherwani",
            "display_name": "Navy Blue Sherwani",
            "colour": "navy blue",
            "product_type": "Sherwani",
            "department": "Men",
            "image_url": "https://example.com/m1.jpg",
            "gender": "men",
            "_role": "seed",
        }
    ],
    "partner_look_id": "partner-look-id",
    "partner_occasion": "reception",
    "partner_look_gender": "men",
    "partner_outfit_rationale": "A navy blue sherwani that complements the rust lehenga.",
    "partner_budget_total_inr": 9000.0,
    "partner_suppressed_slots": None,
    "partner_look_role": "couple_partner",
    "partner_look_title": "His look",
    "partner_coordinated_with": (
        "Coordinated with the rust lehenga — navy blue complements it "
        "at the same reception formal level."
    ),
}

# Ordinary single-look turn — no partner_* fields at all.
_SOLO_RESULT: dict[str, Any] = {
    "retrieved_items": [
        {
            "article_id": "SOLO1",
            "prod_name": "Blue Jacket",
            "display_name": "Blue Jacket",
            "colour": "blue",
            "product_type": "Jacket",
            "department": "Women",
            "image_url": "https://example.com/j1.jpg",
            "gender": "women",
            "_role": "seed",
        }
    ],
    "filters": {},
    "tool_calls": [{"router_decision": {"action": "outfit"}}],
    "final_answer": "**Outfit suggestion**",
    "iteration": 1,
    "new_items_this_turn": True,
    "out_of_catalogue": False,
    "excluded_colours": None,
    "look_id": "solo-look-id",
    "occasion": "casual",
    "look_gender": "women",
    "outfit_rationale": "A casual blue jacket look.",
    "budget_total_inr": 2500.0,
    "suppressed_slots": None,
    "look_role": None,
    "look_title": None,
    "coordinated_with": None,
}

# Pre-existing single-partner-turn flow — look_role="partner" directly, no
# partner_* fields (mirrors tests/test_partner_styling.py's mock).
_SINGLE_PARTNER_RESULT: dict[str, Any] = {
    "retrieved_items": [
        {
            "article_id": "MEN2",
            "prod_name": "Grey Shirt",
            "display_name": "Grey Shirt",
            "colour": "grey",
            "product_type": "shirt",
            "department": "Men",
            "image_url": "https://example.com/m2.jpg",
            "gender": "men",
            "_role": "seed",
        }
    ],
    "filters": {},
    "tool_calls": [{"router_decision": {"action": "outfit"}}],
    "final_answer": "**Your partner's look**",
    "iteration": 1,
    "new_items_this_turn": True,
    "out_of_catalogue": False,
    "excluded_colours": None,
    "look_id": "single-partner-look-id",
    "occasion": "casual",
    "look_gender": "men",
    "outfit_rationale": "A grey shirt look.",
    "budget_total_inr": 1500.0,
    "suppressed_slots": None,
    "look_role": "partner",
    "look_title": "Your partner's look",
    "coordinated_with": "Coordinated with the rust dress.",
}


@pytest.fixture(autouse=True)
def inject_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemorySessionStore()
    monkeypatch.setattr(deps, "_session_store", store)
    monkeypatch.setattr(deps, "_llm", _MockLLM())
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


class TestPostChatCoupleLookSerialization:
    def test_couple_from_scratch_emits_partner_look_and_normalizes_primary_role(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_MockAgent(_COUPLE_RESULT)))

        resp = client.post("/chat", json={"message": "style us as a couple for a reception"})

        assert resp.status_code == 200
        data = resp.json()

        # Primary board — "couple_primary" is internal bookkeeping, normalized
        # to None externally (no "Partner look" badge on the primary board).
        assert data["look_role"] is None
        assert data["items"][0]["article_id"] == "WOMEN1"

        # Partner board — a fully separate payload.
        partner_look = data["partner_look"]
        assert partner_look is not None
        assert partner_look["look_role"] == "partner"
        assert partner_look["look_title"] == "His look"
        assert "rust lehenga" in partner_look["coordinated_with"]
        assert partner_look["items"][0]["article_id"] == "MEN1"
        assert partner_look["items"][0]["gender"] == "men"
        assert partner_look["budget_total_inr"] == 9000.0

    def test_solo_turn_omits_partner_look(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_MockAgent(_SOLO_RESULT)))

        resp = client.post("/chat", json={"message": "show me a blue jacket outfit"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["partner_look"] is None
        assert data["look_role"] is None

    def test_single_partner_turn_flow_is_unaffected(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        """The PRE-EXISTING single-partner-turn flow (e.g. "what should my
        husband wear with this?") never populates partner_retrieved_items —
        must still serialize exactly as before: look_role="partner" passed
        through unchanged, partner_look absent."""
        monkeypatch.setattr(
            deps, "get_agent_factory", _make_factory(_MockAgent(_SINGLE_PARTNER_RESULT))
        )

        resp = client.post("/chat", json={"message": "what should my husband wear with this?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["look_role"] == "partner"
        assert data["look_title"] == "Your partner's look"
        assert data["partner_look"] is None


class TestWsChatCoupleLookSerialization:
    def test_couple_from_scratch_done_frame_carries_partner_look(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_MockAgent(_COUPLE_RESULT)))

        with client.websocket_connect("/chat/stream") as ws:
            ws.send_json(
                {"type": "user_message", "message": "style us as a couple for a reception"}
            )

            done_msg = None
            for _ in range(30):
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done_msg = msg
                    break

        assert done_msg is not None
        final_state = done_msg["final_state"]
        assert final_state["look_role"] is None

        partner_look = final_state["partner_look"]
        assert partner_look is not None
        assert partner_look["look_role"] == "partner"
        assert partner_look["look_title"] == "His look"
        assert partner_look["items"][0]["article_id"] == "MEN1"
        assert partner_look["occasion"] == "reception"

    def test_solo_turn_done_frame_omits_partner_look(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        monkeypatch.setattr(deps, "get_agent_factory", _make_factory(_MockAgent(_SOLO_RESULT)))

        with client.websocket_connect("/chat/stream") as ws:
            ws.send_json({"type": "user_message", "message": "show me a blue jacket outfit"})

            done_msg = None
            for _ in range(30):
                msg = ws.receive_json()
                if msg["type"] == "done":
                    done_msg = msg
                    break

        assert done_msg is not None
        assert done_msg["final_state"]["partner_look"] is None
