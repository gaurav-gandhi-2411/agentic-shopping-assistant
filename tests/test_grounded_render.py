"""V5 grounding by construction: ground_reply branches, the two corpora, and the wiring.

Acceptance rules (a)-(d) live in docs/grounding-v5-acceptance.md. Everything here is hermetic: no
LLM, no network, no real index (the respond_node test drives build_graph with a fake LLM and a
retriever that must never be called).
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import src.agents.grounded_render as gr
from api.main import app
from api.session import InMemorySessionStore
from src.agents.claim_checker import check
from src.agents.grounded_render import (
    GroundResult,
    add_id_rules,
    ground_reply,
    grounded_rendering_enabled,
)
from src.agents.grounding import GROUNDED_FALLBACK_PREFIX
from src.agents.outfit.rationale import generate_rationales
from tests import _grounding_v5_corpus as corpus

ITEMS = corpus.ITEMS


# ---------------------------------------------------------------------------------- corpora


@pytest.mark.parametrize("case", corpus.adversarial(), ids=lambda c: c["id"])
def test_adversarial_reply_has_no_surviving_ungrounded_claim(case: dict[str, Any]) -> None:
    out = ground_reply(case["reply"], ITEMS, user_text=case["user_text"]).text
    assert [b for b in case["bad_spans"] if b in out] == []
    # independent oracle: the checker must also find nothing in what was served
    assert check(out, ITEMS, user_text=case["user_text"]) == []
    assert all(k in out for k in case["must_keep"])


@pytest.mark.parametrize("case", corpus.clean(), ids=lambda c: c["id"])
def test_clean_reply_unchanged_apart_from_id_substitution(case: dict[str, Any]) -> None:
    result = ground_reply(case["reply"], ITEMS, user_text=case["user_text"])
    assert result.text == case["expected"]
    assert not result.replaced


def test_corpus_sizes_meet_the_pre_registered_minimums() -> None:
    assert len(corpus.adversarial()) >= 60
    assert len(corpus.clean()) >= 30
    cats = {c["category"] for c in corpus.adversarial()}
    assert {"wrong_price", "wrong_currency", "invented_size", "stock", "discount",
            "id_not_shown", "numeric_in_prose", "price_format_variant"} <= cats


def test_validator_latency_p95_is_far_below_the_5ms_budget() -> None:
    cases = [(c["reply"], c["user_text"]) for c in corpus.adversarial() + corpus.clean()]
    for reply, user in cases:  # warm-up pass
        ground_reply(reply, ITEMS, user_text=user)
    times: list[float] = []
    for reply, user in cases * 3:
        t0 = time.perf_counter()
        ground_reply(reply, ITEMS, user_text=user)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95 = times[int(0.95 * len(times)) - 1]
    # the registered budget is 5 ms (scripts/grounding_corpus_report.py enforces it); 25 ms here
    # keeps a loaded CI runner from flaking while still catching an order-of-magnitude regression
    assert p95 < 25.0


# ---------------------------------------------------------------------------------- branches


def test_item_and_price_tokens_render_from_the_record() -> None:
    r = ground_reply("Try [item:1001] at [price:1001].", ITEMS)
    assert r.text == "Try Riviera Cotton Sundress at ₹2,999."
    assert r.flags == [] and not r.replaced


def test_token_matching_is_case_and_space_tolerant() -> None:
    assert ground_reply("[ ITEM : 1002 ] is nice.", ITEMS).text == "Gloss Wrap Skirt is nice."


def test_unknown_id_drops_only_that_sentence() -> None:
    r = ground_reply("[item:9999] is lovely. [item:1002] is a good pick.", ITEMS)
    assert r.text == "Gloss Wrap Skirt is a good pick."
    assert "unknown_id:9999" in r.flags


def test_unsupported_token_kind_drops_the_sentence() -> None:
    r = ground_reply("[size:1001] fits well. [item:1002] is a good pick.", ITEMS)
    assert "fits well" not in r.text and "Gloss Wrap Skirt" in r.text
    assert "unsupported_token:size" in r.flags


def test_stray_unclosed_token_drops_the_sentence() -> None:
    r = ground_reply("Look at [item:1001 now. [item:1002] is a good pick.", ITEMS)
    assert "[item" not in r.text and "Gloss Wrap Skirt" in r.text
    assert "stray_token" in r.flags


def test_price_token_for_item_without_price_drops_the_sentence() -> None:
    items = [{"article_id": "7", "display_name": "Mystery Top"}]
    r = ground_reply("[item:7] at [price:7] is nice. [item:7] works for brunch.", items)
    assert r.text == "Mystery Top works for brunch."
    assert "no_record_value:price:7" in r.flags


def test_unattributed_claim_is_judged_against_the_previous_named_item() -> None:
    # item 1005 backs "in stock"; the follow-up "It is in stock." refers to item 1001 (no data)
    r = ground_reply("[item:1001] suits a daytime event. It is in stock.", ITEMS)
    assert r.text == "Riviera Cotton Sundress suits a daytime event."
    assert "unattributed_claim:stock" in r.flags
    ok = ground_reply("[item:1005] suits a daytime event. It is in stock.", ITEMS)
    assert ok.text == "Linen Co-ord Set suits a daytime event. It is in stock."


def test_unattributed_non_price_claim_must_hold_for_every_shown_item() -> None:
    # nobody is named and the sentence precedes any named one: only item 1005 backs "in stock"
    r = ground_reply("Everything here is in stock. [item:1005] is a polished set.", ITEMS)
    assert r.text == "Linen Co-ord Set is a polished set."
    only = ground_reply("Everything here is in stock.", [ITEMS[4]])
    assert only.text == "Everything here is in stock."


def test_all_sentences_dropped_serves_the_default_template() -> None:
    r = ground_reply("It is 30% off.", ITEMS)
    assert r.replaced and not r.failed_closed
    assert r.text.startswith(GROUNDED_FALLBACK_PREFIX) and "Riviera Cotton Sundress" in r.text
    assert "all_dropped" in r.flags


def test_all_sentences_dropped_serves_the_caller_template() -> None:
    r = ground_reply("It is 30% off.", ITEMS, fallback="TEMPLATE")
    assert r.text == "TEMPLATE" and r.replaced


def test_validator_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(gr, "check", boom)
    r = ground_reply("[item:1001] is 30% off.", ITEMS, fallback="TEMPLATE")
    assert r == GroundResult("TEMPLATE", ["validator_error"], replaced=True, failed_closed=True)
    # and with the default template the raw text must still not be served
    r2 = ground_reply("[item:1001] is 30% off.", ITEMS)
    assert "30%" not in r2.text and r2.failed_closed


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_text_is_returned_untouched(blank: str) -> None:
    assert ground_reply(blank, ITEMS) == GroundResult(blank)


def test_user_budget_echo_is_not_a_price_claim() -> None:
    reply = "These stay under ₹2000 as you asked: [item:1002] and [item:1003]."
    ok = ground_reply(reply, ITEMS, user_text="saree under ₹2000")
    assert ok.text.startswith("These stay under ₹2000")
    bad = ground_reply(reply, ITEMS, user_text="saree under ₹3000")
    assert "₹2000" not in bad.text


def test_extra_amounts_back_a_code_computed_figure() -> None:
    reply = "[item:1002] and [item:1003] come to ₹2,798 in total."
    assert ground_reply(reply, ITEMS).text.endswith("in total.") is True  # sum of two items
    assert "₹4,200" not in ground_reply("Your look totals ₹4,200.", ITEMS).text
    assert ground_reply("Your look totals ₹4,200.", ITEMS, extra_amounts=[4200]).text == (
        "Your look totals ₹4,200."
    )


@pytest.mark.parametrize("value", ["0", "false", "OFF", " no "])
def test_kill_switch_off_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(gr.KILL_SWITCH_ENV, value)
    assert grounded_rendering_enabled() is False


@pytest.mark.parametrize("value", [None, "1", "true", "on", ""])
def test_kill_switch_default_and_on_values(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(gr.KILL_SWITCH_ENV, raising=False)
    else:
        monkeypatch.setenv(gr.KILL_SWITCH_ENV, value)
    assert grounded_rendering_enabled() is True


def test_add_id_rules_goes_before_the_closing_instruction() -> None:
    out = add_id_rules("Intro.\n\nWrite your response now.")
    assert out.index("[item:ID]") < out.index("Write your response now.")
    assert add_id_rules("no marker").endswith(gr.RESPOND_ID_RULES)


# ---------------------------------------------------------------------------------- wiring


class _LLM:
    """Fake LLM: router JSON for router prompts, a fixed reply for the respond prompt."""

    model = "fake"

    def __init__(self, reply: str, chunks: list[str] | None = None) -> None:
        self.reply, self.chunks = reply, chunks
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.prompts.append(prompt)
        if "fashion shopping assistant" in prompt:
            return self.reply
        return json.dumps({"action": "respond"})

    def generate_stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        yield from (self.chunks or [self.reply])

    def chat(self, messages: list[dict], **kw: Any) -> str:
        return json.dumps({"action": "respond"})

    def chat_stream(self, messages: list[dict], **kw: Any) -> Iterator[str]:
        yield json.dumps({"action": "respond"})


def _respond_state(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "tell me about these"}],
        "user_query": "tell me about these", "current_plan": None, "tool_calls": [],
        "retrieved_items": items, "filters": {}, "final_answer": None, "iteration": 3,
        "new_items_this_turn": False, "out_of_catalogue": False, "excluded_colours": None,
        "anchor_article_id": None, "outfit_rationale": None, "outfit_variants": None,
        "_memory": None,
    }


def _run_respond_node(llm: _LLM) -> str:
    from src.agents.graph import build_graph

    config = {"agent": {"max_iterations": 3}, "retrieval": {"final_k": 5}}
    # only the prompt-vocabulary builder reads the frame; the retriever is never reached because
    # iteration == max_iterations sends the router straight to respond
    df = pd.DataFrame({"colour_group_name": ["Blue"], "product_type_name": ["Dress"]})
    agent = build_graph(None, df, llm, config, streaming_mode=False)  # type: ignore[arg-type]
    return str(agent.invoke(_respond_state(ITEMS))["final_answer"])


def test_respond_node_renders_ids_and_strips_unbacked_claims() -> None:
    llm = _LLM("[item:1001] at [price:1001] is lovely. It is 30% off today.")
    answer = _run_respond_node(llm)
    assert answer == "Riviera Cotton Sundress at ₹2,999 is lovely."
    respond_prompts = [p for p in llm.prompts if "fashion shopping assistant" in p]
    assert respond_prompts and "[item:ID]" in respond_prompts[0] and "id: 1001" in respond_prompts[0]


def test_respond_node_kill_switch_restores_the_old_prompt_and_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(gr.KILL_SWITCH_ENV, "0")
    llm = _LLM("Riviera Cotton Sundress is lovely.")
    assert _run_respond_node(llm) == "Riviera Cotton Sundress is lovely."
    prompt = next(p for p in llm.prompts if "fashion shopping assistant" in p)
    assert "[item:ID]" not in prompt and "id: 1001" not in prompt


_WS_ITEMS = [{
    "article_id": "1", "prod_name": "Test Jacket", "display_name": "Test Jacket (Blue)",
    "colour": "Blue", "product_type": "Jacket", "department": "Ladieswear",
    "image_url": "https://example.com/1.jpg", "detail_desc": "A jacket.", "score": 0.9,
    "price_inr": 1999,
}]


class _WsAgent:
    def invoke(self, state: dict, **kwargs: Any) -> dict[str, Any]:
        return {
            "retrieved_items": _WS_ITEMS, "filters": {}, "final_answer": None, "iteration": 1,
            "tool_calls": [{"router_decision": {"action": "search", "query": "jacket"}},
                           {"search": {"query": "jacket", "filters": {}}}],
            "current_plan": json.dumps({"action": "pending_respond", "prompt": "Describe."}),
            "new_items_this_turn": True, "out_of_catalogue": False, "excluded_colours": None,
            "messages": state.get("messages", []),
        }


@pytest.fixture
def _ws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_session_store", InMemorySessionStore())
    monkeypatch.setattr(deps, "_config", {
        "agent": {"max_iterations": 3}, "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    })
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.setattr(deps, "get_agent_factory", lambda: (lambda memory, streaming=False: _WsAgent()))


def _ws_turn(llm: _LLM) -> tuple[list[str], str]:
    deps._llm = llm
    client = TestClient(app, raise_server_exceptions=True)
    frames: list[str] = []
    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": "show me jackets"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "token":
                frames.append(msg["text"])
            elif msg["type"] == "done":
                return frames, msg["final_state"]["response"]
            elif msg["type"] == "error":
                pytest.fail(f"unexpected error message: {msg}")


@pytest.mark.usefixtures("_ws")
def test_ws_stream_is_buffered_rendered_and_stripped() -> None:
    llm = _LLM("", chunks=["Love [item:1] ", "at [price:1]. ", "It is 40% off."])
    frames, final = _ws_turn(llm)
    assert final == "Love Test Jacket (Blue) at ₹1,999."
    assert "".join(frames) == final and len(frames) == 1


@pytest.mark.usefixtures("_ws")
def test_ws_clean_stream_replays_the_original_frames() -> None:
    frames, final = _ws_turn(_LLM("", chunks=["Lovely ", "jacket."]))
    assert frames == ["Lovely ", "jacket."] and final == "Lovely jacket."


@pytest.mark.usefixtures("_ws")
def test_ws_kill_switch_off_streams_raw_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gr.KILL_SWITCH_ENV, "0")
    frames, final = _ws_turn(_LLM("", chunks=["It is ", "40% off."]))
    assert frames == ["It is ", "40% off."] and final == "It is 40% off."


@pytest.mark.usefixtures("_ws")
def test_ws_unknown_id_everywhere_serves_the_grounded_template() -> None:
    frames, final = _ws_turn(_LLM("", chunks=["Try [item:424242] today."]))
    assert final.startswith(GROUNDED_FALLBACK_PREFIX) and "Test Jacket (Blue) (₹1,999)" in final
    assert "".join(frames) == final


class _EmptyStreamRetryLLM(_LLM):
    """Streams nothing; the non-streaming retry (llm.generate) returns the raw text."""

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        return self.reply

    def generate_stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        return iter(())


@pytest.mark.usefixtures("_ws")
def test_ws_empty_stream_retry_text_is_rendered_and_stripped() -> None:
    llm = _EmptyStreamRetryLLM("Love [item:1] at [price:1]. It is 40% off.")
    frames, final = _ws_turn(llm)
    assert final == "Love Test Jacket (Blue) at ₹1,999."
    assert "".join(frames) == final


class _RationaleLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        return json.dumps([self.text])


def test_rationale_last_gate_strips_a_size_claim_the_keyword_scrub_misses() -> None:
    look = {"occasion": "casual", "gender": "women", "seed_item": dict(ITEMS[0], _slot="dress"),
            "complements": []}
    text = "The blue dress is a calm choice. It comes in sizes S, M, L."
    out = generate_rationales([look], _RationaleLLM(text), occasion="casual", gender="women")
    assert out == ["The blue dress is a calm choice."]
