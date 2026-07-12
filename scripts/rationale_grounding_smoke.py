"""Standalone in-process proof: rationale generation is grounded in the user's
own words, inherited budget, and owned-anchor status.

Runs a 2-turn conversation through the actual compiled agent graph (no
network hop — same in-process invoke/session-dict plumbing as
api/routes/chat.py::_build_invoke_state / _persist_result, mirrored here the
same way scripts/budget_persistence_smoke.py does):

  turn 1: "kurta under 3000"  -> sets budget_max_inr=3000 (search path)
  turn 2 (simulated post-image-upload): "Style this <owned garment> for a
          sangeet" -> deterministic style-anchor route. The session's
          retrieved_items/anchor_article_id/anchor_is_owned are set exactly as
          image_style.py would after a photo upload, so this exercises the
          SAME "owned anchor + inherited budget + occasion-in-own-words"
          fact-sheet path a real multi-turn conversation would hit.

Part (a): spies on src.agents.graph.generate_rationales (module-level
rebind — the outfit_node closure resolves this name dynamically at call
time, so patching the module attribute is sufficient, no LLM/network
patching needed) to capture the kwargs graph.py passes through, then
independently rebuilds the fact-sheet via rationale.build_fact_sheet with
those same kwargs and asserts it contains budget_inr, anchor_is_owned=True,
and a user_context snippet reflecting the user's own words ("sangeet").

Part (b): confirms the grounding gate still rejects a fabricated attribute
(an invented colour not present in the look) even with the new fields
present, and confirms the new "budget" exemption does NOT open the door to
other price words (e.g. "expensive") which must still be scrubbed.

No live LLM call is required (build_graph's deterministic router path is
exercised; the two rationale-generation LLM calls that DO occur use a
canned-response MockLLM, so the grounding fallback path or JSON-parse path
runs exactly like tests/test_grounding_rationale.py's fixtures — no Ollama
dependency).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["AGENT_LOOP_FAST_PATH"] = "true"

import pandas as pd

import src.agents.graph as graph_mod
from src.agents.grounding import validate_rationale
from src.agents.outfit.rationale import build_fact_sheet
from src.catalogue.loader import load_config
from src.memory.conversation import ConversationMemory
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "unified"

_ANCHOR_ARTICLE_ID = "17048614"
_ANCHOR_PROD_NAME = "Khushal K Women Black Ethnic Motifs Printed Kurta with Palazzos & With Dupatta"


class MockLLM:
    """Cycles through canned responses; the deterministic IntentParser router
    path used here never actually calls the LLM for routing — only
    generate_rationales's single batched call reaches llm.generate().
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str | None = None, **kwargs: object) -> str:
        return self._next()

    def generate_stream(
        self, prompt: str, system: str | None = None, **kwargs: object
    ) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs: object) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs: object) -> Iterator[str]:
        yield self._next()


def _run_turn(agent: object, session: dict, query: str) -> dict:
    """Invoke the graph for one turn and persist the result back into session.

    Mirrors api/routes/chat.py's _build_invoke_state / _persist_result exactly.
    """
    invoke_state = {
        "messages": session["messages"] + [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": session["retrieved_items"],
        "filters": session["filters"],
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": session.get("excluded_colours"),
        "anchor_article_id": session.get("anchor_article_id"),
        "anchor_is_owned": session.get("anchor_is_owned", False),
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": session["_memory"],
    }
    result = agent.invoke(invoke_state)
    session["messages"] = result.get("messages", session["messages"])
    session["retrieved_items"] = result.get("retrieved_items", session["retrieved_items"])
    session["filters"] = result.get("filters", session["filters"])
    if result.get("excluded_colours") is not None:
        session["excluded_colours"] = result["excluded_colours"]
    return result


def main() -> int:
    config = load_config()
    catalogue_df = pd.read_parquet(SAVE_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, SAVE_DIR)
    sparse = SparseRetriever.load(config, SAVE_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    # Canned response is NOT a valid JSON rationale list — this deliberately
    # exercises the template-fallback path (no Ollama dependency) while still
    # letting us inspect the kwargs graph.py passed INTO generate_rationales,
    # which is what this proof is actually about.
    llm = MockLLM(["ok"] * 20)
    session = {
        "messages": [],
        "retrieved_items": [],
        "filters": {},
        "excluded_colours": None,
        "anchor_article_id": None,
        "anchor_is_owned": False,
        "_memory": ConversationMemory(llm, config),
    }
    agent = graph_mod.build_graph(retriever, catalogue_df, llm, config)

    # ── Spy on generate_rationales (module-level rebind) ────────────────────
    _orig_generate_rationales = graph_mod.generate_rationales
    captured_calls: list[dict] = []

    def _spy(*args: object, **kwargs: object) -> list[str]:
        captured_calls.append(kwargs)
        return _orig_generate_rationales(*args, **kwargs)

    graph_mod.generate_rationales = _spy

    ok = True

    # Turn 1: sets budget_max_inr=3000 via the search path.
    result1 = _run_turn(agent, session, "kurta under 3000")
    price_max = result1.get("filters", {}).get("price_max")
    print(f"turn 1: query='kurta under 3000' -> filters.price_max={price_max}")

    # Simulate a photo upload BETWEEN turns (image_style.py's own persistence
    # shape) — the owned anchor becomes the session's seed item.
    session["retrieved_items"] = [
        {
            "article_id": _ANCHOR_ARTICLE_ID,
            "prod_name": _ANCHOR_PROD_NAME,
            "display_name": _ANCHOR_PROD_NAME,
            "colour": "black",
            "product_type": "kurta",
            "gender": "women",
            "_role": "seed",
            "_owned": True,
        },
    ]
    session["anchor_article_id"] = _ANCHOR_ARTICLE_ID
    session["anchor_is_owned"] = True

    # Turn 2: "Style this <owned garment> for a sangeet" — deterministic
    # style-anchor route (resolves the owned anchor by name substring match),
    # occasion reconstructed from THIS turn's own text ("sangeet").
    turn2_query = f"Style this {_ANCHOR_PROD_NAME} for a sangeet"
    result2 = _run_turn(agent, session, turn2_query)

    graph_mod.generate_rationales = _orig_generate_rationales  # restore

    print(f"\nturn 2: query={turn2_query!r}")
    print(f"  look_id={result2.get('look_id')!r}  occasion={result2.get('occasion')!r}")
    print(f"  budget_total_inr={result2.get('budget_total_inr')!r}")
    seed_items = [
        it for it in result2.get("retrieved_items", [])
        if it.get("article_id") == _ANCHOR_ARTICLE_ID
    ]
    print(f"  owned anchor still seed & _owned=True: {bool(seed_items) and seed_items[0].get('_owned') is True}")

    print(f"\ngenerate_rationales was called {len(captured_calls)} time(s); captured kwargs:")
    for i, kw in enumerate(captured_calls):
        printable = {k: v for k, v in kw.items() if k != "partner_context" or v}
        print(f"  call {i}: {printable}")

    # ── Part (a): assert the turn-2 outfit call carried the right facts ────
    outfit_calls = [
        kw for kw in captured_calls
        if kw.get("occasion") == "sangeet" or (kw.get("user_context") or "").find("sangeet") != -1
    ]
    assert outfit_calls, f"expected a generate_rationales call for the sangeet turn, got {captured_calls}"
    call_kwargs = outfit_calls[-1]

    budget_ok = call_kwargs.get("budget_inr") is not None and 2900 <= call_kwargs["budget_inr"] <= 3100
    owned_ok = call_kwargs.get("anchor_is_owned") is True
    user_ctx_ok = "sangeet" in (call_kwargs.get("user_context") or "").lower()

    print("\n[assert] budget_inr inherited from turn 1 ->", "PASS" if budget_ok else f"FAIL ({call_kwargs.get('budget_inr')!r})")
    print("[assert] anchor_is_owned=True ->", "PASS" if owned_ok else f"FAIL ({call_kwargs.get('anchor_is_owned')!r})")
    print("[assert] user_context contains 'sangeet' ->", "PASS" if user_ctx_ok else f"FAIL ({call_kwargs.get('user_context')!r})")
    ok &= budget_ok and owned_ok and user_ctx_ok

    # Rebuild the fact-sheet exactly as generate_rationales would have, to show
    # the literal dict the LLM prompt receives.
    _fake_look = {
        "seed_item": {"colour": "black", "product_type": "kurta"},
        "complements": [],
        "occasion": "sangeet",
        "gender": "women",
    }
    fact_sheet = build_fact_sheet(
        _fake_look,
        user_context=call_kwargs.get("user_context"),
        budget_inr=call_kwargs.get("budget_inr"),
        anchor_is_owned=call_kwargs.get("anchor_is_owned", False),
    )
    print("\nfact_sheet built from captured turn-2 kwargs:")
    print(json.dumps(fact_sheet, indent=2))
    fact_sheet_ok = (
        fact_sheet.get("budget_inr") == call_kwargs.get("budget_inr")
        and fact_sheet.get("anchor_is_owned") is True
        and "sangeet" in fact_sheet.get("user_context", "").lower()
    )
    print("[assert] fact_sheet contains budget_inr/anchor_is_owned/user_context ->",
          "PASS" if fact_sheet_ok else "FAIL")
    ok &= fact_sheet_ok

    # ── Part (b): grounding gate still rejects fabricated attributes ───────
    look_items = [_fake_look["seed_item"]]
    # (b1) Invented colour ("mustard") not present in the look — must be dropped,
    # even though budget_inr is now present in this call.
    fabricated_text = "Pair it with a mustard dupatta to finish the sangeet look."
    cleaned_b1, flags_b1 = validate_rationale(
        fabricated_text, look_items, "sangeet", budget_inr=call_kwargs.get("budget_inr")
    )
    ungrounded_hit = any("ungrounded_colour:mustard" in f for f in flags_b1)
    print("\n[assert] fabricated colour 'mustard' still flagged as ungrounded ->",
          "PASS" if ungrounded_hit else f"FAIL (flags={flags_b1})")
    ok &= ungrounded_hit

    # (b2) "budget" itself is now exempted (grounded via budget_inr)...
    budget_text = "This black kurta look was kept within your budget for the sangeet."
    cleaned_b2, flags_b2 = validate_rationale(
        budget_text, look_items, "sangeet", budget_inr=call_kwargs.get("budget_inr")
    )
    budget_word_not_flagged = not any(f.startswith("price:") for f in flags_b2)
    print("[assert] 'budget' mention NOT scrubbed when budget_inr is known ->",
          "PASS" if budget_word_not_flagged else f"FAIL (flags={flags_b2})")
    ok &= budget_word_not_flagged

    # ...but OTHER price words ("expensive") are still scrubbed even when
    # budget_inr is known — only the literal "budget" keyword is exempted.
    expensive_text = "This black kurta is an expensive choice for the sangeet."
    cleaned_b3, flags_b3 = validate_rationale(
        expensive_text, look_items, "sangeet", budget_inr=call_kwargs.get("budget_inr")
    )
    expensive_still_flagged = any(f.startswith("price:") for f in flags_b3)
    print("[assert] 'expensive' still scrubbed even with budget_inr known ->",
          "PASS" if expensive_still_flagged else f"FAIL (flags={flags_b3})")
    ok &= expensive_still_flagged

    print("\nVERDICT:", "ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
