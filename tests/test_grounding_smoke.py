"""CI smoke for the W10 grounding eval: 10 prompts, mocked LLM, real routes (REST + WS).

Gate: grounded mock templates must yield zero checker violations, and every ungrounded template
must be caught (on REST, validate_response may already have scrubbed it, which also counts).
Runs offline against evals/fixtures/grounding_catalogue.json; needs no index and no network.
"""
from __future__ import annotations

from collections import Counter

import pytest

from evals.grounding import run as harness
from evals.grounding.mock_llm import MockLLM


def _run(mode: str) -> tuple[list[dict], dict]:
    llm = MockLLM(mode=mode)
    cases = harness.select(harness.load_cases(), smoke=True, sample=None)
    with harness.stub_env(llm):
        results = harness.run_all(cases, llm, ["rest", "ws"])
    return results, harness.mock_gate(results, mode)


def test_fixture_shape() -> None:
    cases = harness.load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)) == 150
    by_set = Counter(c["set"] for c in cases)
    assert by_set == {"regular": 100, "adversarial": 50}
    adv = Counter(c["category"] for c in cases if c["set"] == "adversarial")
    assert adv == {"discount": 10, "free_shipping": 10, "wrong_item_price": 10,
                   "out_of_stock": 10, "size_not_offered": 10}
    reg = Counter(c["category"] for c in cases if c["set"] == "regular")
    assert len(reg) == 10 and set(reg.values()) == {10}
    assert all(c["prompt"].strip() for c in cases)
    assert len({c["prompt"] for c in cases}) == 150, "duplicate prompts"


def test_smoke_subset_is_fixed_and_balanced() -> None:
    subset = harness.select(harness.load_cases(), smoke=True, sample=None)
    assert [c["id"] for c in subset] == list(harness.SMOKE_IDS)
    assert Counter(c["set"] for c in subset) == {"regular": 5, "adversarial": 5}
    assert len({c["category"] for c in subset if c["set"] == "adversarial"}) == 5


def test_stratified_sample_is_deterministic_and_stratified() -> None:
    cases = harness.load_cases()
    a = harness.select(cases, smoke=False, sample="10,10")
    b = harness.select(cases, smoke=False, sample="10,10")
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert Counter(c["set"] for c in a) == {"regular": 10, "adversarial": 10}
    assert len({c["category"] for c in a if c["set"] == "regular"}) == 10
    assert Counter(c["category"] for c in a if c["set"] == "adversarial") == {
        "discount": 2, "free_shipping": 2, "wrong_item_price": 2, "out_of_stock": 2,
        "size_not_offered": 2}


def test_grounded_templates_produce_no_violations() -> None:
    results, gate = _run("grounded")
    assert gate["ok"], gate
    assert gate["grounded_runs"] == 20 and gate["injected_runs"] == 0
    assert all(r["http_ok"] for r in results)
    assert sum(len(r["violations"]) for r in results) == 0


def test_ungrounded_templates_are_all_caught_on_ws() -> None:
    results, gate = _run("ungrounded")
    assert gate["ok"], gate
    assert gate["injected_missed"] == []
    ws = [r for r in results if r["path"] == "ws"]
    assert len(ws) == 10 and all(r["violations"] for r in ws), "WS has no scrubber: all must flag"


def test_rest_path_scrubber_is_visible_in_the_gate() -> None:
    # REST runs validate_response; the gate must show what it neutralised vs what the checker caught.
    _, gate = _run("ungrounded")
    assert gate["injected_flagged"] + gate["injected_neutralised_by_validate_response"] >= 20


def test_gate_fails_if_the_checker_is_blinded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "check", lambda *a, **k: [])
    _, gate = _run("ungrounded")
    assert gate["ok"] is False and gate["injected_missed"]


def test_gate_fails_if_grounded_text_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.grounding.checker import Violation

    monkeypatch.setattr(harness, "check", lambda *a, **k: [Violation("price", "x", 0, 1, "forced")])
    _, gate = _run("grounded")
    assert gate["ok"] is False and gate["grounded_runs_flagged"]


def test_mixed_mode_is_deterministic_per_prompt() -> None:
    def injected_map() -> dict[tuple[str, str], bool]:
        results, _ = _run("mixed")
        return {(r["id"], r["path"]): r["mock"]["injected"] for r in results}

    first = injected_map()
    assert first == injected_map()
    for cid in {k[0] for k in first}:
        assert first[(cid, "rest")] == first[(cid, "ws")], "REST and WS must see the same template"
    assert 0 < sum(first.values()) < len(first)
