"""Special-case fixture (evals/fixtures/rerank_special_cases.json): the deterministic rules clear
every violation that the unfiltered order carries, keep 5 cards, and leave the control alone."""
from __future__ import annotations

import json

import pytest

from scripts import rerank_rules_report as rep


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return rep.evaluate(json.loads(rep.FIXTURE.read_text("utf-8"))["cases"])


def test_fixture_shape(rows: list[dict]) -> None:
    assert len(rows) == 10
    assert all(r["new_rules"]["n"] == 5 for r in rows)  # enough compliant candidates in each pool


def test_unfiltered_order_carries_violations_everywhere_but_the_control(rows: list[dict]) -> None:
    by_id = {r["id"]: r for r in rows}
    assert by_id["s07"]["no_rules"]["violations"] == 0  # monsoon: the old prompt had no rule
    assert all(r["no_rules"]["violations"] > 0 for r in rows if r["id"] != "s07")


def test_new_rules_leave_zero_violations(rows: list[dict]) -> None:
    assert [r["new_rules"]["violations"] for r in rows] == [0] * 10


def test_chip_pools_are_the_eight_landing_chips_and_go_to_the_llm() -> None:
    from scripts import rerank_gate as g

    chips = g.load_pools(g.DEFAULT_CHIPS)
    assert [c["text"] for c in chips] == [
        "Sangeet outfit under ₹3,000", "Haldi kurta set", "Wedding guest saree",
        "Office wear under ₹1,500", "Men's kurta for a wedding", "sherwani for groom",
        "lehenga for sangeet", "men's blazer for the office"]
    assert all(len(c["items"]) > g.TOP_K for c in chips)  # > top_k: the reranker is called


def test_kurta_pyjama_set_is_ethnic_wear_not_nightwear() -> None:
    from src.agents.rerank_filters import rule_violations

    kp = {"display_name": "Men's Red Bandhni Kurta Pyjama Set", "colour": "Red",
          "product_type": "kurta"}
    assert rule_violations("kurta set for sangeet", kp) == []
    assert rule_violations("kurta for haldi", {**kp, "display_name": "MLS KURTA PAJAMA"}) == []
    assert rule_violations("haldi outfit", {**kp, "display_name": "Cotton Pyjama Night Suit"})


def test_gate_failure_bar_counts_only_failures() -> None:
    from scripts import rerank_gate as g

    def row(i: int, outcome: str) -> dict:
        return {"query_id": f"q{i:02d}", "text": "t", "final_ids": ["a"], "outcome": outcome,
                "prompt_tokens": 600, "completion_tokens": 100, "latency_ms": 900, "path": "primary"}

    ok_rows = [row(i, "ok") for i in range(15)] + [row(15, "partial"), row(16, "empty_selection")]
    eval_c = {r["query_id"]: {"final_ids": ["a"]} for r in ok_rows}
    cons = {r["query_id"]: {"a": 2.0, "b": 1.0} for r in ok_rows}
    assert g.summarize(ok_rows, eval_c, cons, 0)["pass_failure_bar"] is True  # partial/empty: valid
    bad = ok_rows[:15] + [row(15, "parse_error"), row(16, "length")]  # 2 of 17 = 11.8% > 5%
    s = g.summarize(bad, eval_c, cons, 0)
    assert s["pass_failure_bar"] is False and round(s["failure_rate"], 3) == 0.118
    twenty = [row(i, "ok") for i in range(19)] + [row(19, "http_error")]  # exactly 5%: passes
    eval20 = {r["query_id"]: {"final_ids": ["a"]} for r in twenty}
    cons20 = {r["query_id"]: {"a": 2.0, "b": 1.0} for r in twenty}
    assert g.summarize(twenty, eval20, cons20, 0)["pass_failure_bar"] is True


def test_old_master_code_only_had_the_diversity_swap(rows: list[dict]) -> None:
    # the old code path fixes none of the beach/occasion/season violations
    assert all(r["old_master_code"]["violations"] == r["no_rules"]["violations"] for r in rows)
