"""The frozen retrieval-eval queries must not duplicate the strict-gold label-set queries."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from evals.retrieval import make_queries as mq
from evals.retrieval.common import load_cases

GOLD = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "strict_gold_queries.yaml"
# words that carry no constraint; digits (budgets) never match [a-z]+ so budget-only differences
# do not make two queries "different"
_STOP = {"for", "a", "an", "the", "under", "to", "of", "and", "in", "with", "women", "men"}


def _content(text: str) -> frozenset[str]:
    t = text.lower().replace("women's", "").replace("men's", "")
    return frozenset(w for w in re.findall(r"[a-z]+", t) if w not in _STOP)


def test_no_fixture_equals_a_strict_gold_query_up_to_budget_and_stopwords() -> None:
    gold = {
        _content(g["query"]): g["query"] for g in yaml.safe_load(GOLD.read_text("utf-8"))["queries"]
    }
    clashes = [
        (c["id"], c["text"], gold[_content(c["text"])])
        for c in load_cases()
        if _content(c["text"]) in gold
    ]
    assert clashes == []


def test_the_check_would_have_caught_the_old_q15() -> None:
    gold = {_content(g["query"]) for g in yaml.safe_load(GOLD.read_text("utf-8"))["queries"]}
    assert _content("green outfit for mehendi under ₹1000") in gold


def test_q15_override_keeps_its_stratum_and_changes_nothing_else() -> None:
    cases = {c["id"]: c for c in mq.build_cases()}
    q15 = cases["q15"]
    assert (q15["occasion"], q15["gender"], q15["budget_inr"], q15["language_style"]) == (
        "mehendi",
        "either",
        1000,
        "plain_en",
    )
    assert q15["constraints"]["colour"] == "yellow" and "yellow" in q15["text"]
    # the override sits after the rng draw: with it removed, every OTHER case is byte-identical
    saved = dict(mq.COLOUR_OVERRIDES)
    try:
        mq.COLOUR_OVERRIDES.clear()
        plain = {c["id"]: c for c in mq.build_cases()}
    finally:
        mq.COLOUR_OVERRIDES.update(saved)
    assert plain["q15"]["constraints"]["colour"] == "green"
    assert all(cases[k] == plain[k] for k in cases if k != "q15")
