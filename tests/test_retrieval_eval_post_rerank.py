"""Post-rerank replay: the card count can change with WHICH five the reranker picked."""

from __future__ import annotations

from evals.retrieval.common import CANDIDATE_DIR, load_candidates, load_cases
from evals.retrieval.post_rerank import apply_post_rerank


def _case(qid: str) -> tuple[dict, list[dict]]:
    c = next(c for c in load_cases() if c["id"] == qid)
    return c, load_candidates(qid, CANDIDATE_DIR)["items"]


def _gender(c: dict) -> str:
    return "unisex" if c["gender"] == "either" else c["gender"]


def test_blazer_query_shows_4_or_5_cards_depending_on_the_pick() -> None:
    """The production blazer 4-vs-5 shape: same pool, two different reranker top-5s."""
    c, pool = _case("q30")  # "office ke liye blazer chahiye, 5000 se kam mein"
    first = apply_post_rerank(c["text"], pool[:5], pool, _gender(c))
    assert len(first) == 4  # a post-rerank gate dropped one card and nothing refills it
    survivors = list(first)
    swap_in = next(
        it
        for it in pool[5:]
        if len(apply_post_rerank(c["text"], survivors + [it], pool, _gender(c))) == 5
    )
    other_pick = survivors + [swap_in]  # a different (stochastic) rerank choice
    assert len(apply_post_rerank(c["text"], other_pick, pool, _gender(c))) == 5


def test_dedup_by_name_and_colour_is_backfilled_from_the_pool() -> None:
    c, pool = _case("q02")
    dup = dict(pool[0], article_id="dup-of-first")  # same prod_name + colour as pool[0]
    cards = apply_post_rerank(c["text"], [pool[0], dup, *pool[1:4]], pool + [dup], _gender(c))
    assert "dup-of-first" not in {it["article_id"] for it in cards}
    assert len(cards) <= 5


def test_never_more_than_top_k_and_never_adds_unseen_items_beyond_the_pool() -> None:
    for qid in ("q02", "q05", "q10", "q24"):
        c, pool = _case(qid)
        cards = apply_post_rerank(c["text"], pool[:5], pool, _gender(c))
        assert len(cards) <= 5
        assert {it["article_id"] for it in cards} <= {it["article_id"] for it in pool}
