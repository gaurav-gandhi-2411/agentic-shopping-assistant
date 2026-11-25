"""Store/brand names must never reach the judges (mock only, no network)."""

from __future__ import annotations

import pytest

from evals.retrieval import judge as jd
from evals.retrieval import scrub as sc
from evals.retrieval.common import CANDIDATE_DIR, load_candidates, load_cases
from src.config.store_registry import STORE_REGISTRY


@pytest.mark.parametrize("slug", sorted(STORE_REGISTRY))
def test_every_registry_store_is_scrubbed_from_title_and_description(slug: str) -> None:
    rec = STORE_REGISTRY[slug]
    cand = {
        "article_id": "x1",
        "display_name": f"{rec.display_name} Green Kurta ({slug.upper()})",
        "detail_desc": f"By {slug} and {rec.display_name.upper()}'s own studio. Soft cotton.",
        "price_inr": 999.0,
        "product_type": "kurta",
        "colour": "Green",
    }
    line = jd._fmt_candidate(1, cand)
    assert sc.leaks(line) == [], line
    assert "Green Kurta" in line and "Soft cotton" in line  # only the brand is removed


def test_scrub_keeps_other_text_untouched() -> None:
    text = "Type : Kurta Only Brand Color : Yellow, Libas festive"
    assert sc.scrub_stores(text) == "Type : Kurta Only Brand Color : Yellow, festive"


def test_scrub_respects_word_boundaries() -> None:
    # "libas" glued to letters is not a store mention; leave it alone
    assert sc.scrub_stores("Libaskurta lovers") == "Libaskurta lovers"


def test_scrub_handles_spacing_and_hyphen_variants() -> None:
    for variant in ("Global Republic", "GlobalRepublic", "global-republic", "5 Elements"):
        assert sc.scrub_stores(f"{variant} dress") == "dress"


def test_judge_prompt_on_frozen_candidates_has_zero_store_leaks() -> None:
    """The real measurement: every candidate line of every judge prompt, all queries."""
    bad = n = 0
    for c in load_cases():
        items = load_candidates(c["id"], CANDIDATE_DIR)["items"]
        if not items:
            continue
        p = jd.build_judge_prompt(c, items, 42)
        cand_lines = [ln for ln in p.user.splitlines() if ln[:1].isdigit()]
        assert len(cand_lines) == len(items)
        b, t = sc.leak_rate(cand_lines)
        bad, n = bad + b, n + t
    assert n > 900 and bad == 0


def test_detector_is_not_vacuous_the_unscrubbed_rate_is_high() -> None:
    items = [it for c in load_cases() for it in load_candidates(c["id"], CANDIDATE_DIR)["items"]]
    before = sc.leak_rate(sc.judge_lines(items, scrub=False))
    after = sc.leak_rate(sc.judge_lines(items, scrub=True))
    assert before[0] / before[1] > 0.10 and after[0] == 0


def test_description_is_scrubbed_before_it_is_truncated() -> None:
    cand = {"article_id": "x", "display_name": "Kurta", "detail_desc": "Libas " + "ab " * 80}
    desc = jd._fmt_candidate(1, cand).rsplit(" | ", 1)[1]
    assert len(desc) == jd.SNIPPET_CHARS  # the removed brand does not eat into the snippet budget


def test_leak_detector_catches_respaced_forms() -> None:
    assert sc.leaks("a v a s t r a m a y")
    assert not sc.leaks("plain green kurta")
