"""src/agents/rerank_filters.py: one test class per old prompt rule (beach / occasion / season /
colour constraint / date-night / diversity), the never-empty guard, and rerank() integration."""
from __future__ import annotations

from typing import Any

import pytest

from src.agents.rerank_filters import apply_rules, named_colours, rule_violations
from src.agents.reranker import rerank
from src.llm.client import LLMCallError


def it(i: int, name: str, colour: str | None = "Black", ptype: str = "", **kw: Any) -> dict:
    return {"article_id": f"a{i}", "display_name": name, "colour": colour, "product_type": ptype,
            "price_inr": 1000.0, **kw}


def ids(items: list[dict]) -> list[str]:
    return [x["article_id"] for x in items]


class TestOccasionSleepSwim:
    @pytest.mark.parametrize("q", ["haldi outfit", "wedding guest saree", "sangeet look",
                                   "date night dress", "office wear", "team meeting outfit"])
    def test_nightwear_and_swimwear_rejected(self, q: str) -> None:
        assert "occasion_sleep_swim" in rule_violations(q, it(1, "Cotton Pyjama Set"))
        assert "occasion_sleep_swim" in rule_violations(q, it(2, "Floral Bikini Top"))

    def test_not_applied_to_casual_queries(self) -> None:
        assert rule_violations("night suit for women", it(1, "Cotton Night Suit")) == []

    def test_word_boundary_not_substring(self) -> None:
        assert "occasion_sleep_swim" not in rule_violations("wedding", it(1, "Brassy Embroidered Lehenga"))


class TestDateNightRules:
    Q = "date night dress"

    @pytest.mark.parametrize("colour", ["Beige", "Light Beige", "Camel", "Brown", "Light Brown",
                                        "Pastel Pink", "Peach", "Lavender"])
    def test_casual_neutral_and_pastel_colours_rejected(self, colour: str) -> None:
        assert "date_night_casual_colour" in rule_violations(self.Q, it(1, "Dress", colour))

    def test_rich_colours_pass(self) -> None:
        for c in ("Black", "Red", "Burgundy", "Emerald Green", "Wine", None):
            assert rule_violations(self.Q, it(1, "Satin Dress", c)) == []

    def test_a_colour_the_user_asked_for_is_not_casual(self) -> None:
        assert rule_violations("brown dress for date night", it(1, "Dress", "Brown")) == []
        assert rule_violations("blue dress evening", it(1, "Dress", "Light Blue")) == []

    @pytest.mark.parametrize("name", ["Cable Knit Sweater", "Hooded Sweatshirt", "Cotton T-Shirt",
                                      "Graphic Tee", "Zip Hoodie"])
    def test_casual_types_rejected_unless_requested(self, name: str) -> None:
        assert "date_night_casual_type" in rule_violations(self.Q, it(1, name))
        assert "date_night_casual_type" not in rule_violations(
            f"{name.split()[-1].lower()} for date night", it(1, name))

    def test_wedding_and_cocktail_are_date_night_queries_too(self) -> None:
        for q in ("wedding guest dress", "cocktail dress", "evening gown"):
            assert "date_night_casual_colour" in rule_violations(q, it(1, "Dress", "Beige"))

    def test_sangeet_is_formal_but_not_a_colour_rule_query(self) -> None:
        assert rule_violations("sangeet lehenga", it(1, "Lehenga", "Beige")) == []


class TestBeachRules:
    Q = "beach vacation outfit"

    @pytest.mark.parametrize("desc", ["bodycon", "shift", "midi", "formal", "cocktail",
                                      "structured", "tailored"])
    def test_dress_styles_rejected(self, desc: str) -> None:
        assert "beach_dress_style" in rule_violations(self.Q, it(1, f"{desc.title()} Dress"))

    def test_style_found_in_description(self) -> None:
        d = it(1, "Printed Dress", detail_desc="A bodycon silhouette with ruching")
        assert "beach_dress_style" in rule_violations(self.Q, d)

    @pytest.mark.parametrize("name", ["Formal Blouse", "Wool Blazer", "Slim Trousers", "Blue Jeans",
                                      "Long Coat", "Chunky Knitwear", "Heavy Sweater"])
    def test_heavy_and_formal_items_rejected(self, name: str) -> None:
        assert "beach_heavy_item" in rule_violations(self.Q, it(1, name))

    def test_sand_colours_rejected_unless_swim_or_sundress(self) -> None:
        assert "beach_sand" in rule_violations(self.Q, it(1, "Cotton Top", "Beige"))
        assert "beach_sand" in rule_violations(self.Q, it(2, "Dress", "Sand"))
        assert "beach_sand" not in rule_violations(self.Q, it(3, "Bikini Set", "Beige"))
        assert "beach_sand" not in rule_violations(self.Q, it(4, "Linen Sundress", "Beige"))

    def test_casual_sundress_and_swimwear_pass(self) -> None:
        for n in ("White Linen Sundress", "Striped Bikini", "Cotton Cover-up", "Board Shorts"):
            assert rule_violations(self.Q, it(1, n, "White")) == []

    def test_preferred_beach_items_move_to_the_front_stably(self) -> None:
        pool = [it(1, "Cotton Top", "White"), it(2, "Bikini Set", "Red"), it(3, "Linen Tee", "Blue"),
                it(4, "Sundress", "Yellow")]
        shown, _ = apply_rules(self.Q, pool, 4)
        assert ids(shown) == ["a2", "a4", "a1", "a3"]


class TestSeasonRules:
    def test_winter_rejects_light_items(self) -> None:
        for n in ("Denim Shorts", "Bikini", "Leather Sandals", "Flip-Flops", "Swimsuit"):
            assert "winter_light_item" in rule_violations("winter wedding outfit", it(1, n))
        assert rule_violations("autumn look", it(1, "Wool Coat")) == []

    def test_summer_rejects_coats_and_heavy_knitwear(self) -> None:
        for n in ("Wool Coat", "Puffer Jacket", "Chunky Sweater", "Parka"):
            assert "summer_heavy_item" in rule_violations("summer party dress", it(1, n))
        assert rule_violations("summer dress", it(1, "Linen Dress")) == []

    def test_monsoon_has_no_rule(self) -> None:  # the old prompt had none; kept faithful
        assert rule_violations("monsoon kurta", it(1, "Cotton Kurta Shorts")) == []


class TestColourConstraint:
    def test_named_colour_is_enforced_upstream_not_here(self) -> None:
        # regression: substring matching flagged label-2 "Dark Blue" items for a "navy" query
        # (stage-2 q05); the catalogue's colour families live in intent_parser / retrieval
        navy = it(1, "Navy Shimmery Sheath Party Dress", "Dark Blue")
        assert rule_violations("navy outfit for reception party", navy) == []
        assert rule_violations("black kurta", it(2, "Kurta", "Red")) == []

    def test_style_words_and_unknown_colour_do_not_trigger(self) -> None:
        assert rule_violations("minimalist elegant kurta", it(1, "Kurta", "Red")) == []
        assert rule_violations("black kurta", it(1, "Kurta", None)) == []  # no colour data: kept

    def test_gray_grey_folding(self) -> None:
        assert named_colours("Gray suit") == {"grey"}
        assert rule_violations("gray suit", it(1, "Suit", "Grey")) == []


class TestApplyRulesAndDiversity:
    def test_violators_removed_from_the_whole_order_and_replaced_by_the_next(self) -> None:
        pool = [it(1, "Pyjama Set"), it(2, "Silk Dress", "Red"), it(3, "Satin Dress", "Black"),
                it(4, "Midi Dress", "Wine"), it(5, "Gown", "Green"), it(6, "Dress", "Navy"),
                it(7, "Dress", "Maroon")]
        shown, info = apply_rules("haldi outfit", pool, 5)
        assert ids(shown) == ["a2", "a3", "a4", "a5", "a6"] and info["n_removed"] == 1

    def test_fewer_than_top_k_when_few_comply(self) -> None:
        pool = [it(1, "Pyjama Set"), it(2, "Dress", "Red"), it(3, "Bikini")]
        shown, _ = apply_rules("wedding", pool, 5)
        assert ids(shown) == ["a2"]

    def test_never_empty_when_everything_violates(self) -> None:
        pool = [it(1, "Pyjama Set"), it(2, "Bikini")]
        shown, info = apply_rules("wedding", pool, 5)
        assert ids(shown) == ["a1", "a2"] and info["skipped"] is True

    def test_diversity_swap_replaces_the_last_pick_for_date_night(self) -> None:
        pool = [it(i, "Dress", "Black") for i in range(1, 7)] + [it(7, "Dress", "Red")]
        shown, info = apply_rules("date night look", pool, 5)
        assert ids(shown) == ["a1", "a2", "a3", "a4", "a7"] and info["diversity_swap"] is True

    def test_no_swap_when_already_diverse_or_not_date_night_or_colour_named(self) -> None:
        mono = [it(i, "Dress", "Black") for i in range(1, 7)] + [it(7, "Dress", "Red")]
        assert ids(apply_rules("casual dress", mono, 5)[0]) == ["a1", "a2", "a3", "a4", "a5"]
        assert ids(apply_rules("black dress date night", mono, 5)[0])[-1] == "a5"
        diverse = [it(1, "Dress", "Black"), it(2, "Dress", "Red")] + mono[2:]
        assert apply_rules("date night", diverse, 5)[1]["diversity_swap"] is False

    def test_swap_none_colour_safe(self) -> None:
        pool = [it(i, "Dress", None) for i in range(1, 7)]
        assert len(apply_rules("date night", pool, 5)[0]) == 5  # colour=None must not crash


class _Llm:
    def __init__(self, reply: Any) -> None:
        self.reply = reply

    def complete(self, messages: list[dict], **kw: Any) -> Any:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class TestRerankAppliesRules:
    POOL = [it(1, "Cotton Pyjama Set", "Red"), *[it(i, f"Dress {i}", "Black") for i in range(2, 9)]]

    def test_model_pick_of_a_violator_is_removed(self) -> None:
        from tests.test_reranker import _done

        out = rerank("haldi outfit", self.POOL, _Llm(_done("1,2,3,4,5,6")))
        assert "a1" not in ids(out) and len(out) == 5

    def test_retrieval_order_fallback_is_filtered_too(self) -> None:
        out = rerank("haldi outfit", self.POOL, _Llm(LLMCallError("fatal", 400)))
        assert "a1" not in ids(out)
