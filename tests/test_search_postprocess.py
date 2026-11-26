"""Plain-search post-processing: de-dup key, per-store cap with backfill, gender interleave,
refine visibility, budget guard, and the no-op regression (a clean list is returned unchanged).

Hermetic: plain dicts, no retriever, no network.
"""
from __future__ import annotations

from src.agents import search_postprocess as pp


def _it(
    aid: str,
    store: str = "s1",
    title: str | None = None,
    price: float | None = 1000.0,
    gender: str = "women",
    colour: str = "Red",
    image: str | None = None,
) -> dict:
    t = title if title is not None else f"Item {aid}"
    return {
        "article_id": aid, "store": store, "prod_name": t, "display_name": t,
        "price_inr": price, "gender": gender, "colour": colour, "image_url": image,
    }


def _ids(items: list[dict]) -> list[str]:
    return [i["article_id"] for i in items]


# --- de-duplication key -----------------------------------------------------------------


class TestSameProduct:
    def test_same_store_same_title_is_duplicate(self) -> None:
        a = _it("1", "kisah", "Yellow Printed Kurta Set", 2999)
        b = _it("2", "kisah", "yellow  printed kurta set!", 3499)  # case/punct/space only
        assert pp.same_product(a, b)

    def test_celebrity_prefixed_copy_with_same_price_is_duplicate(self) -> None:
        a = _it("1", "kalpraag", "Chaitanya Haldi Kurta Set", 16818)
        b = _it("2", "kalpraag", "Abhiijeet Saawant in Chaitanya Haldi Kurta Set", 16818)
        assert pp.same_product(a, b)

    def test_containment_needs_equal_price(self) -> None:
        a = _it("1", "kalpraag", "Chaitanya Haldi Kurta Set", 16818)
        b = _it("2", "kalpraag", "Abhiijeet Saawant in Chaitanya Haldi Kurta Set", 17000)
        assert not pp.same_product(a, b)

    def test_short_titles_never_match_by_containment(self) -> None:
        a = _it("1", "s1", "Blue Shirt", 500)
        b = _it("2", "s1", "Blue Shirt Slim", 500)
        assert not pp.same_product(a, b)

    def test_different_store_is_not_duplicate(self) -> None:
        a = _it("1", "kisah", "Yellow Printed Kurta Set", 2999)
        b = _it("2", "libas", "Yellow Printed Kurta Set", 2999)
        assert not pp.same_product(a, b)

    def test_same_image_same_store_is_duplicate(self) -> None:
        a = _it("1", "s1", "A", 100, image="http://x/1.jpg")
        b = _it("2", "s1", "B", 900, image="http://x/1.jpg")
        assert pp.same_product(a, b)

    def test_two_distinct_designs_same_price_are_kept(self) -> None:
        a = _it("1", "kisah", "Yellow Geometric Printed Kurta Set", 3299)
        b = _it("2", "kisah", "Yellow Printed Front Open Kurta Set", 3299)
        assert not pp.same_product(a, b)


class TestDedupSelection:
    def test_duplicate_dropped_and_backfilled_to_top_k(self) -> None:
        ranked = [
            _it("1", "a", "Chaitanya Haldi Kurta Set", 16818),
            _it("2", "b", "Plain One"),
            _it("3", "a", "Abhiijeet Saawant in Chaitanya Haldi Kurta Set", 16818),
            _it("4", "c", "Plain Two"),
            _it("5", "d", "Plain Three"),
            _it("6", "e", "Plain Four"),
        ]
        out, rep = pp.select_results(ranked, 5)
        assert _ids(out) == ["1", "2", "4", "5", "6"]
        assert rep["removed"] == {"duplicate": 1}
        assert rep["short_reason"] is None

    def test_legacy_name_colour_key_still_dedups_across_stores(self) -> None:
        ranked = [_it("1", "a", "Gyda blouse", colour="Red"), _it("2", "b", "Gyda!", colour="Red"),
                  _it("3", "c", "Other")]
        out, _ = pp.select_results(ranked, 5)
        assert _ids(out) == ["1", "3"]


# --- per-store cap ----------------------------------------------------------------------


class TestStoreCap:
    def test_cap_constant_is_three(self) -> None:
        assert pp.MAX_PER_STORE == 3

    def test_at_most_three_per_store_with_backfill(self) -> None:
        ranked = [_it(str(i), "libas", f"Saree {i}") for i in range(1, 7)] + [
            _it("7", "x", "Other seven"), _it("8", "y", "Other eight"), _it("9", "z", "Other nine")]
        out, rep = pp.select_results(ranked, 5)
        assert _ids(out) == ["1", "2", "3", "7", "8"]
        assert rep["removed"] == {"store_cap": 3}

    def test_cap_off_is_dedupe_only(self) -> None:
        ranked = [_it(str(i), "libas", f"Saree {i}") for i in range(1, 7)]
        out, rep = pp.select_results(ranked, 5, max_per_store=99)
        assert _ids(out) == ["1", "2", "3", "4", "5"]
        assert rep["removed"] == {}

    def test_reranker_order_kept_otherwise(self) -> None:
        ranked = [_it("1", "a"), _it("2", "b"), _it("3", "a"), _it("4", "a"), _it("5", "a"),
                  _it("6", "b"), _it("7", "c")]
        out, _ = pp.select_results(ranked, 5)
        assert _ids(out) == ["1", "2", "3", "4", "6"]

    def test_fewer_than_top_k_when_pool_is_narrow_with_logged_reason(self) -> None:
        ranked = [_it(str(i), "jadeblue" if i < 5 else "mls", f"Sherwani {i}") for i in range(1, 9)]
        out, rep = pp.select_results(ranked, 7)
        assert len(out) == 6  # 3 + 3, never relaxed silently
        assert rep["n_shown"] == 6
        assert rep["short_reason"] and "store_cap=2" in rep["short_reason"]


# --- budget guard -----------------------------------------------------------------------


class TestBudget:
    def test_selection_never_shows_over_budget_and_backfills(self) -> None:
        ranked = [_it("1", "a", price=900), _it("2", "b", price=1600), _it("3", "c", price=1500),
                  _it("4", "d", price=None), _it("5", "e", price=100), _it("6", "f", price=50),
                  _it("7", "g", price=700)]
        out, rep = pp.select_results(ranked, 5, price_max=1500)
        assert _ids(out) == ["1", "3", "5", "6", "7"]
        assert rep["removed"] == {"over_budget": 1, "no_price": 1}

    def test_no_cap_means_no_price_filtering(self) -> None:
        ranked = [_it("1", price=None), _it("2", "b", price=10**6)]
        out, _ = pp.select_results(ranked, 5, price_max=None)
        assert _ids(out) == ["1", "2"]

    def test_final_guard_removes_over_budget_and_unpriced(self) -> None:
        items = [_it("1", price=1500), _it("2", "b", price=1500.5), _it("3", "c", price=None)]
        assert _ids(pp.enforce_budget(items, 1500)) == ["1"]
        assert _ids(pp.enforce_budget(items, None)) == ["1", "2", "3"]

    def test_budget_cap_prefers_filters_then_parsed_intent(self) -> None:
        assert pp.budget_cap({"price_max": 1500}, 900) == 1500.0
        assert pp.budget_cap({}, 900) == 900.0
        assert pp.budget_cap({}, None) is None
        assert pp.budget_cap({"price_max": 0}, None) is None


# --- gender interleave ------------------------------------------------------------------


class TestInterleave:
    def _pool(self) -> list[dict]:
        return [_it("m1", "a", gender="men"), _it("m2", "b", gender="men"), _it("m3", "c", gender="men"),
                _it("m4", "d", gender="men"), _it("w1", "e", gender="women"),
                _it("w2", "f", gender="women"), _it("w3", "g", gender="women")]

    def test_alternates_starting_with_best_ranked_gender_keeping_each_order(self) -> None:
        out, rep = pp.select_results(self._pool(), 5, interleave_genders=True)
        assert _ids(out) == ["m1", "w1", "m2", "w2", "m3"]
        assert rep["interleaved"] is True

    def test_women_first_when_best_item_is_a_woman(self) -> None:
        pool = list(reversed(self._pool()))
        out, _ = pp.select_results(pool, 5, interleave_genders=True)
        assert [i["gender"] for i in out] == ["women", "men", "women", "men", "women"]

    def test_honours_store_cap_while_interleaving(self) -> None:
        pool = [_it("m1", "a", gender="men"), _it("w1", "a", gender="women"),
                _it("m2", "a", gender="men"), _it("w2", "b", gender="women"),
                _it("m3", "c", gender="men"), _it("w3", "d", gender="women")]
        out, _ = pp.select_results(pool, 5, interleave_genders=True, max_per_store=2)
        assert max(sum(1 for i in out if i["store"] == s) for s in "abcd") <= 2
        assert _ids(out) == ["m1", "w1", "m3", "w2", "w3"]

    def test_one_gender_only_is_unchanged_order(self) -> None:
        pool = [_it(str(i), f"s{i}", gender="men") for i in range(1, 8)]
        out, rep = pp.select_results(pool, 5, interleave_genders=True)
        assert _ids(out) == ["1", "2", "3", "4", "5"]
        assert rep["interleaved"] is False

    def test_exhausted_gender_falls_back_to_the_other(self) -> None:
        pool = [_it("m1", "a", gender="men"), _it("w1", "b", gender="women"),
                _it("m2", "c", gender="men"), _it("m3", "d", gender="men"), _it("m4", "e", gender="men")]
        out, _ = pp.select_results(pool, 5, interleave_genders=True)
        assert _ids(out) == ["m1", "w1", "m2", "m3", "m4"]

    def test_unisex_keeps_its_rank_and_takes_no_turn(self) -> None:
        pool = [_it("m1", "a", gender="men"), _it("u1", "b", gender="unknown"),
                _it("m2", "c", gender="men"), _it("w1", "d", gender="women"),
                _it("w2", "e", gender="women")]
        out, _ = pp.select_results(pool, 5, interleave_genders=True)
        assert _ids(out) == ["m1", "u1", "w1", "m2", "w2"]

    def test_mix_window_ignores_a_gender_that_only_appears_behind_the_window(self) -> None:
        pool = [_it(f"m{i}", f"s{i}", gender="men") for i in range(1, 7)] + [
            _it("w1", "sw", gender="women")]
        out, rep = pp.select_results(pool, 5, interleave_genders=True, mix_window=6)
        assert _ids(out) == ["m1", "m2", "m3", "m4", "m5"]
        assert rep["interleaved"] is False

    def test_mix_window_mixes_inside_and_fills_from_behind(self) -> None:
        pool = [_it("m1", "a", gender="men"), _it("w1", "b", gender="women"),
                _it("m2", "c", gender="men"), _it("m3", "d", gender="men"),
                _it("w2", "e", gender="women"), _it("w3", "f", gender="women")]
        out, rep = pp.select_results(pool, 5, interleave_genders=True, mix_window=3)
        # window = m1, w1, m2 -> alternate m1, w1, m2; then plain ranked order from behind it
        assert _ids(out) == ["m1", "w1", "m2", "m3", "w2"]
        assert rep["interleaved"] is True

    def test_interleave_off_keeps_rank_order(self) -> None:
        out, _ = pp.select_results(self._pool(), 5, interleave_genders=False)
        assert _ids(out) == ["m1", "m2", "m3", "m4", "w1"]


# --- refine control visibility ----------------------------------------------------------


class TestRefineVisibility:
    both = [_it("1", gender="women"), _it("2", "b", gender="men")]

    def test_shown_when_no_gender_and_both_present(self) -> None:
        assert pp.gender_refine_options({}, self.both)

    def test_hidden_when_query_fixed_gender_via_gender_key(self) -> None:
        assert not pp.gender_refine_options({"gender": "men"}, self.both)

    def test_hidden_when_query_fixed_gender_via_index_group(self) -> None:
        assert not pp.gender_refine_options({"index_group_name": "ladieswear"}, self.both)
        assert not pp.gender_refine_options({"index_group_name": "menswear"}, self.both)

    def test_hidden_when_candidates_are_one_gender(self) -> None:  # e.g. a saree query
        assert not pp.gender_refine_options({}, [_it("1"), _it("2", "b")])
        assert not pp.gender_refine_options({}, [_it("1", gender="men"), _it("2", "b", gender="unknown")])

    def test_unisex_filters_do_not_count_as_a_fixed_gender(self) -> None:
        assert pp.gender_refine_options({"index_group_name": "divided"}, self.both)

    def test_refine_queries_restate_the_query_with_a_gender(self) -> None:
        out = pp.refine_queries("Haldi kurta set ")
        assert out == [{"gender": "women", "query": "Haldi kurta set for women"},
                       {"gender": "men", "query": "Haldi kurta set for men"}]

    def test_refine_queries_parse_back_to_the_stated_gender_and_budget(self) -> None:
        from src.agents.intent_parser import parse_intent

        for g in ("women", "men"):
            q = next(o["query"] for o in pp.refine_queries("Office wear under ₹1,500") if o["gender"] == g)
            intent = parse_intent(q)
            assert intent.gender == g
            assert intent.budget_max_inr == 1500


class TestRefineRerun:
    def test_rerun_of_previous_query_is_detected(self) -> None:
        msgs = [{"role": "user", "content": "Haldi kurta set"}, {"role": "assistant", "content": "x"},
                {"role": "user", "content": "Haldi kurta set for women"}]
        assert pp.is_gender_refine_rerun("Haldi kurta set for women", msgs)

    def test_ordinary_follow_up_is_not(self) -> None:
        msgs = [{"role": "user", "content": "red dress"}, {"role": "assistant", "content": "x"},
                {"role": "user", "content": "kurta for women"}]
        assert not pp.is_gender_refine_rerun("kurta for women", msgs)

    def test_first_turn_is_not(self) -> None:
        assert not pp.is_gender_refine_rerun("kurta for men", [{"role": "user", "content": "kurta for men"}])


# --- no-op regression -------------------------------------------------------------------


class TestNoOp:
    def test_clean_list_is_returned_unchanged(self) -> None:
        ranked = [_it(str(i), f"s{i}", f"Distinct title number {i}", price=100.0 * i) for i in range(1, 9)]
        out, rep = pp.select_results(ranked, 5, price_max=5000)
        assert out == ranked[:5]
        assert rep["removed"] == {} and rep["short_reason"] is None and not rep["interleaved"]

    def test_short_clean_list_is_returned_as_is(self) -> None:
        ranked = [_it("1", "a"), _it("2", "b")]
        out, rep = pp.select_results(ranked, 5)
        assert out == ranked
        assert rep["short_reason"] == "fewer candidates than top_k"

    def test_gendered_query_does_not_interleave(self) -> None:
        ranked = [_it(str(i), f"s{i}", gender="men") for i in range(1, 4)] + [_it("9", "z", gender="women")]
        out, rep = pp.select_results(ranked, 5, interleave_genders=False)
        assert out == ranked and not rep["interleaved"]
