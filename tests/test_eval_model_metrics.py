"""
Unit tests for scripts/eval_model.py's pure metric functions, schema loader, and
data-ceiling PASS-conversion logic.

No network/LLM/index dependency — every test here exercises pure functions that
operate on hand-built dicts/lists, plus one small pandas.DataFrame built inline
for catalogue_universe_ids (pandas is a project dependency already, not a heavy
retrieval/LLM component).

Groups:
  1. item_matches_must / item_matches_graded / item_gain
  2. precision_at_k
  3. dcg_at_k / ndcg_at_k (includes the hand-computed NDCG case)
  4. recall_at_k / catalogue_universe_ids
  5. Gate checks: gender_pure / budget_respected / no_novelty
  6. evaluate_suppression_honest + the data-ceiling PASS-conversion logic
  7. Judge JSON defensive parsing
  8. stratified_sample determinism
  9. Fixture schema loader (inline 6-query sample fixture written to tmp_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.eval_model import (
    catalogue_universe_ids,
    check_budget_respected,
    check_gender_pure,
    check_no_novelty,
    dcg_at_k,
    evaluate_gate_checks,
    evaluate_suppression_honest,
    item_gain,
    item_matches_graded,
    item_matches_must,
    load_fixture_queries,
    ndcg_at_k,
    parse_judge_json,
    precision_at_k,
    recall_at_k,
    stratified_sample,
)

# ---------------------------------------------------------------------------
# Group 1: item_matches_must / item_matches_graded / item_gain
# ---------------------------------------------------------------------------


class TestItemMatchers:
    def test_must_matches_on_product_type_substring(self) -> None:
        item = {"product_type": "dress", "prod_name": "Black Party Dress", "gender": "women"}
        must = {"product_type_contains": ["dress"]}
        assert item_matches_must(item, must) is True

    def test_must_matches_on_prod_name_substring_when_product_type_differs(self) -> None:
        # product_type may be under-classified ("top") but prod_name still says "dress".
        item = {"product_type": "top", "prod_name": "Shirt Dress", "gender": "women"}
        must = {"product_type_contains": ["dress"]}
        assert item_matches_must(item, must) is True

    def test_must_fails_when_no_term_matches(self) -> None:
        item = {"product_type": "trousers", "prod_name": "Blue Chinos", "gender": "women"}
        must = {"product_type_contains": ["dress"]}
        assert item_matches_must(item, must) is False

    def test_must_gender_in_filters(self) -> None:
        item = {"product_type": "dress", "prod_name": "Black Dress", "gender": "men"}
        must = {"product_type_contains": ["dress"], "gender_in": ["women"]}
        assert item_matches_must(item, must) is False

    def test_must_gender_in_case_insensitive(self) -> None:
        item = {"product_type": "dress", "prod_name": "Black Dress", "gender": "Women"}
        must = {"product_type_contains": ["dress"], "gender_in": ["WOMEN"]}
        assert item_matches_must(item, must) is True

    def test_empty_must_is_vacuously_satisfied(self) -> None:
        item = {"product_type": "anything", "prod_name": "x", "gender": "unknown"}
        assert item_matches_must(item, {}) is True

    def test_graded_colour_in_matches(self) -> None:
        item = {"colour": "Black"}
        assert item_matches_graded(item, {"colour_in": ["black", "navy"]}) is True

    def test_graded_colour_in_no_match(self) -> None:
        item = {"colour": "Red"}
        assert item_matches_graded(item, {"colour_in": ["black", "navy"]}) is False

    def test_graded_empty_never_matches(self) -> None:
        item = {"colour": "Black"}
        assert item_matches_graded(item, {}) is False

    def test_item_gain_zero_when_must_fails(self) -> None:
        item = {"product_type": "trousers", "prod_name": "x", "gender": "women", "colour": "Black"}
        relevance = {
            "must": {"product_type_contains": ["dress"]},
            "graded": {"colour_in": ["black"]},
        }
        assert item_gain(item, relevance) == 0

    def test_item_gain_one_when_must_only(self) -> None:
        item = {"product_type": "dress", "prod_name": "x", "gender": "women", "colour": "Red"}
        relevance = {
            "must": {"product_type_contains": ["dress"]},
            "graded": {"colour_in": ["black"]},
        }
        assert item_gain(item, relevance) == 1

    def test_item_gain_two_when_must_and_graded(self) -> None:
        item = {"product_type": "dress", "prod_name": "x", "gender": "women", "colour": "Black"}
        relevance = {
            "must": {"product_type_contains": ["dress"]},
            "graded": {"colour_in": ["black"]},
        }
        assert item_gain(item, relevance) == 2


# ---------------------------------------------------------------------------
# Group 2: precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def _item(self, product_type: str) -> dict[str, Any]:
        return {"product_type": product_type, "prod_name": "", "gender": "women"}

    def test_precision_at_k_basic(self) -> None:
        items = [self._item("dress"), self._item("dress"), self._item("trousers")]
        relevance = {"must": {"product_type_contains": ["dress"]}}
        assert precision_at_k(items, relevance, 3) == pytest.approx(2 / 3)

    def test_precision_at_k_uses_available_window_not_k(self) -> None:
        # k=5 but only 2 items retrieved -> denominator is 2, not 5.
        items = [self._item("dress"), self._item("trousers")]
        relevance = {"must": {"product_type_contains": ["dress"]}}
        assert precision_at_k(items, relevance, 5) == pytest.approx(0.5)

    def test_precision_at_k_empty_items_is_zero(self) -> None:
        relevance = {"must": {"product_type_contains": ["dress"]}}
        assert precision_at_k([], relevance, 5) == 0.0

    def test_precision_at_k_perfect(self) -> None:
        items = [self._item("dress")] * 5
        relevance = {"must": {"product_type_contains": ["dress"]}}
        assert precision_at_k(items, relevance, 5) == 1.0


# ---------------------------------------------------------------------------
# Group 3: dcg_at_k / ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_dcg_hand_computed(self) -> None:
        # DCG([2,0,1], k=3) = (2^2-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4)
        #                   =        3/1      +        0/1.585  +        1/2
        #                   = 3 + 0 + 0.5 = 3.5
        assert dcg_at_k([2, 0, 1], 3) == pytest.approx(3.5)

    def test_ndcg_hand_computed(self) -> None:
        # Hand-computed (see also ndcg_at_k's docstring):
        #   gains = [2, 0, 1], k=3
        #   actualDCG = 3.5 (see test_dcg_hand_computed)
        #   ideal sorted gains = [2, 1, 0]
        #   idealDCG = 3/1 + 1/log2(3) + 0/log2(4) = 3 + 0.63093 + 0 = 3.63093
        #   NDCG@3 = 3.5 / 3.63093 ≈ 0.96394
        items = [
            {"product_type": "dress", "prod_name": "", "gender": "women", "colour": "Black"},
            {"product_type": "trousers", "prod_name": "", "gender": "women", "colour": "Black"},
            {"product_type": "dress", "prod_name": "", "gender": "women", "colour": "Red"},
        ]
        relevance = {
            "must": {"product_type_contains": ["dress"]},
            "graded": {"colour_in": ["black"]},
        }
        # gains for items above: dress+black=2, trousers=0 (fails must), dress+red=1
        assert ndcg_at_k(items, relevance, 3) == pytest.approx(0.9639404333, abs=1e-6)

    def test_ndcg_perfect_ordering_is_one(self) -> None:
        items = [
            {"product_type": "dress", "prod_name": "", "gender": "women", "colour": "Black"},
            {"product_type": "dress", "prod_name": "", "gender": "women", "colour": "Black"},
        ]
        relevance = {
            "must": {"product_type_contains": ["dress"]},
            "graded": {"colour_in": ["black"]},
        }
        assert ndcg_at_k(items, relevance, 2) == pytest.approx(1.0)

    def test_ndcg_all_zero_gain_is_zero(self) -> None:
        items = [{"product_type": "trousers", "prod_name": "", "gender": "women", "colour": "x"}]
        relevance = {"must": {"product_type_contains": ["dress"]}}
        assert ndcg_at_k(items, relevance, 1) == 0.0


# ---------------------------------------------------------------------------
# Group 4: recall_at_k / catalogue_universe_ids
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_at_k_basic(self) -> None:
        universe = {"a1", "a2", "a3", "a4"}
        retrieved = ["a1", "x", "a2", "y", "z"]
        assert recall_at_k(retrieved, universe, 50) == pytest.approx(2 / 4)

    def test_recall_at_k_respects_window(self) -> None:
        universe = {"a1", "a2"}
        retrieved = ["x", "y", "a1", "a2"]  # a1/a2 only appear after position k=2
        assert recall_at_k(retrieved, universe, 2) == 0.0

    def test_recall_at_k_empty_universe_is_zero(self) -> None:
        assert recall_at_k(["a1"], set(), 50) == 0.0

    def test_catalogue_universe_ids_vectorized_mask(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "article_id": ["a1", "a2", "a3", "a4"],
                "product_type_name": ["dress", "trousers", "dress", "footwear"],
                "prod_name": ["Black Dress", "Blue Chinos", "Shirt Dress", "Sneakers"],
                "gender": ["women", "women", "men", "women"],
            }
        ).set_index("article_id")
        must = {"product_type_contains": ["dress"], "gender_in": ["women"]}
        assert catalogue_universe_ids(df, must) == {"a1"}

    def test_catalogue_universe_ids_no_gender_column_is_empty_for_gender_scoped_must(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "article_id": ["a1"],
                "product_type_name": ["dress"],
                "prod_name": ["Black Dress"],
            }
        ).set_index("article_id")
        must = {"product_type_contains": ["dress"], "gender_in": ["women"]}
        assert catalogue_universe_ids(df, must) == set()


# ---------------------------------------------------------------------------
# Group 5: Gate checks — gender_pure / budget_respected / no_novelty
# ---------------------------------------------------------------------------


def _look(
    gender: str = "women",
    budget_total_inr: float | None = None,
    item_genders: list[str] | None = None,
    novelty_name: str | None = None,
) -> dict[str, Any]:
    item_genders = item_genders if item_genders is not None else [gender, gender]
    complements = [{"prod_name": f"Item {i}", "gender": g} for i, g in enumerate(item_genders)]
    if novelty_name:
        complements.append({"prod_name": novelty_name, "gender": gender})
    return {
        "seed_item": {"prod_name": "Seed", "gender": gender},
        "complements": complements,
        "gender": gender,
        "budget_total_inr": budget_total_inr,
        "empty_slots": [],
        "suppressed_slots": [],
    }


class TestGateChecks:
    def test_gender_pure_true_when_all_match(self) -> None:
        look = _look(gender="women")
        assert check_gender_pure(look, "women", couple=False) is True

    def test_gender_pure_false_on_mixed_gender_item(self) -> None:
        look = _look(gender="women", item_genders=["women", "men"])
        assert check_gender_pure(look, "women", couple=False) is False

    def test_gender_pure_couple_checks_each_board_independently(self) -> None:
        primary = _look(gender="men")
        partner = _look(gender="women")
        assert check_gender_pure((primary, partner), None, couple=True) is True

    def test_gender_pure_couple_false_when_one_board_mixed(self) -> None:
        primary = _look(gender="men")
        partner = _look(gender="women", item_genders=["women", "men"])
        assert check_gender_pure((primary, partner), None, couple=True) is False

    def test_budget_respected_none_budget_is_noop(self) -> None:
        look = _look(budget_total_inr=999999.0)
        assert check_budget_respected(look, None, couple=False) is True

    def test_budget_respected_within_cap(self) -> None:
        look = _look(budget_total_inr=5000.0)
        assert check_budget_respected(look, 8000.0, couple=False) is True

    def test_budget_respected_over_cap_fails(self) -> None:
        look = _look(budget_total_inr=9000.0)
        assert check_budget_respected(look, 8000.0, couple=False) is False

    def test_budget_respected_couple_is_per_person_not_split(self) -> None:
        # Each board independently <= budget_inr -- neither is half of it.
        primary = _look(budget_total_inr=7000.0)
        partner = _look(budget_total_inr=7500.0)
        assert check_budget_respected((primary, partner), 8000.0, couple=True) is True

    def test_budget_respected_couple_fails_if_either_board_over(self) -> None:
        primary = _look(budget_total_inr=7000.0)
        partner = _look(budget_total_inr=8500.0)
        assert check_budget_respected((primary, partner), 8000.0, couple=True) is False

    def test_no_novelty_true_when_clean(self) -> None:
        look = _look()
        assert check_no_novelty(look, couple=False) is True

    def test_no_novelty_false_on_costume_item(self) -> None:
        look = _look(novelty_name="Luxury Piano Shape Statement Handbag")
        assert check_no_novelty(look, couple=False) is False

    def test_no_novelty_case_insensitive(self) -> None:
        look = _look(novelty_name="QUIRKY Guitar Print Tee")
        assert check_no_novelty(look, couple=False) is False


# ---------------------------------------------------------------------------
# Group 6: evaluate_suppression_honest + data-ceiling PASS-conversion logic
# ---------------------------------------------------------------------------


class TestSuppressionAndDataCeiling:
    def test_suppression_honest_vacuously_true_with_no_empty_slots(self) -> None:
        assert evaluate_suppression_honest([], []) is True

    def test_suppression_honest_true_when_reason_recorded(self) -> None:
        suppressed = [{"slot": "footwear", "reason": "No women's footwear in our partner stores"}]
        assert evaluate_suppression_honest(suppressed, ["footwear"]) is True

    def test_suppression_dishonest_when_no_reason_recorded(self) -> None:
        # Slot is empty but suppressed_slots never explains why -- a real code defect.
        assert evaluate_suppression_honest([], ["footwear"]) is False

    def test_suppression_dishonest_when_reason_is_blank(self) -> None:
        suppressed = [{"slot": "footwear", "reason": ""}]
        assert evaluate_suppression_honest(suppressed, ["footwear"]) is False

    def test_evaluate_gate_checks_data_ceiling_pass_conversion(self) -> None:
        """An honestly-suppressed slot on a data_ceiling_tags-tagged query is a
        thin-inventory outcome, not a model/code defect: suppression_honest is
        still True, and the result is separately flagged data_ceiling=True so
        report-builders never conflate the two."""
        look = _look(gender="women")
        look["empty_slots"] = ["footwear"]
        look["suppressed_slots"] = [
            {"slot": "footwear", "reason": "No women's footwear in our partner stores yet"}
        ]
        result = evaluate_gate_checks(
            look,
            checks_wanted=["gender_pure", "suppression_honest"],
            expected_gender="women",
            budget_inr=None,
            data_ceiling_tags=["womens_footwear"],
            couple=False,
        )
        assert result["checks"]["suppression_honest"] is True
        assert result["data_ceiling"] is True

    def test_evaluate_gate_checks_no_data_ceiling_tag_is_model_code_row(self) -> None:
        """Same honest suppression, but with NO data_ceiling_tags -- reported as
        an ordinary MODEL/CODE row (data_ceiling stays False); still PASSES on
        suppression_honest since the code behaved honestly."""
        look = _look(gender="women")
        look["empty_slots"] = ["footwear"]
        look["suppressed_slots"] = [
            {"slot": "footwear", "reason": "No women's footwear in our partner stores yet"}
        ]
        result = evaluate_gate_checks(
            look,
            checks_wanted=["suppression_honest"],
            expected_gender="women",
            budget_inr=None,
            data_ceiling_tags=[],
            couple=False,
        )
        assert result["checks"]["suppression_honest"] is True
        assert result["data_ceiling"] is False

    def test_evaluate_gate_checks_dishonest_suppression_never_converted_by_tag(self) -> None:
        """A dishonest suppression (no reason) stays a genuine failure even when
        data_ceiling_tags is set -- the tag only excuses THIN INVENTORY, never a
        code defect that failed to explain itself."""
        look = _look(gender="women")
        look["empty_slots"] = ["footwear"]
        look["suppressed_slots"] = []  # no reason recorded -- code bug
        result = evaluate_gate_checks(
            look,
            checks_wanted=["suppression_honest"],
            expected_gender="women",
            budget_inr=None,
            data_ceiling_tags=["womens_footwear"],
            couple=False,
        )
        assert result["checks"]["suppression_honest"] is False
        # data_ceiling is still True (a slot WAS empty on a tagged category) but the
        # per-check pass/fail is not silently flipped -- callers see both facts.
        assert result["data_ceiling"] is True

    def test_evaluate_gate_checks_only_requested_checks_are_scored(self) -> None:
        look = _look(gender="women")
        result = evaluate_gate_checks(
            look,
            checks_wanted=["gender_pure"],
            expected_gender="women",
            budget_inr=None,
            data_ceiling_tags=[],
            couple=False,
        )
        assert set(result["checks"].keys()) == {"gender_pure"}


# ---------------------------------------------------------------------------
# Group 7: Judge JSON defensive parsing
# ---------------------------------------------------------------------------


class TestParseJudgeJson:
    def test_parses_clean_json(self) -> None:
        raw = '{"occasion_score": 4, "coherence_score": 5, "reason": "cohesive look"}'
        parsed = parse_judge_json(raw)
        assert parsed == {"occasion_score": 4, "coherence_score": 5, "reason": "cohesive look"}

    def test_parses_json_wrapped_in_prose_and_fences(self) -> None:
        raw = (
            "Sure, here is my rating:\n```json\n"
            '{"occasion_score": 3, "coherence_score": 2, "reason": "ok"}\n```'
        )
        parsed = parse_judge_json(raw)
        assert parsed is not None
        assert parsed["occasion_score"] == 3
        assert parsed["coherence_score"] == 2

    def test_returns_none_on_empty_string(self) -> None:
        assert parse_judge_json("") is None
        assert parse_judge_json(None) is None

    def test_returns_none_on_malformed_json(self) -> None:
        assert parse_judge_json("{occasion_score: 4, not valid json}") is None

    def test_returns_none_when_score_out_of_range(self) -> None:
        raw = '{"occasion_score": 7, "coherence_score": 3, "reason": "x"}'
        assert parse_judge_json(raw) is None

    def test_returns_none_when_score_not_an_int(self) -> None:
        raw = '{"occasion_score": "high", "coherence_score": 3, "reason": "x"}'
        assert parse_judge_json(raw) is None

    def test_returns_none_when_score_is_bool(self) -> None:
        # bool is an int subclass in Python -- must be explicitly rejected.
        raw = '{"occasion_score": true, "coherence_score": 3, "reason": "x"}'
        assert parse_judge_json(raw) is None

    def test_returns_none_missing_required_field(self) -> None:
        raw = '{"occasion_score": 4, "reason": "x"}'
        assert parse_judge_json(raw) is None


# ---------------------------------------------------------------------------
# Group 8: stratified_sample determinism
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def _items(self) -> list[dict[str, Any]]:
        return [{"id": f"{cat}_{i}", "cat": cat} for cat in ("a", "b", "c") for i in range(4)]

    def test_returns_all_when_n_exceeds_length(self) -> None:
        items = self._items()
        result = stratified_sample(items, 100, seed=42, key_fn=lambda x: x["cat"])
        assert len(result) == len(items)

    def test_deterministic_across_calls(self) -> None:
        items = self._items()
        r1 = stratified_sample(items, 6, seed=42, key_fn=lambda x: x["cat"])
        r2 = stratified_sample(items, 6, seed=42, key_fn=lambda x: x["cat"])
        assert [i["id"] for i in r1] == [i["id"] for i in r2]

    def test_spreads_across_all_groups_when_possible(self) -> None:
        items = self._items()
        result = stratified_sample(items, 3, seed=42, key_fn=lambda x: x["cat"])
        cats = {i["cat"] for i in result}
        assert cats == {"a", "b", "c"}  # one from each group, not 3 from the same one

    def test_returns_zero_for_n_le_zero(self) -> None:
        items = self._items()
        assert stratified_sample(items, 0, seed=42, key_fn=lambda x: x["cat"]) == []

    def test_different_seeds_can_produce_different_samples(self) -> None:
        items = self._items()
        r1 = stratified_sample(items, 3, seed=42, key_fn=lambda x: x["cat"])
        r2 = stratified_sample(items, 3, seed=1, key_fn=lambda x: x["cat"])
        # Not asserting inequality (could coincidentally match) -- just that both
        # are valid, fully-determined-by-seed samples of the right size.
        assert len(r1) == 3
        assert len(r2) == 3


# ---------------------------------------------------------------------------
# Group 9: Fixture schema loader — inline 6-query sample (tmp_path)
# ---------------------------------------------------------------------------

_SAMPLE_FIXTURE_YAML = """
version: 1
queries:
  - id: search_001
    category: search
    turns: ["show me a black dress"]
    expected_intent:
      garment_type: dress
      gender: null
      colour: Black
      occasion: null
      budget_max_inr: null
      body_type: null
      is_product_query: true
    relevance:
      must:
        product_type_contains: ["dress"]
        gender_in: ["women"]
      graded:
        colour_in: ["black"]

  - id: occasion_001
    category: occasion
    turns: ["style me for a haldi function, budget 8000"]
    expected_intent:
      garment_type: null
      gender: women
      colour: null
      occasion: haldi
      budget_max_inr: 8000
      body_type: null
      is_product_query: true
    gates:
      compose: true
      occasion_slug: haldi
      gender: women
      budget_inr: 8000
      checks: ["gender_pure", "budget_respected", "no_novelty", "suppression_honest"]
    data_ceiling_tags: []

  - id: couple_001
    category: couple
    turns: ["style us as a couple for a sangeet, I'm the husband"]
    expected_intent:
      garment_type: null
      gender: men
      colour: null
      occasion: sangeet
      budget_max_inr: null
      body_type: null
      is_product_query: true
    gates:
      compose: true
      occasion_slug: sangeet
      gender: men
      couple: true
      checks: ["gender_pure", "no_novelty", "suppression_honest"]
    data_ceiling_tags: ["mens_occasionwear"]

  - id: body_type_001
    category: body_type
    turns: ["I'm pear-shaped, style me for the office"]
    expected_intent:
      garment_type: null
      gender: women
      colour: null
      occasion: office
      budget_max_inr: null
      body_type: pear
      is_product_query: true
    gates:
      compose: true
      occasion_slug: office
      gender: women
      body_type: pear
      checks: ["gender_pure", "suppression_honest"]

  - id: refinement_001
    category: refinement
    turns: ["show me trousers", "something cheaper"]
    expected_intent:
      garment_type: trousers
      gender: null
      colour: null
      occasion: null
      budget_max_inr: null
      body_type: null
      is_product_query: true
    relevance:
      must:
        product_type_contains: ["trousers"]

  - id: adversarial_001
    category: adversarial
    turns: ["asdkjaslkdj qqqqq"]
    expected_intent:
      garment_type: null
      gender: null
      colour: null
      occasion: null
      budget_max_inr: null
      body_type: null
      is_product_query: false
"""


class TestFixtureSchemaLoader:
    def test_loads_all_queries(self, tmp_path: Path) -> None:
        fixture_path = tmp_path / "model_eval_queries.yaml"
        fixture_path.write_text(_SAMPLE_FIXTURE_YAML, encoding="utf-8")
        queries = load_fixture_queries(fixture_path)
        assert len(queries) == 6
        assert {q["id"] for q in queries} == {
            "search_001", "occasion_001", "couple_001",
            "body_type_001", "refinement_001", "adversarial_001",
        }

    def test_categories_cover_expected_set(self, tmp_path: Path) -> None:
        fixture_path = tmp_path / "model_eval_queries.yaml"
        fixture_path.write_text(_SAMPLE_FIXTURE_YAML, encoding="utf-8")
        queries = load_fixture_queries(fixture_path)
        cats = {q["category"] for q in queries}
        assert cats == {"search", "occasion", "couple", "body_type", "refinement", "adversarial"}

    def test_last_turn_is_the_scored_turn_for_refinement(self, tmp_path: Path) -> None:
        fixture_path = tmp_path / "model_eval_queries.yaml"
        fixture_path.write_text(_SAMPLE_FIXTURE_YAML, encoding="utf-8")
        queries = load_fixture_queries(fixture_path)
        refinement = next(q for q in queries if q["id"] == "refinement_001")
        assert refinement["turns"][-1] == "something cheaper"
        assert len(refinement["turns"]) == 2

    def test_gates_block_parses_for_couple_query(self, tmp_path: Path) -> None:
        fixture_path = tmp_path / "model_eval_queries.yaml"
        fixture_path.write_text(_SAMPLE_FIXTURE_YAML, encoding="utf-8")
        queries = load_fixture_queries(fixture_path)
        couple = next(q for q in queries if q["id"] == "couple_001")
        assert couple["gates"]["couple"] is True
        assert couple["data_ceiling_tags"] == ["mens_occasionwear"]

    def test_missing_queries_key_raises(self, tmp_path: Path) -> None:
        fixture_path = tmp_path / "bad.yaml"
        fixture_path.write_text("version: 1\n", encoding="utf-8")
        with pytest.raises(KeyError):
            load_fixture_queries(fixture_path)

    def test_raw_yaml_still_parses_with_pyyaml(self) -> None:
        # Sanity check the inline fixture text itself is well-formed YAML,
        # independent of load_fixture_queries.
        data = yaml.safe_load(_SAMPLE_FIXTURE_YAML)
        assert data["version"] == 1
        assert len(data["queries"]) == 6
