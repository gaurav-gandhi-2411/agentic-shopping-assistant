"""Schema validation for eval/fixtures/model_eval_queries.yaml.

Pure-Python structural checks only — this file does NOT execute the agent,
retriever, or intent parser against the fixture queries (that is
scripts/eval_model.py's job, built concurrently against this exact schema).
It guarantees the fixture file itself is well-formed so the harness can load
it without per-entry defensive code:

  - the file parses as YAML and has the expected top-level shape
  - every entry has a unique, pattern-conformant id ("<category>_<3-digit>")
  - every entry's category is one of the six allowed values
  - each category meets its minimum query-count floor
  - expected_intent only ever uses real IntentV1 field names (introspected
    from the dataclass itself, not hand-copied — so this test can't drift
    from the real schema)
  - gates.checks entries are drawn from this fixture set's own defined
    vocabulary (see the YAML file's header comment)
  - gates.occasion_slug (when present) resolves to a real Occasion
  - expected_intent.body_type / gates.body_type (when present) are in the
    body-type registry's base-shape vocabulary
  - relevance/gates block-omission rules are internally consistent per
    category (e.g. body_type bare statements never carry relevance/gates)
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.agents.intent_parser import IntentV1
from src.agents.outfit.body_type import BASE_SHAPE_SLUGS, MODIFIER_SLUGS
from src.agents.outfit.occasions import OCCASIONS, get_occasion

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "model_eval_queries.yaml"

ALLOWED_CATEGORIES = {
    "search",
    "occasion",
    "couple",
    "body_type",
    "refinement",
    "adversarial",
}

# Minimum per-category counts requested for this fixture set. Some categories
# may exceed the floor (e.g. adversarial ships one extra dedicated regression
# probe) — counts are asserted as ">=", never "==".
CATEGORY_MINIMUMS = {
    "search": 60,
    "occasion": 60,
    "couple": 30,
    "body_type": 30,
    "refinement": 40,
    "adversarial": 30,
}

ID_PATTERN = re.compile(r"^(search|occasion|couple|body_type|refinement|adversarial)_\d{3}$")

# This fixture set's gates.checks vocabulary — exactly the 4 checks the task
# brief specified, confirmed implemented in scripts/eval_model.py's
# evaluate_gate_checks (built concurrently against this exact schema). The
# adversarial section covers two additional bug classes (kids-item leak,
# cross-gender ethnic composition) that have no dedicated check in the
# harness today; those entries deliberately reuse gender_pure/
# suppression_honest as the closest implemented proxies rather than
# reference a check name the harness would silently no-op (see the YAML
# header comment for the full explanation).
ALLOWED_CHECKS = {
    "gender_pure",
    "budget_respected",
    "no_novelty",
    "suppression_honest",
}

EXPECTED_INTENT_FIELDS = {f.name for f in dataclasses.fields(IntentV1)}

VALID_GENDERS = {"women", "men", "unisex"}
VALID_BODY_TYPES = set(BASE_SHAPE_SLUGS)
VALID_BODY_MODIFIERS = set(MODIFIER_SLUGS)


@pytest.fixture(scope="module")
def fixture_doc() -> dict[str, Any]:
    assert FIXTURE_PATH.exists(), f"fixture file not found: {FIXTURE_PATH}"
    with FIXTURE_PATH.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict), "top-level YAML document must be a mapping"
    return doc


@pytest.fixture(scope="module")
def queries(fixture_doc: dict[str, Any]) -> list[dict[str, Any]]:
    qs = fixture_doc.get("queries")
    assert isinstance(qs, list) and qs, "'queries' must be a non-empty list"
    return qs


class TestTopLevelShape:
    def test_version_present(self, fixture_doc: dict[str, Any]) -> None:
        assert fixture_doc.get("version") == 1

    def test_queries_is_list_of_mappings(self, queries: list[dict[str, Any]]) -> None:
        assert all(isinstance(q, dict) for q in queries)


class TestIds:
    def test_all_ids_unique(self, queries: list[dict[str, Any]]) -> None:
        ids = [q["id"] for q in queries]
        dupes = {i for i in ids if ids.count(i) > 1}
        assert not dupes, f"duplicate ids: {dupes}"

    def test_all_ids_pattern_conformant(self, queries: list[dict[str, Any]]) -> None:
        offenders = [q["id"] for q in queries if not ID_PATTERN.match(q["id"])]
        assert not offenders, f"ids not matching <category>_<3-digit>: {offenders}"

    def test_id_prefix_matches_category(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries if not q["id"].startswith(f"{q['category']}_")
        ]
        assert not offenders, f"id prefix != category: {offenders}"


class TestCategories:
    def test_every_category_allowed(self, queries: list[dict[str, Any]]) -> None:
        offenders = {q["category"] for q in queries} - ALLOWED_CATEGORIES
        assert not offenders, f"unexpected categories: {offenders}"

    @pytest.mark.parametrize("category,minimum", sorted(CATEGORY_MINIMUMS.items()))
    def test_category_meets_minimum_count(
        self, queries: list[dict[str, Any]], category: str, minimum: int
    ) -> None:
        n = sum(1 for q in queries if q["category"] == category)
        assert n >= minimum, f"{category}: expected >= {minimum}, got {n}"

    def test_total_equals_sum_of_categories(self, queries: list[dict[str, Any]]) -> None:
        assert len(queries) == sum(
            1 for q in queries if q["category"] in ALLOWED_CATEGORIES
        )


class TestTurns:
    def test_every_entry_has_nonempty_turns(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries
            if not isinstance(q.get("turns"), list) or not q["turns"]
        ]
        assert not offenders, f"missing/empty turns: {offenders}"

    def test_every_turn_is_nonempty_string(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries
            if any(not isinstance(t, str) or not t.strip() for t in q["turns"])
        ]
        assert not offenders, f"non-string/empty turn text: {offenders}"

    def test_refinement_category_is_multi_turn(self, queries: list[dict[str, Any]]) -> None:
        """refinement entries score the END-of-conversation state, so each
        must carry at least 2 turns (see YAML header comment)."""
        offenders = [
            q["id"] for q in queries
            if q["category"] == "refinement" and len(q["turns"]) < 2
        ]
        assert not offenders, f"refinement entries with < 2 turns: {offenders}"


class TestExpectedIntentSchema:
    def test_every_entry_has_expected_intent(self, queries: list[dict[str, Any]]) -> None:
        offenders = [q["id"] for q in queries if "expected_intent" not in q]
        assert not offenders, f"missing expected_intent: {offenders}"

    def test_expected_intent_is_mapping(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries if not isinstance(q["expected_intent"], dict)
        ]
        assert not offenders, f"expected_intent not a mapping: {offenders}"

    def test_expected_intent_fields_are_real_intentv1_fields(
        self, queries: list[dict[str, Any]]
    ) -> None:
        """Every key used in any expected_intent block must be a genuine
        IntentV1 dataclass field — introspected live so this test can never
        silently drift from the real schema (see src/agents/intent_parser.py)."""
        offenders: dict[str, set[str]] = {}
        for q in queries:
            bad = set(q["expected_intent"].keys()) - EXPECTED_INTENT_FIELDS
            if bad:
                offenders[q["id"]] = bad
        assert not offenders, f"expected_intent has non-IntentV1 fields: {offenders}"

    def test_is_product_query_is_bool_when_present(
        self, queries: list[dict[str, Any]]
    ) -> None:
        offenders = [
            q["id"] for q in queries
            if "is_product_query" in q["expected_intent"]
            and not isinstance(q["expected_intent"]["is_product_query"], bool)
        ]
        assert not offenders, f"is_product_query not a bool: {offenders}"

    def test_gender_values_valid_when_present(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries
            if q["expected_intent"].get("gender") not in ({None} | VALID_GENDERS)
        ]
        assert not offenders, f"invalid expected_intent.gender: {offenders}"

    def test_body_type_values_valid_when_present(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries
            if q["expected_intent"].get("body_type") not in ({None} | VALID_BODY_TYPES)
        ]
        assert not offenders, f"invalid expected_intent.body_type: {offenders}"

    def test_body_modifiers_are_valid_list(self, queries: list[dict[str, Any]]) -> None:
        offenders = []
        for q in queries:
            mods = q["expected_intent"].get("body_modifiers", [])
            if mods is None:
                continue
            if not isinstance(mods, list) or any(m not in VALID_BODY_MODIFIERS for m in mods):
                offenders.append(q["id"])
        assert not offenders, f"invalid expected_intent.body_modifiers: {offenders}"

    def test_occasion_resolves_via_get_occasion_when_present(
        self, queries: list[dict[str, Any]]
    ) -> None:
        offenders = []
        for q in queries:
            slug = q["expected_intent"].get("occasion")
            if slug is None:
                continue
            if slug not in OCCASIONS:
                offenders.append((q["id"], slug))
                continue
            # get_occasion must never silently fall back to "casual" for a
            # slug that is supposed to be one of the real 12.
            assert get_occasion(slug).slug == slug
        assert not offenders, f"expected_intent.occasion not a real slug: {offenders}"

    def test_budget_is_int_or_none(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries
            if not isinstance(q["expected_intent"].get("budget_max_inr"), (int, type(None)))
        ]
        assert not offenders, f"budget_max_inr not int/None: {offenders}"


class TestBodyTypeCategoryRules:
    def _bare_statement_ids(self, queries: list[dict[str, Any]]) -> list[str]:
        return [
            q["id"] for q in queries
            if q["category"] == "body_type" and q["expected_intent"].get("is_product_query") is False
        ]

    def test_bare_statements_have_no_relevance_block(
        self, queries: list[dict[str, Any]]
    ) -> None:
        by_id = {q["id"]: q for q in queries}
        offenders = [i for i in self._bare_statement_ids(queries) if "relevance" in by_id[i]]
        assert not offenders, f"bare body_type statements must omit relevance: {offenders}"

    def test_bare_statements_have_no_gates_block(self, queries: list[dict[str, Any]]) -> None:
        by_id = {q["id"]: q for q in queries}
        offenders = [i for i in self._bare_statement_ids(queries) if "gates" in by_id[i]]
        assert not offenders, f"bare body_type statements must omit gates: {offenders}"

    def test_at_least_one_bare_statement_per_base_shape(
        self, queries: list[dict[str, Any]]
    ) -> None:
        bare = [
            q for q in queries
            if q["category"] == "body_type" and q["expected_intent"].get("is_product_query") is False
        ]
        shapes_covered = {q["expected_intent"].get("body_type") for q in bare} - {None}
        missing = VALID_BODY_TYPES - shapes_covered
        assert not missing, f"no bare statement fixture for shapes: {missing}"


class TestGatesSchema:
    def _entries_with_gates(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [q for q in queries if "gates" in q]

    def test_gates_is_mapping(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_gates(queries)
            if not isinstance(q["gates"], dict)
        ]
        assert not offenders, f"gates not a mapping: {offenders}"

    def test_all_occasion_category_entries_have_gates(
        self, queries: list[dict[str, Any]]
    ) -> None:
        offenders = [
            q["id"] for q in queries if q["category"] == "occasion" and "gates" not in q
        ]
        assert not offenders, f"occasion entries missing gates: {offenders}"

    def test_all_occasion_category_entries_compose_true(
        self, queries: list[dict[str, Any]]
    ) -> None:
        offenders = [
            q["id"] for q in queries
            if q["category"] == "occasion" and q["gates"].get("compose") is not True
        ]
        assert not offenders, f"occasion entries must set gates.compose=true: {offenders}"

    def test_all_couple_category_entries_have_gates_couple_true(
        self, queries: list[dict[str, Any]]
    ) -> None:
        offenders = [
            q["id"] for q in queries
            if q["category"] == "couple"
            and (q.get("gates") is None or q["gates"].get("couple") is not True)
        ]
        assert not offenders, f"couple entries must set gates.couple=true: {offenders}"

    def test_search_category_never_has_gates(self, queries: list[dict[str, Any]]) -> None:
        offenders = [q["id"] for q in queries if q["category"] == "search" and "gates" in q]
        assert not offenders, f"search entries must omit gates entirely: {offenders}"

    def test_checks_are_in_allowed_vocabulary(self, queries: list[dict[str, Any]]) -> None:
        offenders: dict[str, set[str]] = {}
        for q in self._entries_with_gates(queries):
            checks = q["gates"].get("checks", [])
            bad = set(checks) - ALLOWED_CHECKS
            if bad:
                offenders[q["id"]] = bad
        assert not offenders, f"gates.checks outside allowed vocabulary: {offenders}"

    def test_occasion_slug_resolves_when_present(self, queries: list[dict[str, Any]]) -> None:
        offenders = []
        for q in self._entries_with_gates(queries):
            slug = q["gates"].get("occasion_slug")
            if slug is None:
                continue
            if slug not in OCCASIONS:
                offenders.append((q["id"], slug))
        assert not offenders, f"gates.occasion_slug not a real slug: {offenders}"

    def test_gates_gender_valid_when_present(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_gates(queries)
            if q["gates"].get("gender") not in ({None} | VALID_GENDERS)
        ]
        assert not offenders, f"invalid gates.gender: {offenders}"

    def test_gates_body_type_valid_when_present(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_gates(queries)
            if q["gates"].get("body_type") not in ({None} | VALID_BODY_TYPES)
        ]
        assert not offenders, f"invalid gates.body_type: {offenders}"

    def test_gates_budget_is_int_or_none(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_gates(queries)
            if not isinstance(q["gates"].get("budget_inr"), (int, type(None)))
        ]
        assert not offenders, f"gates.budget_inr not int/None: {offenders}"

    def test_gates_couple_is_bool(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_gates(queries)
            if not isinstance(q["gates"].get("couple"), bool)
        ]
        assert not offenders, f"gates.couple not a bool: {offenders}"

    def test_gates_compose_is_bool(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_gates(queries)
            if not isinstance(q["gates"].get("compose"), bool)
        ]
        assert not offenders, f"gates.compose not a bool: {offenders}"

    def test_compose_true_never_has_null_gender(self, queries: list[dict[str, Any]]) -> None:
        """compose_outfit's gender and compose_couple_look's partner_gender
        are both required (non-optional, non-nullable) parameters — a
        gates.compose=true entry with gender=null would error inside the
        composer rather than exercise a real gate check."""
        offenders = [
            q["id"] for q in self._entries_with_gates(queries)
            if q["gates"].get("compose") is True and q["gates"].get("gender") is None
        ]
        assert not offenders, f"compose=true with null gates.gender: {offenders}"

    def test_couple_category_always_composes(self, queries: list[dict[str, Any]]) -> None:
        """scripts/eval_model.py's gates stage always calls
        compose_couple_look for category="couple" rows (never
        compose_partner_look, which needs a session anchor this eval harness
        does not simulate) — so every couple entry must set compose=true or
        it is silently skipped by the harness entirely."""
        offenders = [
            q["id"] for q in queries
            if q["category"] == "couple" and q["gates"].get("compose") is not True
        ]
        assert not offenders, f"couple entries must set gates.compose=true: {offenders}"


class TestRelevanceSchema:
    def _entries_with_relevance(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [q for q in queries if "relevance" in q]

    def test_relevance_has_must(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_relevance(queries)
            if not isinstance(q["relevance"].get("must"), dict) or not q["relevance"]["must"]
        ]
        assert not offenders, f"relevance.must missing/empty: {offenders}"

    def test_must_values_are_lowercase_string_lists(
        self, queries: list[dict[str, Any]]
    ) -> None:
        offenders = []
        for q in self._entries_with_relevance(queries):
            must = q["relevance"]["must"]
            for _key, values in must.items():
                if not isinstance(values, list) or not values:
                    offenders.append(q["id"])
                    continue
                for v in values:
                    if not isinstance(v, str):
                        offenders.append(q["id"])
                        continue
                    if "product_type_contains" in must and v != v.lower():
                        offenders.append(q["id"])
        assert not offenders, f"relevance.must malformed values: {sorted(set(offenders))}"

    def test_graded_implies_must_present(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in self._entries_with_relevance(queries)
            if "graded" in q["relevance"] and not q["relevance"].get("must")
        ]
        assert not offenders, f"graded present without must: {offenders}"

    def test_search_and_is_product_query_false_never_has_relevance(
        self, queries: list[dict[str, Any]]
    ) -> None:
        """A query the parser itself resolves as non-product (is_product_query
        False) has nothing meaningful to score at retrieval — relevance must
        be omitted."""
        offenders = [
            q["id"] for q in queries
            if q["expected_intent"].get("is_product_query") is False and "relevance" in q
        ]
        assert not offenders, f"is_product_query=False entries must omit relevance: {offenders}"


class TestDataCeilingTags:
    def test_data_ceiling_tags_is_list_when_present(
        self, queries: list[dict[str, Any]]
    ) -> None:
        offenders = [
            q["id"] for q in queries
            if "data_ceiling_tags" in q and not isinstance(q["data_ceiling_tags"], list)
        ]
        assert not offenders, f"data_ceiling_tags not a list: {offenders}"

    def test_data_ceiling_tag_values_are_strings(self, queries: list[dict[str, Any]]) -> None:
        offenders = [
            q["id"] for q in queries
            if isinstance(q.get("data_ceiling_tags"), list)
            and any(not isinstance(t, str) for t in q["data_ceiling_tags"])
        ]
        assert not offenders, f"non-string data_ceiling_tags entries: {offenders}"


class TestSourceCitationHeader:
    def test_header_comment_present(self) -> None:
        """The YAML file must open with a '#'-comment header documenting
        sources and the gates.checks/data_ceiling_tags vocabularies (STYLE
        requirement) — a lightweight raw-text check since comments are
        stripped by yaml.safe_load."""
        text = FIXTURE_PATH.read_text(encoding="utf-8")
        first_non_blank = next(line for line in text.splitlines() if line.strip())
        assert first_non_blank.startswith("#"), "fixture file must open with a comment header"
        assert "gates.checks" in text
        assert "data_ceiling_tags" in text
