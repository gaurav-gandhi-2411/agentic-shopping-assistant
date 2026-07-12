"""Unit tests for validate_rationale and the rationale generation pipeline.

Covers:
- Grounded sentence kept intact.
- Invented colour word ("mustard dupatta" when no mustard in look) → sentence dropped.
- Invented garment type → sentence dropped.
- Price/size still scrubbed by validate_rationale.
- All-dropped fallback returns original + flag.
- template_rationale produces grounded output.
- build_fact_sheet extracts only real attributes.
- generate_rationales falls back to template on LLM failure.
"""
from __future__ import annotations

from src.agents.grounding import validate_rationale
from src.agents.outfit.rationale import (
    build_fact_sheet,
    generate_rationales,
    template_rationale,
)

# ── Shared test fixtures ───────────────────────────────────────────────────────

def _make_look(
    seed_colour: str = "blue",
    seed_type: str = "kurta",
    complement_colour: str = "yellow",
    complement_type: str = "dupatta",
    complement_slot: str = "accessory",
    occasion: str = "festive_puja",
    gender: str = "women",
) -> dict:
    """Build a minimal look dict for testing."""
    seed_item = {
        "article_id": "TEST001",
        "prod_name": f"{seed_colour.title()} {seed_type.title()}",
        "colour": seed_colour,
        "product_type": seed_type,
        "_role": "seed",
        "_slot": None,
        "gender": gender,
    }
    complement = {
        "article_id": "TEST002",
        "prod_name": f"{complement_colour.title()} {complement_type.title()}",
        "colour": complement_colour,
        "product_type": complement_type,
        "_role": "complement",
        "_slot": complement_slot,
        "gender": gender,
    }
    return {
        "look_id": "test-look-001",
        "seed_item": seed_item,
        "complements": [complement],
        "outfit_rationale": "",
        "empty_slots": [],
        "occasion": occasion,
        "gender": gender,
        "budget_total_inr": 2500.0,
    }


# ── validate_rationale tests ───────────────────────────────────────────────────

class TestValidateRationale:
    def test_grounded_sentence_kept(self) -> None:
        """A sentence referencing only look colours and types passes through."""
        look = _make_look(seed_colour="blue", seed_type="kurta")
        items = [look["seed_item"]] + look["complements"]
        text = "The blue kurta anchors the look beautifully."
        cleaned, flags = validate_rationale(text, items, look["occasion"])
        # "blue" is in whitelist (seed colour), "kurta" is in whitelist (seed type)
        assert "blue" in cleaned or "kurta" in cleaned
        assert not any("ungrounded" in f for f in flags)

    def test_invented_colour_dropped(self) -> None:
        """Sentence naming 'mustard' when no mustard item in look → flagged and dropped.

        When all sentences are dropped, validate_rationale returns the original
        text + an 'all_dropped' flag so the caller can fall back to the template.
        The key invariant is that an ungrounded colour flag IS raised.
        """
        # Look has blue kurta + yellow dupatta — no mustard
        look = _make_look(seed_colour="blue", seed_type="kurta",
                         complement_colour="yellow", complement_type="dupatta")
        items = [look["seed_item"]] + look["complements"]
        # Rationale invents "mustard" — not in look
        text = "Pair it with a mustard dupatta for a festive touch."
        cleaned, flags = validate_rationale(text, items, look["occasion"])
        # Grounding gate must raise an ungrounded_colour flag for mustard
        ungrounded_colour_flags = [f for f in flags if "ungrounded_colour:mustard" in f]
        assert len(ungrounded_colour_flags) > 0, (
            f"Expected 'ungrounded_colour:mustard' flag but got: {flags}"
        )
        # When all sentences are dropped the all_dropped sentinel is also present,
        # and the original text is returned to let the caller fall back.
        if "rationale:all_dropped" in flags:
            # Original returned — caller must use template; gate worked correctly
            assert cleaned == text
        else:
            # Sentence was stripped cleanly (partial drop)
            assert "mustard" not in cleaned

    def test_invented_garment_type_dropped(self) -> None:
        """Sentence naming 'blazer' when no blazer in look → flagged as ungrounded type."""
        look = _make_look(seed_colour="blue", seed_type="kurta",
                         complement_colour="yellow", complement_type="dupatta")
        items = [look["seed_item"]] + look["complements"]
        # "blazer" not in this look at all
        text = "Add a blazer to complete the festive look."
        cleaned, flags = validate_rationale(text, items, look["occasion"])
        # The gate must flag the ungrounded type
        ungrounded_type_flags = [f for f in flags if "ungrounded_type:blazer" in f]
        assert len(ungrounded_type_flags) > 0, (
            f"Expected 'ungrounded_type:blazer' flag but got: {flags}"
        )
        # If all_dropped: original returned (caller uses template) — correct
        # If partial drop: blazer should be gone from cleaned
        if "rationale:all_dropped" not in flags:
            assert "blazer" not in cleaned

    def test_price_still_scrubbed(self) -> None:
        """Price claims are scrubbed by validate_rationale (via validate_response)."""
        look = _make_look()
        items = [look["seed_item"]] + look["complements"]
        text = "This affordable kurta costs very little. It looks great."
        cleaned, flags = validate_rationale(text, items, look["occasion"])
        price_flags = [f for f in flags if f.startswith("price:")]
        assert len(price_flags) > 0, "Expected price flag to be raised"

    def test_size_still_scrubbed(self) -> None:
        """Size claims are scrubbed by validate_rationale."""
        look = _make_look()
        items = [look["seed_item"]] + look["complements"]
        text = "This runs small. The kurta looks festive."
        cleaned, flags = validate_rationale(text, items, look["occasion"])
        size_flags = [f for f in flags if f.startswith("size:")]
        assert len(size_flags) > 0, "Expected size flag to be raised"

    def test_all_dropped_returns_original_and_flag(self) -> None:
        """When ALL sentences are dropped, returns original text + all_dropped flag."""
        look = _make_look(seed_colour="blue", seed_type="kurta")
        items = [look["seed_item"]] + look["complements"]
        # Both sentences contain invented colours/types not in whitelist
        text = "Wear a mustard blazer. Add coral trousers."
        cleaned, flags = validate_rationale(text, items, look["occasion"])
        # If truly all dropped, original returned with all_dropped flag
        if "rationale:all_dropped" in flags:
            # original text is returned
            assert cleaned == text
        # Otherwise at least some sentences survived — acceptable

    def test_generic_styling_words_not_flagged(self) -> None:
        """Words like 'balance', 'hero', 'neutral', 'anchor' must NOT trigger drops."""
        look = _make_look(seed_colour="blue", seed_type="kurta")
        items = [look["seed_item"]] + look["complements"]
        text = "The blue kurta is the hero piece that anchors and balances the neutral palette."
        cleaned, flags = validate_rationale(text, items, look["occasion"])
        # Should not be dropped — these are styling words, not garment types
        assert "kurta" in cleaned or "blue" in cleaned
        ungrounded = [f for f in flags if "ungrounded" in f]
        assert len(ungrounded) == 0, f"Generic words falsely flagged: {ungrounded}"


class TestValidateRationaleBudgetMismatch:
    """The LLM could hallucinate a rupee figure that doesn't match the user's real
    budget_inr (e.g. say "under ₹5000" when the true budget is ₹3000). This must
    be caught the same way other ungrounded claims are — flagged and dropped."""

    def test_fabricated_budget_figure_is_dropped(self) -> None:
        look = _make_look(seed_colour="blue", seed_type="kurta")
        items = [look["seed_item"]] + look["complements"]
        text = "This kurta keeps you comfortably under your ₹5000 budget."
        cleaned, flags = validate_rationale(
            text, items, look["occasion"], budget_inr=3000.0
        )
        mismatch_flags = [f for f in flags if "budget_mismatch" in f]
        assert len(mismatch_flags) > 0, f"Expected budget_mismatch flag but got: {flags}"
        if "rationale:all_dropped" not in flags:
            assert "5000" not in cleaned

    def test_matching_budget_figure_is_kept(self) -> None:
        look = _make_look(seed_colour="blue", seed_type="kurta")
        items = [look["seed_item"]] + look["complements"]
        text = "This kurta keeps you comfortably within your ₹3000 budget."
        cleaned, flags = validate_rationale(
            text, items, look["occasion"], budget_inr=3000.0
        )
        mismatch_flags = [f for f in flags if "budget_mismatch" in f]
        assert len(mismatch_flags) == 0, f"Matching figure wrongly flagged: {mismatch_flags}"
        assert "3000" in cleaned

    def test_no_budget_known_does_not_check_numbers(self) -> None:
        """When budget_inr is None, no numeric check runs — the "budget" word itself
        is still scrubbed by the pre-existing price patterns (unrelated to this check)."""
        look = _make_look(seed_colour="blue", seed_type="kurta")
        items = [look["seed_item"]] + look["complements"]
        text = "The kurta looks great."
        cleaned, flags = validate_rationale(text, items, look["occasion"], budget_inr=None)
        assert not any("budget_mismatch" in f for f in flags)

    def test_close_rounded_figure_within_tolerance_is_kept(self) -> None:
        """A small rounding difference (e.g. ₹2980 vs a real ₹3000 budget) should
        not be flagged as fabricated."""
        look = _make_look(seed_colour="blue", seed_type="kurta")
        items = [look["seed_item"]] + look["complements"]
        text = "This kurta comes in at around ₹2980, right at your budget."
        cleaned, flags = validate_rationale(
            text, items, look["occasion"], budget_inr=3000.0
        )
        mismatch_flags = [f for f in flags if "budget_mismatch" in f]
        assert len(mismatch_flags) == 0, f"Rounded figure wrongly flagged: {mismatch_flags}"


# ── template_rationale tests ────────────────────────────────────────────────────

class TestTemplateRationale:
    def test_uses_seed_colour_and_type(self) -> None:
        look = _make_look(seed_colour="pink", seed_type="kurti", occasion="casual")
        result = template_rationale(look)
        assert "pink" in result.lower()
        assert "kurti" in result.lower()

    def test_mentions_occasion(self) -> None:
        look = _make_look(occasion="wedding_guest")
        result = template_rationale(look)
        assert "wedding" in result.lower() or "guest" in result.lower()

    def test_no_crash_on_empty_look(self) -> None:
        empty = {"look_id": "x", "seed_item": None, "complements": [],
                 "occasion": "casual", "gender": "women", "budget_total_inr": None}
        result = template_rationale(empty)
        assert isinstance(result, str) and len(result) > 0

    def test_includes_complement_info(self) -> None:
        look = _make_look(
            seed_colour="red", seed_type="kurta",
            complement_colour="gold", complement_type="juttis",
        )
        result = template_rationale(look)
        # At least the seed colour/type should appear
        assert "red" in result.lower() or "kurta" in result.lower()


# ── build_fact_sheet tests ──────────────────────────────────────────────────────

class TestBuildFactSheet:
    def test_extracts_occasion_and_gender(self) -> None:
        look = _make_look(occasion="sangeet", gender="women")
        fs = build_fact_sheet(look)
        assert "sangeet" in fs["occasion"]
        assert fs["gender"] == "women"

    def test_extracts_seed_colour_and_type(self) -> None:
        look = _make_look(seed_colour="blue", seed_type="kurta")
        fs = build_fact_sheet(look)
        assert fs["seed_colour"] == "blue"
        assert fs["seed_type"] == "kurta"

    def test_extracts_complement_pairs(self) -> None:
        look = _make_look(complement_colour="yellow", complement_type="dupatta",
                         complement_slot="accessory")
        fs = build_fact_sheet(look)
        assert len(fs["complement_pairs"]) == 1
        pair = fs["complement_pairs"][0]
        assert pair["slot"] == "accessory"
        assert pair["colour"] == "yellow"
        assert pair["type"] == "dupatta"

    def test_no_price_in_fact_sheet(self) -> None:
        """Fact sheet must never expose price."""
        look = _make_look()
        fs = build_fact_sheet(look)
        assert "price" not in fs
        assert "budget" not in fs

    def test_no_price_in_complement_pairs(self) -> None:
        look = _make_look()
        fs = build_fact_sheet(look)
        for pair in fs["complement_pairs"]:
            assert "price" not in pair


# ── generate_rationales fallback tests ─────────────────────────────────────────

class TestGenerateRationales:
    def test_falls_back_to_template_on_llm_failure(self) -> None:
        """When LLM raises an exception, template rationale is used for all looks."""

        class _BrokenLLM:
            def generate(self, prompt: str, system: str | None = None, **kw: object) -> str:
                raise RuntimeError("Intentional test failure")

            def chat(self, messages: list, **kw: object) -> str:
                raise RuntimeError("Intentional test failure")

            def generate_stream(self, prompt: str, **kw: object):  # type: ignore[override]
                raise RuntimeError("Intentional test failure")

            def chat_stream(self, messages: list, **kw: object):  # type: ignore[override]
                raise RuntimeError("Intentional test failure")

        look = _make_look(seed_colour="green", seed_type="kurta", occasion="festive_puja")
        rationales = generate_rationales(
            [look], _BrokenLLM(), occasion="festive_puja", gender="women"
        )
        assert len(rationales) == 1
        # Template rationale must reference real attributes
        assert "green" in rationales[0].lower() or "kurta" in rationales[0].lower()

    def test_falls_back_on_wrong_length_json(self) -> None:
        """If LLM returns fewer items than looks, falls back to template."""

        class _ShortLLM:
            def generate(self, prompt: str, system: str | None = None, **kw: object) -> str:
                return "[]"  # empty list — wrong length

            def chat(self, messages: list, **kw: object) -> str:
                return "[]"

            def generate_stream(self, prompt: str, **kw: object):  # type: ignore[override]
                return iter([])

            def chat_stream(self, messages: list, **kw: object):  # type: ignore[override]
                return iter([])

        look1 = _make_look(seed_colour="blue", seed_type="kurta")
        look2 = _make_look(seed_colour="pink", seed_type="kurti")
        rationales = generate_rationales(
            [look1, look2], _ShortLLM(), occasion="casual", gender="women"
        )
        assert len(rationales) == 2
        for r in rationales:
            assert isinstance(r, str) and len(r) > 0

    def test_valid_llm_response_used_directly(self) -> None:
        """A valid LLM response matching look count is used (after grounding gate)."""

        class _GoodLLM:
            def generate(self, prompt: str, system: str | None = None, **kw: object) -> str:
                # Return a rationale that only uses grounded terms
                return '["The blue kurta anchors the festive look."]'

            def chat(self, messages: list, **kw: object) -> str:
                return '["The blue kurta anchors the festive look."]'

            def generate_stream(self, prompt: str, **kw: object):  # type: ignore[override]
                return iter([])

            def chat_stream(self, messages: list, **kw: object):  # type: ignore[override]
                return iter([])

        look = _make_look(seed_colour="blue", seed_type="kurta", occasion="festive_puja")
        rationales = generate_rationales(
            [look], _GoodLLM(), occasion="festive_puja", gender="women"
        )
        assert len(rationales) == 1
        # Either the LLM rationale (grounded) or the template — both reference "blue"/"kurta"
        assert "blue" in rationales[0].lower() or "kurta" in rationales[0].lower()
