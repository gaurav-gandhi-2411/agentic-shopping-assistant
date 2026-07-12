"""Regression tests for the 2026-07-10 live-sweep Batch 1 trust-killer fixes.

Each test encodes a defect proven on the live URL (reports/ui_defect_sweep_20260710.md):
  (#1, the OOC word-boundary fix, is covered separately in test_ooc_detection.py)
  #2  men's boards shipped bottomless with a false "no men's bottoms" claim
  #6  stylist notes narrated raw category strings / wrong colours / broken grammar
"""
from __future__ import annotations

import pandas as pd

from src.agents.outfit import composer
from src.agents.outfit.composer import _suppression_reason, compose_outfit
from src.agents.outfit.rationale import (
    _display_colour,
    _display_noun,
    build_fact_sheet,
    template_rationale,
)


class TestStylistNoteGrounding:
    def test_display_noun_prefers_name_over_raw_category(self) -> None:
        assert _display_noun(
            "Blue Blazers, Waistcoats and Suits", "Self Design Men Waistcoat"
        ) == "waistcoat"

    def test_display_noun_clean_product_type_passthrough(self) -> None:
        assert _display_noun("trousers", "Some Unrecognisable Name 42") == "trousers"

    def test_display_colour_prefers_name_colour_over_colour_group(self) -> None:
        # Live defect: a Wine kurta narrated as "purple" (its colour-group).
        assert _display_colour("Purple", "Inddus Women Wine Self Design Anarkali") == "wine"

    def test_template_never_emits_raw_category_string(self) -> None:
        look = {
            "seed_item": {
                "prod_name": "Mods Western Star Self Design Sherwani",
                "product_type": "Kurtas, Ethnic Sets and Bottoms",
                "colour": "Black",
            },
            "complements": [
                {
                    "_slot": "outerwear",
                    "prod_name": "Self Design Men Waistcoat",
                    "product_type": "Blue Blazers, Waistcoats and Suits",
                    "colour": "Red",
                },
            ],
            "occasion": "reception",
        }
        note = template_rationale(look)
        assert "Blazers, Waistcoats and Suits" not in note
        assert "Ethnic Sets and Bottoms" not in note
        assert "sherwani" in note
        assert "waistcoat" in note

    def test_template_plural_verb_agreement(self) -> None:
        look = {
            "seed_item": {"prod_name": "Navy Kurta", "product_type": "kurta", "colour": "Blue"},
            "complements": [
                {
                    "_slot": "bottom",
                    "prod_name": "Black Tailored Trousers",
                    "product_type": "trousers",
                    "colour": "Black",
                },
            ],
            "occasion": "reception",
        }
        note = template_rationale(look)
        assert "trousers keep the focus" in note
        assert "trousers keeps" not in note

    def test_fact_sheet_uses_grounded_display_attributes(self) -> None:
        look = {
            "seed_item": {
                "prod_name": "Inddus Women Wine Self Design Anarkali Kurta",
                "product_type": "kurta",
                "colour": "Purple",
            },
            "complements": [],
            "occasion": "sangeet",
            "gender": "women",
        }
        sheet = build_fact_sheet(look)
        assert sheet["seed_colour"] == "wine"


class TestMensBottomSlot:
    def test_suppression_reason_never_claims_absolute_inventory_absence(self) -> None:
        reason = _suppression_reason("bottom", "men")
        assert "that match this look" in reason

    def test_bottom_slot_retries_with_western_formal_query_for_men(self, monkeypatch) -> None:
        """The ethnic bottom query failing must trigger ONE western-trouser retry."""
        calls: list[str] = []

        def fake_find(query: str, slot_name: str, **kwargs) -> dict | None:
            calls.append(f"{slot_name}:{query}")
            if slot_name == "bottom" and "tailored trousers" in query:
                return {
                    "article_id": "TRS1",
                    "prod_name": "Black Tailored Trousers",
                    "product_type": "trousers",
                    "colour": "Black",
                    "price_inr": 1500.0,
                    "store": "flipkart",
                }
            return None

        monkeypatch.setattr(composer, "_find_best_candidate", fake_find)
        catalogue_df = pd.DataFrame(
            [
                {
                    "article_id": "SHERWANI1",
                    "prod_name": "Black Self Design Sherwani",
                    "product_type_name": "sherwani",
                    "colour_group_name": "Black",
                    "gender": "men",
                    "price_inr": 5200.0,
                    "image_url": "http://example.com/x.jpg",
                    "store": "flipkart",
                    "detail_desc": "",
                }
            ]
        )
        look = compose_outfit(
            catalogue_df,
            retriever=None,  # fake_find never touches it
            seed_article_id="SHERWANI1",
            occasion_slug="reception",
            gender="men",
        )
        bottoms = [c for c in look["complements"] if c["_slot"] == "bottom"]
        assert len(bottoms) == 1, f"bottom slot not filled; calls={calls}"
        assert bottoms[0]["article_id"] == "TRS1"
        assert not any(s["slot"] == "bottom" for s in look["suppressed_slots"])
