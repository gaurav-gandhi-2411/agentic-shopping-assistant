"""Structural checks on src/catalogue/hygiene_lexicon.py (the rules are tested in
test_catalogue_hygiene.py)."""

from __future__ import annotations

import re

from src.catalogue.hygiene_lexicon import GENERIC_TYPES, LEXICON

_CATEGORIES = {"garment", "accessory", "jewellery", "footwear", "non_fashion"}


def test_every_row_compiles_and_uses_a_known_category() -> None:
    for gtype, category, pattern in LEXICON:
        assert category in _CATEGORIES, (gtype, category)
        re.compile(rf"\b(?:{pattern})\b")


def test_group_names_are_unique_and_each_maps_to_one_category() -> None:
    names = [g for g, _, _ in LEXICON]
    assert len(names) == len(set(names))


def test_generic_types_are_normalised_form() -> None:
    # classify_type lower-cases and collapses non-alphanumerics before the membership test.
    for t in GENERIC_TYPES:
        assert t == re.sub(r"[^a-z0-9]+", " ", t).strip()
