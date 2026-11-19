"""Pins the hand audit of src/catalogue/hygiene.py (evals/fixtures/catalogue_hygiene_audit.json).

200 rows, stratified by store, labelled by reading product_type + title. The rules must keep
reproducing those labels; the known misses (stores whose titles are bare model names under a
generic product type) must stay "unknown" (fail open), never a wrong class."""

from __future__ import annotations

import json
from pathlib import Path

from src.catalogue.hygiene import derive_category

_FIXTURE = Path(__file__).resolve().parent.parent / "evals" / "fixtures" / "catalogue_hygiene_audit.json"
_ROWS = json.loads(_FIXTURE.read_text(encoding="utf-8"))["rows"]


def test_audit_fixture_shape() -> None:
    assert len(_ROWS) == 200
    assert {r["truth"] for r in _ROWS} <= {"garment", "accessory", "jewellery", "footwear", "non_fashion"}


def test_rules_reproduce_the_audit_labels() -> None:
    got = [derive_category(r["product_type_name"], r["prod_name"]) for r in _ROWS]
    wrong = [(r, g) for r, g in zip(_ROWS, got, strict=True) if g.category != r["truth"]]
    assert len(_ROWS) - len(wrong) >= 196  # measured 196/200 on the day of the audit
    assert all(g.category == "unknown" for _, g in wrong), "a miss must fail open, not mislabel"
    assert not [g for g in got if g.quarantine_reason], "no audited row is a genuine contradiction"
