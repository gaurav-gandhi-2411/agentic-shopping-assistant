#!/usr/bin/env python
"""Class-wide probe: does the anchor-vs-complement gate asymmetry (found for
"office" in eval/gate_leak_audit.py, 4/293 real composed looks) exist for
OTHER EITHER-lean occasions that also carry a register gate in
is_coherent_candidate (gym=gate 5, party_evening=gate 6), or is it genuinely
office-only?

Structural read first (src/agents/outfit/composer.py::_anchor_matches_occasion,
src/agents/outfit/occasions.py): _anchor_matches_occasion only ever rejects an
anchor when occ.ethnic_lean is ETHNIC_ONLY or ETHNIC_HEAVY. office/gym/
party_evening are ALL ethnic_lean=EITHER -- so _anchor_matches_occasion is a
structural no-op for exactly these 3 occasions, which are ALSO the only 3
occasions with a register gate in is_coherent_candidate that isn't already
implied by ethnic_lean (gates 4/5/6). This predicts the SAME leak shape for
gym and party_evening, not just office -- this script tests that prediction
directly by calling the real compose_outfit() repeatedly for each occasion,
varying gender/budget/body_type to get pool diversity, and re-checking every
seed_item against is_coherent_candidate.

Usage:
    python -m eval.anchor_complement_gate_class_probe
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from src.agents.outfit.coherence import is_coherent_candidate  # noqa: E402
from src.agents.outfit.composer import compose_outfit  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"

_BASE_SHAPE_SLUGS = ("pear", "apple", "hourglass", "rectangle", "inverted_triangle", "lean_build")
_MODIFIER_SLUGS = ("petite", "tall", "plus_size", "short_build")

_PROBE_MATRIX: list[tuple[str, str, int | None, str | None]] = []
for _occ in ("gym", "party_evening", "office"):  # office = live control, known to leak in the fixture
    for _gender in ("women", "men"):
        _PROBE_MATRIX.append((_occ, _gender, None, None))
        for _shape in _BASE_SHAPE_SLUGS:
            _PROBE_MATRIX.append((_occ, _gender, None, _shape))
        for _mod in _MODIFIER_SLUGS:
            _PROBE_MATRIX.append((_occ, _gender, None, f"__mod__{_mod}"))


def main() -> None:
    config = load_config()
    catalogue_df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)

    print(f"{'occasion':<16} {'gender':<6} {'shape/mod':<14}  seed_item  ->  gate result")
    leaks = []
    seen_anchors: set[tuple[str, str, str]] = set()
    for occasion_slug, gender, budget_inr, shape_field in _PROBE_MATRIX:
        if shape_field is not None and shape_field.startswith("__mod__"):
            body_type, body_modifiers = None, [shape_field.removeprefix("__mod__")]
        else:
            body_type, body_modifiers = shape_field, []
        label = shape_field or "-"
        try:
            board = compose_outfit(
                catalogue_df, retriever, occasion_slug=occasion_slug, gender=gender,
                budget_inr=budget_inr, body_type=body_type, body_modifiers=body_modifiers,
            )
        except Exception as e:  # noqa: BLE001
            print(f"{occasion_slug:<16} {gender:<6} {label:<14}  COMPOSE ERROR: {e}")
            continue

        seed = board.get("seed_item")
        if seed is None:
            print(f"{occasion_slug:<16} {gender:<6} {label:<14}  (no seed)")
            continue

        name = seed.get("prod_name") or seed.get("display_name") or "?"
        dedup_key = (occasion_slug, gender, name)
        passed = is_coherent_candidate(seed, occasion_slug, gender, "top")
        status = "PASS" if passed else "LEAK"
        tag = "" if dedup_key not in seen_anchors else "  (repeat anchor)"
        seen_anchors.add(dedup_key)
        print(f"{occasion_slug:<16} {gender:<6} {label:<14}  {name[:50]:<50}  {status}{tag}")
        if not passed:
            leaks.append((occasion_slug, gender, label, name, seed.get("article_id")))

    print(f"\nTotal probes: {len(_PROBE_MATRIX)}. Leaks found: {len(leaks)}.")
    print(f"Distinct anchors seen: {len(seen_anchors)}")
    if leaks:
        print("\nLeak detail:")
        for occasion_slug, gender, label, name, aid in leaks:
            print(f"  {occasion_slug}/{gender}/{label}: {name} (article_id={aid})")


if __name__ == "__main__":
    main()
