#!/usr/bin/env python3
"""
Build OOD-30: 30 natural queries for generalization validation.

Criteria:
- No query from v1, v2, v3 training/test data
- No query from OOD-20
- Unambiguous ground-truth labels
- Covers all 6 route types at realistic proportions
- Specifically stresses the clarify/search boundary and stateful routing

Routes:  search=12, clarify=7, respond=5, filter=3, compare=2, outfit=1
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "router_ood30_test.jsonl"

EXAMPLES = [
    # ----------------------------------------
    # SEARCH (12) — first-turn, enough signal
    # ----------------------------------------
    {"query": "something sporty for the gym",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_00"},

    {"query": "I want to dress up for a brunch",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_01"},

    {"query": "classic workwear for women",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_02"},

    {"query": "show me autumn layers",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_03"},

    {"query": "what's new in denim this season",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_04"},

    {"query": "I need a look for a rooftop bar",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_05"},

    {"query": "something lightweight for hot weather",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_06"},

    {"query": "office appropriate but not stuffy",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_07"},

    {"query": "festival outfits for summer",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_08"},

    {"query": "I'm rebuilding my wardrobe from scratch, start with basics",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_09"},

    {"query": "sustainable everyday wear",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_10"},

    {"query": "cozy knits for winter evenings",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood30_11"},

    # ----------------------------------------
    # CLARIFY (7) — not enough signal to search
    # ----------------------------------------
    {"query": "I want to change my look completely",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "clarify", "id": "ood30_12"},

    {"query": "help me find something for her",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "clarify", "id": "ood30_13"},

    {"query": "I need a gift, no idea what though",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "clarify", "id": "ood30_14"},

    {"query": "shopping for someone else and I'm clueless",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "clarify", "id": "ood30_15"},

    {"query": "I want something stylish",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "clarify", "id": "ood30_16"},

    {"query": "can you just pick something for me",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "clarify", "id": "ood30_17"},

    {"query": "what would you recommend for someone like me",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "clarify", "id": "ood30_18"},

    # ----------------------------------------
    # RESPOND (5) — stateful, items in context
    # ----------------------------------------
    {"query": "tell me more about the second one",
     "last_action": "search", "items_retrieved": 5, "active_filters": {},
     "route": "respond", "id": "ood30_19"},

    {"query": "what material is it made from",
     "last_action": "search", "items_retrieved": 3, "active_filters": {},
     "route": "respond", "id": "ood30_20"},

    {"query": "is this machine washable",
     "last_action": "filter", "items_retrieved": 4, "active_filters": {"colour_group_name": "Black"},
     "route": "respond", "id": "ood30_21"},

    {"query": "which of these would you recommend",
     "last_action": "search", "items_retrieved": 5, "active_filters": {},
     "route": "respond", "id": "ood30_22"},

    {"query": "does it come in plus sizes",
     "last_action": "search", "items_retrieved": 5, "active_filters": {},
     "route": "respond", "id": "ood30_23"},

    # ----------------------------------------
    # FILTER (3) — refine after search
    # ----------------------------------------
    {"query": "only show me things under forty pounds",
     "last_action": "search", "items_retrieved": 8, "active_filters": {},
     "route": "filter", "id": "ood30_24"},

    {"query": "narrow it down to red ones",
     "last_action": "search", "items_retrieved": 6, "active_filters": {},
     "route": "filter", "id": "ood30_25"},

    {"query": "filter by size medium",
     "last_action": "filter", "items_retrieved": 5, "active_filters": {"colour_group_name": "Blue"},
     "route": "filter", "id": "ood30_26"},

    # ----------------------------------------
    # COMPARE (2) — explicit compare intent
    # ----------------------------------------
    {"query": "which one is better quality",
     "last_action": "search", "items_retrieved": 5, "active_filters": {},
     "route": "compare", "id": "ood30_27"},

    {"query": "compare the first two for me",
     "last_action": "search", "items_retrieved": 5, "active_filters": {},
     "route": "compare", "id": "ood30_28"},

    # ----------------------------------------
    # OUTFIT (1) — explicit outfit request with items
    # ----------------------------------------
    {"query": "what shoes would go with this dress",
     "last_action": "search", "items_retrieved": 3, "active_filters": {},
     "route": "outfit", "id": "ood30_29"},
]


def main() -> None:
    for ex in EXAMPLES:
        ex["source"] = "ood30"

    with open(OUT, "w", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex) + "\n")

    from collections import Counter
    dist = Counter(ex["route"] for ex in EXAMPLES)
    print(f"Wrote {len(EXAMPLES)} OOD-30 queries -> {OUT}")
    print(f"  Distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
