#!/usr/bin/env python3
"""
Build OOD-10: held-out evaluation set for "I want to [event-verb] for [occasion]" pattern.

These queries are EXCLUDED from V3.1 training data. They are written first
to ensure no overlap with the contrastive training pairs.

All 10 are unambiguous search queries where occasion implies clear shopping intent.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "router_ood10_event_verb.jsonl"

EXAMPLES = [
    {"query": "I want to look great for a job interview",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_00"},

    {"query": "I need something to wear to a birthday party",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_01"},

    {"query": "I'm going to a wedding and need an outfit",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_02"},

    {"query": "looking to impress at a first date",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_03"},

    {"query": "I want to wear something special for my anniversary dinner",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_04"},

    {"query": "I'm attending a conference and want to look sharp",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_05"},

    {"query": "I need to get dressed up for a graduation ceremony",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_06"},

    {"query": "I have a concert tonight and need an outfit",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_07"},

    {"query": "I want to dress nicely for Sunday brunch with family",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_08"},

    {"query": "want to look put together for a casual Friday at the office",
     "last_action": "none", "items_retrieved": 0, "active_filters": {},
     "route": "search", "id": "ood10_09"},
]


def main() -> None:
    for ex in EXAMPLES:
        ex["source"] = "ood10"
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(EXAMPLES)} OOD-10 queries -> {OUT}")
    print("All routes:", set(ex['route'] for ex in EXAMPLES))


if __name__ == "__main__":
    main()
