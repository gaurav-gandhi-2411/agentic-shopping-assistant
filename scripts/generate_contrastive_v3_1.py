#!/usr/bin/env python3
"""
V3.1 contrastive pairs — "I want to [event-verb] for [occasion]" boundary.

20 pairs: 15 search + 5 clarify.

Occasions used here are DIFFERENT from OOD-10 (job interview, birthday party,
wedding, first date, anniversary dinner, conference, graduation, concert,
Sunday brunch, casual Friday) to ensure zero contamination.

Rule:
  SEARCH = desire expression + specific occasion → implied shopping intent
  CLARIFY = desire expression + no occasion OR occasion so vague no item is implied
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "router_training_v3_contrastive_v3_1.jsonl"

PAIRS: list[tuple[str, str]] = [
    # ---- SEARCH (15) — occasion implies clear shopping intent ----
    ("I want to dress up for brunch",                              "search"),
    ("I need a look for a rooftop bar",                            "search"),
    ("help me put together something for a date night",            "search"),
    ("I want to wear something nice to an art gallery opening",    "search"),
    ("need to look professional for a client meeting",             "search"),
    ("I want to stand out at a company awards dinner",             "search"),
    ("I need an outfit for a friend's housewarming party",         "search"),
    ("I want to look polished for a networking event",             "search"),
    ("help me find something to wear to a baby shower",            "search"),
    ("I need to dress up for my partner's work gala",              "search"),
    ("I want to look stylish for a wine tasting evening",          "search"),
    ("I need something for a charity fundraiser this weekend",     "search"),
    ("I want to dress appropriately for a university open day",    "search"),
    ("something to wear for a farewell party at work",             "search"),
    ("I need to look great for a school reunion",                  "search"),

    # ---- CLARIFY (5) — no occasion, or occasion too vague to imply an item ----
    ("I want to dress up but I'm not sure for what",               "clarify"),
    ("I need a look but haven't decided where I'm going yet",      "clarify"),
    ("help me put something together",                             "clarify"),
    ("I want to wear something special but don't know the occasion", "clarify"),
    ("I need to dress up for something coming up, details TBD",    "clarify"),
]


def main() -> None:
    records = []
    for i, (query, route) in enumerate(PAIRS):
        records.append({
            "query":           query,
            "last_action":     "none",
            "items_retrieved": 0,
            "active_filters":  {},
            "route":           route,
            "id":              f"contrast_v31_{i:03d}",
            "source":          "contrastive_v31",
        })

    from collections import Counter
    dist = Counter(r["route"] for r in records)

    # Verify no query matches OOD-10
    ood10_path = ROOT / "data" / "router_ood10_event_verb.jsonl"
    ood10_queries = set()
    if ood10_path.exists():
        with open(ood10_path) as f:
            for line in f:
                ood10_queries.add(json.loads(line)["query"].strip().lower())
    overlaps = [r for r in records if r["query"].strip().lower() in ood10_queries]
    if overlaps:
        raise ValueError(f"Overlap with OOD-10: {[r['query'] for r in overlaps]}")
    print("OOD-10 overlap check: 0 overlaps — CLEAN")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} pairs -> {OUT}")
    print(f"  search: {dist['search']}  clarify: {dist['clarify']}")


if __name__ == "__main__":
    main()
