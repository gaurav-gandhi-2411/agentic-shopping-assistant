#!/usr/bin/env python3
"""
Add 20 targeted clarify training examples and re-split the dataset.

Addresses the diagnosis finding: model has 19 clarify train errors,
primarily on gift-intent and vague-preference queries.

Generates 5 seed examples + 3 paraphrases each = 20 new examples,
then performs a stratified 80/10/10 re-split.

Usage:
    python scripts/augment_clarify.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
JSONL    = DATA_DIR / "router_dataset.jsonl"

SEED = 42

# ---------------------------------------------------------------------------
# New clarify examples — targeting the 4 failure clusters from diagnosis
# ---------------------------------------------------------------------------

NEW_EXAMPLES: list[dict] = [
    # ── Group 1: Birthday / occasion gift, unknown preferences ───────────────
    {
        "query": "I'm shopping for a birthday gift but I'm not sure what to get",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "seed_053", "source": "seed",
    },
    {
        "query": "What should I get as a birthday present? I genuinely have no idea",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_053_0", "source": "paraphrase",
    },
    {
        "query": "Birthday gift needed but I'm completely clueless on what to choose",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_053_1", "source": "paraphrase",
    },
    {
        "query": "Shopping for a birthday, I can't decide what category to even start with",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_053_2", "source": "paraphrase",
    },

    # ── Group 2: Gift for specific person (sister, friend, family) ────────────
    {
        "query": "I'd like to find something nice for my sister but I'm not sure what she'd like",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "seed_054", "source": "seed",
    },
    {
        "query": "I want to buy something for a friend but have no clue what they'd want",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_054_0", "source": "paraphrase",
    },
    {
        "query": "My friend needs a gift from here — she has varied tastes and I need guidance",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_054_1", "source": "paraphrase",
    },
    {
        "query": "I need to buy something for a family member but have no idea what to get",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_054_2", "source": "paraphrase",
    },

    # ── Group 3: Vague wants ("something nice", no product type) ─────────────
    {
        "query": "I want to find something stylish but I'm not sure what type of thing",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "seed_055", "source": "seed",
    },
    {
        "query": "Looking for something fashionable, just not sure what category to start in",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_055_0", "source": "paraphrase",
    },
    {
        "query": "I'd like something nice, I'm open to suggestions if you can help narrow it down",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_055_1", "source": "paraphrase",
    },
    {
        "query": "Show me something stylish — I don't really have a specific item in mind",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_055_2", "source": "paraphrase",
    },

    # ── Group 4: General fashion help / "where do I start" ───────────────────
    {
        "query": "I need fashion advice but I'm not sure what I'm looking for",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "seed_056", "source": "seed",
    },
    {
        "query": "Can you help me decide what to buy? I'm overwhelmed and not sure where to start",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_056_0", "source": "paraphrase",
    },
    {
        "query": "Could you give me some direction? I want to update my wardrobe but I'm lost",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_056_1", "source": "paraphrase",
    },
    {
        "query": "I'm looking for styling help but I don't know what type of item I need",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_056_2", "source": "paraphrase",
    },

    # ── Group 5: Explicitly undecided ("I'm not sure what I want") ───────────
    {
        "query": "I'm not sure what I want — can you ask me some questions to help?",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "seed_057", "source": "seed",
    },
    {
        "query": "I'd like to refresh my wardrobe but I'm unsure what I'm looking for",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_057_0", "source": "paraphrase",
    },
    {
        "query": "Help me decide — I'm not sure if I want a top, dress, or something else",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_057_1", "source": "paraphrase",
    },
    {
        "query": "I want something new but I genuinely can't decide what to get",
        "last_action": "none", "items_retrieved": 0, "active_filters": {},
        "route": "clarify", "id": "para_057_2", "source": "paraphrase",
    },
]


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    examples: list[dict],
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples stratified by route. Returns (train, val, test)."""
    rng = random.Random(seed)

    by_class: dict[str, list[dict]] = {}
    for ex in examples:
        by_class.setdefault(ex["route"], []).append(ex)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val  = max(1, round(n * val_frac))
        n_test = max(1, round(n * (1 - train_frac - val_frac)))
        n_train = n - n_val - n_test

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    # Shuffle each split so class order is mixed
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples),
        encoding="utf-8",
    )


def main(dry_run: bool = False) -> None:
    # Load existing data
    existing = [
        json.loads(l)
        for l in JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"Existing examples: {len(existing)}")
    old_clarify = sum(1 for e in existing if e["route"] == "clarify")
    print(f"  clarify before: {old_clarify}")

    # Collision check
    existing_ids = {e["id"] for e in existing}
    new_ids = {e["id"] for e in NEW_EXAMPLES}
    collisions = existing_ids & new_ids
    if collisions:
        raise ValueError(f"ID collision: {collisions}")

    # Merge
    combined = existing + NEW_EXAMPLES
    new_clarify = sum(1 for e in combined if e["route"] == "clarify")
    print(f"  clarify after:  {new_clarify}")
    print(f"Total examples:   {len(combined)}")

    class_counts = Counter(e["route"] for e in combined)
    print("\nClass distribution after augmentation:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")

    # Re-split
    train, val, test = stratified_split(combined)

    train_cls = Counter(e["route"] for e in train)
    val_cls   = Counter(e["route"] for e in val)
    test_cls  = Counter(e["route"] for e in test)

    print(f"\nNew split: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  clarify -> train={train_cls['clarify']} val={val_cls['clarify']} test={test_cls['clarify']}")

    if dry_run:
        print("\n[DRY RUN] — no files written.")
        return

    # Write files
    write_jsonl(JSONL, combined)
    write_jsonl(DATA_DIR / "router_dataset_train.jsonl", train)
    write_jsonl(DATA_DIR / "router_dataset_val.jsonl", val)
    write_jsonl(DATA_DIR / "router_dataset_test.jsonl", test)

    print("\nFiles written:")
    print(f"  {JSONL}")
    print(f"  {DATA_DIR / 'router_dataset_train.jsonl'}")
    print(f"  {DATA_DIR / 'router_dataset_val.jsonl'}")
    print(f"  {DATA_DIR / 'router_dataset_test.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan, do not write files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
