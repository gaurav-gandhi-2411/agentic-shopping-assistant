#!/usr/bin/env python3
"""
Phase 2 Day 2.5 — Build v3 router training dataset.

Sources:
  data/router_dataset.jsonl              — original v1 base queries (388 examples)
  data/router_training_v2_contrastive.jsonl — 110 new clarify/search contrastive pairs

Process:
  1. Apply the same stateful augmentation as v2 to the v1 base queries
  2. Add contrastive pairs as additional base queries (no further augmentation)
  3. Split at BASE-QUERY level (all augmented variants of a base query go to the same split)
  4. Write v3 train/val/test splits

Key guarantee: no base-query string appears in both train and test.

Output:
  data/router_dataset_v3.jsonl
  data/router_dataset_v3_{train,val,test}.jsonl
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
random.seed(SEED)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

SRC_V1        = DATA_DIR / "router_dataset.jsonl"
CONTRASTIVE   = DATA_DIR / "router_training_v2_contrastive.jsonl"
OUT_ALL       = DATA_DIR / "router_dataset_v3.jsonl"
OUT_TRAIN     = DATA_DIR / "router_dataset_v3_train.jsonl"
OUT_VAL       = DATA_DIR / "router_dataset_v3_val.jsonl"
OUT_TEST      = DATA_DIR / "router_dataset_v3_test.jsonl"

# Realistic active_filter values (same as v2)
COLOUR_FILTERS = [
    {"colour_group_name": "Black"},
    {"colour_group_name": "Blue"},
    {"colour_group_name": "Red"},
    {"colour_group_name": "White"},
    {"colour_group_name": "Dark Blue"},
    {"colour_group_name": "Light Pink"},
]
TYPE_FILTERS = [
    {"product_type_name": "Dress"},
    {"product_type_name": "Jacket"},
    {"product_type_name": "Trousers"},
    {"product_type_name": "Top"},
]
COMBINED_FILTERS = [
    {"colour_group_name": "Black", "product_type_name": "Dress"},
    {"colour_group_name": "Blue",  "product_type_name": "Jacket"},
    {"colour_group_name": "White", "product_type_name": "Top"},
]
ALL_FILTER_OPTIONS = COLOUR_FILTERS + TYPE_FILTERS + COMBINED_FILTERS


def _filter_for(idx: int) -> dict:
    return ALL_FILTER_OPTIONS[idx % len(ALL_FILTER_OPTIONS)]


def _colour_filter(idx: int) -> dict:
    return COLOUR_FILTERS[idx % len(COLOUR_FILTERS)]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def make_aug(src: dict, aug_id: str, last_action: str,
             items_retrieved: int, active_filters: dict, route: str) -> dict:
    return {
        "query":           src["query"],
        "last_action":     last_action,
        "items_retrieved": items_retrieved,
        "active_filters":  active_filters,
        "route":           route,
        "id":              aug_id,
        "source":          "augmented",
    }


def augment_v1_examples(examples: list[dict]) -> list[dict]:
    """Identical augmentation logic to v2 — same coverage gaps closed."""
    augmented: list[dict] = []
    by_route: dict[str, list[dict]] = {}
    for ex in examples:
        by_route.setdefault(ex["route"], []).append(ex)

    search_exs  = by_route.get("search",  [])
    filter_exs  = by_route.get("filter",  [])
    compare_exs = by_route.get("compare", [])
    outfit_exs  = by_route.get("outfit",  [])
    respond_exs = by_route.get("respond", [])
    clarify_exs = by_route.get("clarify", [])

    # GAP 1A: la=respond → search
    for i, ex in enumerate(search_exs):
        augmented.append(make_aug(ex, f"aug_resp_srch_{i:03d}a", "respond", 5, {}, "search"))
        augmented.append(make_aug(ex, f"aug_resp_srch_{i:03d}b", "respond", 1, {}, "search"))

    # GAP 1B: la=respond → clarify
    for i, ex in enumerate(clarify_exs):
        augmented.append(make_aug(ex, f"aug_resp_clry_{i:03d}", "respond", 5, {}, "clarify"))

    # GAP 2: la=clarify → search
    for i, ex in enumerate(search_exs):
        augmented.append(make_aug(ex, f"aug_clry_srch_{i:03d}", "clarify", 0, {}, "search"))

    # GAP 3A: active_filters for filter route
    for i, ex in enumerate(filter_exs):
        augmented.append(make_aug(ex, f"aug_filt_actf_{i:03d}a", "filter",  3, _colour_filter(i), "filter"))
        augmented.append(make_aug(ex, f"aug_filt_srch_{i:03d}b", "search",  7, {},                "filter"))

    # GAP 3B: active_filters for compare/outfit
    for i, ex in enumerate(compare_exs):
        augmented.append(make_aug(ex, f"aug_cmp_actf_{i:03d}", "filter", 6, _colour_filter(i), "compare"))

    for i, ex in enumerate(outfit_exs):
        augmented.append(make_aug(ex, f"aug_out_actf_{i:03d}", "filter", 4, _filter_for(i), "outfit"))

    # GAP 4: items=1,3 for respond
    for i, ex in enumerate(respond_exs):
        augmented.append(make_aug(ex, f"aug_resp_i1_{i:03d}", "search", 1, {},               "respond"))
        augmented.append(make_aug(ex, f"aug_resp_i3_{i:03d}", "filter", 3, _colour_filter(i), "respond"))

    # la=compare → search
    sampled_cmp = random.sample(search_exs, min(50, len(search_exs)))
    for i, ex in enumerate(sampled_cmp):
        augmented.append(make_aug(ex, f"aug_cmp_to_srch_{i:03d}", "compare", 6, {}, "search"))

    # la=outfit → search
    sampled_out = random.sample(search_exs, min(40, len(search_exs)))
    for i, ex in enumerate(sampled_out):
        augmented.append(make_aug(ex, f"aug_out_to_srch_{i:03d}", "outfit", 5, {}, "search"))

    return augmented


def split_at_base_query_level(
    originals: list[dict],
    augmented: list[dict],
    val_frac:  float = 0.10,
    test_frac: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """All augmented variants of a base query follow that query's split assignment."""
    aug_by_query: dict[str, list[dict]] = {}
    for ex in augmented:
        key = ex["query"].strip().lower()
        aug_by_query.setdefault(key, []).append(ex)

    orig_copy = originals.copy()
    random.shuffle(orig_copy)

    n      = len(orig_copy)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))

    test_orig  = orig_copy[:n_test]
    val_orig   = orig_copy[n_test:n_test + n_val]
    train_orig = orig_copy[n_test + n_val:]

    def collect(subset: list[dict]) -> list[dict]:
        result = list(subset)
        for ex in subset:
            key = ex["query"].strip().lower()
            result.extend(aug_by_query.get(key, []))
        random.shuffle(result)
        return result

    return collect(train_orig), collect(val_orig), collect(test_orig)


def verify_no_base_query_overlap(train: list[dict], test: list[dict]) -> None:
    train_queries = {ex["query"].strip().lower() for ex in train}
    test_queries  = {ex["query"].strip().lower() for ex in test}
    overlap = train_queries & test_queries
    if overlap:
        print(f"  WARNING: {len(overlap)} base-query strings in both train and test!")
        for q in list(overlap)[:5]:
            print(f"    {q!r}")
    else:
        print(f"  Base-query overlap: 0 / {len(test_queries)} test queries — CLEAN")


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def print_stats(label: str, examples: list[dict]) -> None:
    route_counts = Counter(ex["route"] for ex in examples)
    source_counts = Counter(ex.get("source", "?") for ex in examples)
    stateful = sum(1 for ex in examples if ex["last_action"] != "none" or ex["items_retrieved"] > 0)
    print(f"\n  {label}: {len(examples)} examples")
    print(f"    Routes:  " + "  ".join(f"{r}={c}" for r, c in sorted(route_counts.items())))
    print(f"    Sources: " + "  ".join(f"{s}={c}" for s, c in sorted(source_counts.items())))
    print(f"    Stateful: {stateful} ({100*stateful//max(1,len(examples))}%)")


def main() -> None:
    print("Loading v1 base queries …")
    v1_originals = load_jsonl(SRC_V1)
    print(f"  {len(v1_originals)} v1 examples")

    print("Loading contrastive pairs …")
    contrastive = load_jsonl(CONTRASTIVE)
    print(f"  {len(contrastive)} contrastive pairs")
    contrast_dist = Counter(ex["route"] for ex in contrastive)
    print(f"  Distribution: {dict(contrast_dist)}")

    print("\nGenerating stateful augmentation of v1 examples …")
    augmented = augment_v1_examples(v1_originals)
    print(f"  {len(augmented)} augmented examples")

    # Contrastive pairs are standalone base queries (no augmented variants).
    # They go into the split as if they are originals with aug_by_query = [].
    all_originals = v1_originals + contrastive
    print(f"\nTotal base queries: {len(all_originals)} ({len(v1_originals)} v1 + {len(contrastive)} contrastive)")

    print("\nSplitting at base-query level (80/10/10) …")
    train, val, test = split_at_base_query_level(all_originals, augmented)

    print("\nVerifying train/test base-query overlap …")
    verify_no_base_query_overlap(train, test)
    print("Verifying val/test base-query overlap …")
    verify_no_base_query_overlap(val, test)

    # Write outputs
    all_examples = train + val + test
    random.shuffle(all_examples)
    write_jsonl(OUT_ALL,   all_examples)
    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_VAL,   val)
    write_jsonl(OUT_TEST,  test)

    print("\n=== V3 Dataset Summary ===")
    print_stats("Train", train)
    print_stats("Val",   val)
    print_stats("Test",  test)

    # Check clarify representation specifically
    print("\n=== Clarify class detail ===")
    for split_name, split in [("Train", train), ("Val", val), ("Test", test)]:
        clarify_exs = [ex for ex in split if ex["route"] == "clarify"]
        contrastive_clarify = [ex for ex in clarify_exs if ex.get("source") == "contrastive"]
        print(f"  {split_name}: {len(clarify_exs)} clarify ({len(contrastive_clarify)} from contrastive)")

    print(f"\nWrote: {OUT_TRAIN} ({len(train)}), {OUT_VAL} ({len(val)}), {OUT_TEST} ({len(test)})")


if __name__ == "__main__":
    main()
