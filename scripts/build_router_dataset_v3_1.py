#!/usr/bin/env python3
"""
Build V3.1 router training dataset.

Sources:
  data/router_dataset.jsonl                       — v1 base queries (388)
  data/router_training_v2_contrastive.jsonl       — Day 2.5 clarify/search pairs (110)
  data/router_training_v3_contrastive_v3_1.jsonl  — V3.1 event-verb pairs (20)

All contrastive pairs are base queries (no augmented variants).
Augmentation applied only to v1 examples (same logic as v2/v3).

Overlap guarantees:
  - Base-query-level split: all variants of a base query go to one split
  - OOD-10 queries are NOT in this dataset (verified by generate_contrastive_v3_1.py)

Output: data/router_dataset_v3_1_{train,val,test}.jsonl
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

SRC_V1          = DATA_DIR / "router_dataset.jsonl"
CONTRAST_V2_5   = DATA_DIR / "router_training_v2_contrastive.jsonl"
CONTRAST_V3_1   = DATA_DIR / "router_training_v3_contrastive_v3_1.jsonl"

OUT_TRAIN = DATA_DIR / "router_dataset_v3_1_train.jsonl"
OUT_VAL   = DATA_DIR / "router_dataset_v3_1_val.jsonl"
OUT_TEST  = DATA_DIR / "router_dataset_v3_1_test.jsonl"

# Augmentation filter/colour options (identical to v2/v3)
COLOUR_FILTERS = [
    {"colour_group_name": "Black"}, {"colour_group_name": "Blue"},
    {"colour_group_name": "Red"},   {"colour_group_name": "White"},
    {"colour_group_name": "Dark Blue"}, {"colour_group_name": "Light Pink"},
]
TYPE_FILTERS = [
    {"product_type_name": "Dress"}, {"product_type_name": "Jacket"},
    {"product_type_name": "Trousers"}, {"product_type_name": "Top"},
]
ALL_FILTER_OPTIONS = COLOUR_FILTERS + TYPE_FILTERS + [
    {"colour_group_name": "Black", "product_type_name": "Dress"},
    {"colour_group_name": "Blue",  "product_type_name": "Jacket"},
    {"colour_group_name": "White", "product_type_name": "Top"},
]


def _filter_for(idx): return ALL_FILTER_OPTIONS[idx % len(ALL_FILTER_OPTIONS)]
def _colour_filter(idx): return COLOUR_FILTERS[idx % len(COLOUR_FILTERS)]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def make_aug(src, aug_id, last_action, items_retrieved, active_filters, route):
    return {"query": src["query"], "last_action": last_action,
            "items_retrieved": items_retrieved, "active_filters": active_filters,
            "route": route, "id": aug_id, "source": "augmented"}


def augment(examples: list[dict]) -> list[dict]:
    augmented = []
    by_route: dict[str, list[dict]] = {}
    for ex in examples:
        by_route.setdefault(ex["route"], []).append(ex)

    search_exs  = by_route.get("search",  [])
    filter_exs  = by_route.get("filter",  [])
    compare_exs = by_route.get("compare", [])
    outfit_exs  = by_route.get("outfit",  [])
    respond_exs = by_route.get("respond", [])
    clarify_exs = by_route.get("clarify", [])

    for i, ex in enumerate(search_exs):
        augmented.append(make_aug(ex, f"aug_resp_srch_{i:03d}a", "respond", 5, {}, "search"))
        augmented.append(make_aug(ex, f"aug_resp_srch_{i:03d}b", "respond", 1, {}, "search"))
    for i, ex in enumerate(clarify_exs):
        augmented.append(make_aug(ex, f"aug_resp_clry_{i:03d}", "respond", 5, {}, "clarify"))
    for i, ex in enumerate(search_exs):
        augmented.append(make_aug(ex, f"aug_clry_srch_{i:03d}", "clarify", 0, {}, "search"))
    for i, ex in enumerate(filter_exs):
        augmented.append(make_aug(ex, f"aug_filt_actf_{i:03d}a", "filter",  3, _colour_filter(i), "filter"))
        augmented.append(make_aug(ex, f"aug_filt_srch_{i:03d}b", "search",  7, {},                "filter"))
    for i, ex in enumerate(compare_exs):
        augmented.append(make_aug(ex, f"aug_cmp_actf_{i:03d}", "filter", 6, _colour_filter(i), "compare"))
    for i, ex in enumerate(outfit_exs):
        augmented.append(make_aug(ex, f"aug_out_actf_{i:03d}", "filter", 4, _filter_for(i), "outfit"))
    for i, ex in enumerate(respond_exs):
        augmented.append(make_aug(ex, f"aug_resp_i1_{i:03d}", "search", 1, {},               "respond"))
        augmented.append(make_aug(ex, f"aug_resp_i3_{i:03d}", "filter", 3, _colour_filter(i), "respond"))
    for i, ex in enumerate(random.sample(search_exs, min(50, len(search_exs)))):
        augmented.append(make_aug(ex, f"aug_cmp_to_srch_{i:03d}", "compare", 6, {}, "search"))
    for i, ex in enumerate(random.sample(search_exs, min(40, len(search_exs)))):
        augmented.append(make_aug(ex, f"aug_out_to_srch_{i:03d}", "outfit", 5, {}, "search"))
    return augmented


def split_at_base_query_level(originals, augmented, val_frac=0.10, test_frac=0.10):
    aug_by_query: dict[str, list[dict]] = {}
    for ex in augmented:
        aug_by_query.setdefault(ex["query"].strip().lower(), []).append(ex)

    # Group originals by query string so duplicate-query edge cases
    # (same surface, different context/route) stay in the same split.
    orig_by_query: dict[str, list[dict]] = {}
    for ex in originals:
        orig_by_query.setdefault(ex["query"].strip().lower(), []).append(ex)

    query_groups = list(orig_by_query.items())  # [(query_str, [examples, ...])]
    random.shuffle(query_groups)
    n = len(query_groups)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))

    def collect(group_slice):
        result = []
        for qstr, exs in group_slice:
            result.extend(exs)
            result.extend(aug_by_query.get(qstr, []))
        random.shuffle(result)
        return result

    test_groups  = query_groups[:n_test]
    val_groups   = query_groups[n_test:n_test + n_val]
    train_groups = query_groups[n_test + n_val:]
    return collect(train_groups), collect(val_groups), collect(test_groups)


def verify_overlap(a_name, a, b_name, b):
    a_q = {ex["query"].strip().lower() for ex in a}
    b_q = {ex["query"].strip().lower() for ex in b}
    overlap = a_q & b_q
    if overlap:
        print(f"  WARNING: {len(overlap)} overlap between {a_name} and {b_name}!")
        for q in list(overlap)[:3]: print(f"    {q!r}")
    else:
        print(f"  {a_name}/{b_name} overlap: 0 — CLEAN")


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main():
    print("Loading sources …")
    v1       = load_jsonl(SRC_V1)
    c_v25    = load_jsonl(CONTRAST_V2_5)
    c_v31    = load_jsonl(CONTRAST_V3_1)
    print(f"  v1={len(v1)}  contrastive_v2.5={len(c_v25)}  contrastive_v3.1={len(c_v31)}")

    print("Generating augmentation of v1 examples …")
    augmented = augment(v1)
    print(f"  {len(augmented)} augmented examples")

    all_originals = v1 + c_v25 + c_v31
    print(f"Total base queries: {len(all_originals)}")

    print("Splitting at base-query level (80/10/10) …")
    train, val, test = split_at_base_query_level(all_originals, augmented)

    print("Verifying overlaps …")
    verify_overlap("train", train, "test",  test)
    verify_overlap("val",   val,   "test",  test)
    verify_overlap("train", train, "val",   val)

    # Verify OOD-10 not in training
    ood10 = load_jsonl(DATA_DIR / "router_ood10_event_verb.jsonl")
    ood10_q = {ex["query"].strip().lower() for ex in ood10}
    train_q = {ex["query"].strip().lower() for ex in train}
    ood10_overlap = ood10_q & train_q
    if ood10_overlap:
        print(f"  WARNING: OOD-10 overlap with train: {ood10_overlap}")
    else:
        print(f"  OOD-10/train overlap: 0 — CLEAN")

    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_VAL,   val)
    write_jsonl(OUT_TEST,  test)

    from collections import Counter
    for label, split in [("Train", train), ("Val", val), ("Test", test)]:
        routes = Counter(ex["route"] for ex in split)
        clarify_src = Counter(ex.get("source","?") for ex in split if ex["route"] == "clarify")
        print(f"  {label}: n={len(split)}  routes={dict(sorted(routes.items()))}")
        print(f"    clarify sources: {dict(clarify_src)}")

    print(f"\nWrote: {OUT_TRAIN} ({len(train)}), {OUT_VAL} ({len(val)}), {OUT_TEST} ({len(test)})")


if __name__ == "__main__":
    main()
