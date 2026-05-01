"""
Day 1 — Phase 2 data augmentation for DistilBERT router retraining.

Generates stateful variants of existing training examples to close
four coverage gaps discovered during train/serve skew analysis:

  GAP 1: la=respond as prior state (0 examples)
  GAP 2: la=clarify as prior state (0 examples)
  GAP 3: active_filters in state context (only 12/388)
  GAP 4: items_retrieved values 1, 3 (never used)

All augmented labels are derived by deterministic rules — no LLM needed.
State-to-route rules match production graph.py routing logic.

Output: data/router_dataset_v2.jsonl  (original + augmented, shuffled)
        data/router_dataset_v2_{train,val,test}.jsonl
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR  = Path(__file__).parent.parent / "data"
SRC_FILE  = DATA_DIR / "router_dataset.jsonl"
OUT_FILE  = DATA_DIR / "router_dataset_v2.jsonl"
TRAIN_OUT = DATA_DIR / "router_dataset_v2_train.jsonl"
VAL_OUT   = DATA_DIR / "router_dataset_v2_val.jsonl"
TEST_OUT  = DATA_DIR / "router_dataset_v2_test.jsonl"

# Realistic active_filter values sampled from the H&M catalogue
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


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def make_example(source_ex: dict, aug_id: str, last_action: str,
                 items_retrieved: int, active_filters: dict, route: str) -> dict:
    return {
        "query":            source_ex["query"],
        "last_action":      last_action,
        "items_retrieved":  items_retrieved,
        "active_filters":   active_filters,
        "route":            route,
        "id":               aug_id,
        "source":           "augmented",
    }


def augment(examples: list[dict]) -> list[dict]:
    augmented: list[dict] = []

    # Index by route for targeted augmentation
    by_route: dict[str, list[dict]] = {}
    for ex in examples:
        by_route.setdefault(ex["route"], []).append(ex)

    search_exs  = by_route.get("search",  [])
    filter_exs  = by_route.get("filter",  [])
    compare_exs = by_route.get("compare", [])
    outfit_exs  = by_route.get("outfit",  [])
    respond_exs = by_route.get("respond", [])
    clarify_exs = by_route.get("clarify", [])

    # ------------------------------------------------------------------
    # GAP 1A: la=respond → search
    # Real scenario: user got a respond turn, now asks for new search.
    # All search queries are valid here with varied items counts.
    # ------------------------------------------------------------------
    for i, ex in enumerate(search_exs):
        # la=respond, items=5 (had results), no filters
        augmented.append(make_example(
            ex, f"aug_resp_srch_{i:03d}a",
            last_action="respond", items_retrieved=5, active_filters={}, route="search",
        ))
        # la=respond, items=1 (sparse results → covers items=1 gap)
        augmented.append(make_example(
            ex, f"aug_resp_srch_{i:03d}b",
            last_action="respond", items_retrieved=1, active_filters={}, route="search",
        ))

    # ------------------------------------------------------------------
    # GAP 1B: la=respond → clarify
    # Real scenario: agent responded but user query is still ambiguous.
    # ------------------------------------------------------------------
    for i, ex in enumerate(clarify_exs):
        augmented.append(make_example(
            ex, f"aug_resp_clry_{i:03d}",
            last_action="respond", items_retrieved=5, active_filters={}, route="clarify",
        ))

    # ------------------------------------------------------------------
    # GAP 2: la=clarify → search
    # Real scenario: user answered clarify question with a search intent.
    # ------------------------------------------------------------------
    for i, ex in enumerate(search_exs):
        augmented.append(make_example(
            ex, f"aug_clry_srch_{i:03d}",
            last_action="clarify", items_retrieved=0, active_filters={}, route="search",
        ))

    # ------------------------------------------------------------------
    # GAP 3A: active_filters in state for filter route
    # Real scenario: user already applied a colour filter, now narrows by type.
    # ------------------------------------------------------------------
    for i, ex in enumerate(filter_exs):
        # la=filter, items=3 (covers items=3 gap), active colour filter in context
        augmented.append(make_example(
            ex, f"aug_filt_actf_{i:03d}a",
            last_action="filter", items_retrieved=3,
            active_filters=_colour_filter(i), route="filter",
        ))
        # la=search, items=7, no active filters (natural pre-filter state)
        augmented.append(make_example(
            ex, f"aug_filt_srch_{i:03d}b",
            last_action="search", items_retrieved=7, active_filters={}, route="filter",
        ))

    # ------------------------------------------------------------------
    # GAP 3B: active_filters in state for compare/outfit
    # ------------------------------------------------------------------
    for i, ex in enumerate(compare_exs):
        augmented.append(make_example(
            ex, f"aug_cmp_actf_{i:03d}",
            last_action="filter", items_retrieved=6,
            active_filters=_colour_filter(i), route="compare",
        ))

    for i, ex in enumerate(outfit_exs):
        augmented.append(make_example(
            ex, f"aug_out_actf_{i:03d}",
            last_action="filter", items_retrieved=4,
            active_filters=_filter_for(i), route="outfit",
        ))

    # ------------------------------------------------------------------
    # GAP 4: items_retrieved=1,3 for respond
    # Real scenario: sparse retrieval, user still asks a question.
    # ------------------------------------------------------------------
    for i, ex in enumerate(respond_exs):
        # items=1
        augmented.append(make_example(
            ex, f"aug_resp_i1_{i:03d}",
            last_action="search", items_retrieved=1, active_filters={}, route="respond",
        ))
        # items=3 with active filter
        augmented.append(make_example(
            ex, f"aug_resp_i3_{i:03d}",
            last_action="filter", items_retrieved=3,
            active_filters=_colour_filter(i), route="respond",
        ))

    # ------------------------------------------------------------------
    # la=compare → search (missing transition)
    # Real scenario: user compared items, now wants a fresh search.
    # ------------------------------------------------------------------
    # Sample a subset of search examples to avoid over-representing search
    sampled_for_compare = random.sample(search_exs, min(50, len(search_exs)))
    for i, ex in enumerate(sampled_for_compare):
        augmented.append(make_example(
            ex, f"aug_cmp_to_srch_{i:03d}",
            last_action="compare", items_retrieved=6, active_filters={}, route="search",
        ))

    # ------------------------------------------------------------------
    # la=outfit → search (missing transition)
    # ------------------------------------------------------------------
    sampled_for_outfit = random.sample(search_exs, min(40, len(search_exs)))
    for i, ex in enumerate(sampled_for_outfit):
        augmented.append(make_example(
            ex, f"aug_out_to_srch_{i:03d}",
            last_action="outfit", items_retrieved=5, active_filters={}, route="search",
        ))

    return augmented


def split_dataset_by_query(
    originals: list[dict],
    augmented: list[dict],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by BASE QUERY, not by example.

    All augmented variants of a query follow their source query's split
    assignment, so the same surface query never appears in both train and test.
    """
    # Index augmented examples by their source query string
    aug_by_query: dict[str, list[dict]] = {}
    for ex in augmented:
        key = ex["query"].strip().lower()
        aug_by_query.setdefault(key, []).append(ex)

    # Shuffle the original (base) queries as a group
    orig_copy = originals.copy()
    random.shuffle(orig_copy)

    n = len(orig_copy)
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


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def print_stats(label: str, examples: list[dict]) -> None:
    route_counts = Counter(ex["route"] for ex in examples)
    stateful = sum(
        1 for ex in examples
        if ex["last_action"] != "none" or ex["items_retrieved"] > 0
    )
    la_counts = Counter(ex["last_action"] for ex in examples)
    filter_count = sum(1 for ex in examples if ex["active_filters"])
    items_counts = Counter(ex["items_retrieved"] for ex in examples)
    aug_count = sum(1 for ex in examples if ex.get("source") == "augmented")

    print(f"\n{'='*60}")
    print(f"{label}  ({len(examples)} examples, {aug_count} augmented)")
    print(f"{'='*60}")
    print(f"  Stateful (la!=none or items>0): {stateful} ({100*stateful//len(examples)}%)")
    print(f"  Route distribution:")
    for route, cnt in sorted(route_counts.items()):
        print(f"    {route:10s}: {cnt}")
    print(f"  last_action priors:")
    for la, cnt in sorted(la_counts.items(), key=lambda x: -x[1]):
        print(f"    {la:12s}: {cnt}")
    print(f"  items_retrieved values: {dict(sorted(items_counts.items()))}")
    print(f"  With active_filters: {filter_count}")


def main() -> None:
    print(f"Loading {SRC_FILE} ...")
    originals = load_examples(SRC_FILE)
    print(f"  {len(originals)} original examples")

    augmented = augment(originals)
    print(f"  {len(augmented)} augmented examples generated")

    all_examples = originals + augmented
    random.shuffle(all_examples)

    print_stats("FULL v2 dataset", all_examples)

    train, val, test = split_dataset_by_query(originals, augmented)
    print_stats("TRAIN", train)
    print_stats("VAL",   val)
    print_stats("TEST",  test)

    write_jsonl(OUT_FILE,   all_examples)
    write_jsonl(TRAIN_OUT,  train)
    write_jsonl(VAL_OUT,    val)
    write_jsonl(TEST_OUT,   test)

    print(f"\nWrote:")
    print(f"  {OUT_FILE}   ({len(all_examples)} examples)")
    print(f"  {TRAIN_OUT}  ({len(train)} examples)")
    print(f"  {VAL_OUT}    ({len(val)} examples)")
    print(f"  {TEST_OUT}   ({len(test)} examples)")


if __name__ == "__main__":
    main()
