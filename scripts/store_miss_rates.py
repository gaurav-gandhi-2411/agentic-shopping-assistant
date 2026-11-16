#!/usr/bin/env python
"""Per-store miss rate from the hand-labelled strict gold set (read-only, no retrieval, no LLM).

Joins eval/fixtures/strict_gold_labels.yaml (every labelled (query, item) pair) to a unified
catalogue for the item's store, and prints per store: n labelled items, misses, miss rate with a
Wilson 95% interval, the miss reasons, and which categories (catalogue ``category``) the misses
fall in. The labelled items are the items retrieval returned for the gold queries when the labels
were written (they include non-public stores), so n per store is small and biased towards what the
retriever surfaces; it is a signal, not a store quality score.

Usage:
    python scripts/store_miss_rates.py --catalogue data/processed/unified/catalogue.parquet
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_LABELS = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (n > 0)."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def per_store_table(labels: list[dict], store_of: dict[str, str]) -> pd.DataFrame:
    """One row per store: n, misses, miss_rate, Wilson bounds, reasons."""
    n: Counter[str] = Counter()
    miss: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in labels:
        for it in entry["items"]:
            store = store_of.get(str(it["article_id"]), "?")
            n[store] += 1
            if not it["relevant"]:
                miss[store] += 1
                reasons[store][it.get("reason", "unspecified")] += 1
    rows = []
    for s in n:
        lo, hi = wilson(miss[s], n[s])
        rows.append(
            {"store": s, "n": n[s], "misses": miss[s], "miss_rate": miss[s] / n[s],
             "wilson_lo": lo, "wilson_hi": hi,
             "reasons": ", ".join(f"{r} {c}" for r, c in reasons[s].most_common())}
        )
    return pd.DataFrame(rows).sort_values(["miss_rate", "n"], ascending=[False, False])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogue", required=True, type=Path)
    args = ap.parse_args()
    cat = pd.read_parquet(args.catalogue, columns=["article_id", "store"])
    store_of = dict(zip(cat["article_id"].astype(str), cat["store"], strict=True))
    labels = yaml.safe_load(_LABELS.read_text(encoding="utf-8"))["labels"]
    table = per_store_table(labels, store_of)
    pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tot_n, tot_m = int(table["n"].sum()), int(table["misses"].sum())
    print(f"\nTOTAL n={tot_n} misses={tot_m} miss_rate={tot_m / tot_n:.3f}")


if __name__ == "__main__":
    main()
