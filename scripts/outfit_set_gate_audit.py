"""Count real multi-piece sets that the anchor garment gate rejects (public catalogue view).

A row that ``is_multi_piece_set`` is a whole outfit, so ``_is_outfit_garment`` rejecting it as a
seed would be a false rejection (e.g. a kurta-churidar-dupatta set typed ``dupatta``). Reports
counts by store and by product_type, and as a fraction of all set rows. Read-only; no .env.

    python scripts/outfit_set_gate_audit.py --index-dir <dir> [--examples 15]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        sys.exit("INCLUDE_NON_PUBLIC_STORES is set; this audit is the public view only")

    import pandas as pd

    from src.agents.outfit.composer import _is_outfit_garment
    from src.agents.outfit.slots import is_multi_piece_set
    from src.config.store_registry import public_stores

    df = pd.read_parquet(
        args.index_dir / "catalogue.parquet",
        columns=["article_id", "prod_name", "product_type_name", "store"],
    )
    df = df[df["store"].fillna("").astype(str).str.lower().isin(public_stores())]
    sets = rejected = 0
    by_store: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    set_by_store: Counter[str] = Counter()
    examples: list[tuple[str, str, str]] = []
    for pt_raw, nm_raw, store in zip(
        df["product_type_name"], df["prod_name"], df["store"], strict=True
    ):
        pt = "" if pd.isna(pt_raw) else str(pt_raw)
        nm = "" if pd.isna(nm_raw) else str(nm_raw)
        if not is_multi_piece_set(pt, nm):
            continue
        sets += 1
        set_by_store[str(store)] += 1
        if not _is_outfit_garment(pt, nm):
            rejected += 1
            by_store[str(store)] += 1
            by_type[pt.lower()] += 1
            if len(examples) < args.examples:
                examples.append((str(store), pt, nm))
    print(f"public rows={len(df)} set rows={sets} sets rejected by _is_outfit_garment={rejected}")
    print(f"fraction of set rows rejected: {rejected / sets:.4%}" if sets else "no set rows")
    print("by store (rejected / set rows):")
    for s, n in by_store.most_common():
        print(f"  {s}: {n} / {set_by_store[s]}")
    print("by product_type (rejected):")
    for t, n in by_type.most_common(30):
        print(f"  {t!r}: {n}")
    print("examples (store | product_type | name):")
    for e in examples:
        print("  ", " | ".join(e))


if __name__ == "__main__":
    main()
