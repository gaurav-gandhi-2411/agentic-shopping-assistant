"""Budget grid for occasion-driven outfit composition against the local unified index.

For every (gender, occasion, budget) cell, composes a look (no LLM) and reports the card count
(seed + complements), the look total, and which required slots stayed empty. Used to choose the
anchor-price fraction from data and to document which cells the catalogue cannot fill.

    python scripts/outfit_budget_grid.py --index-dir <dir> [--fractions 0.4,0.5,0.6] [--json out]

The index is the PUBLIC view (INCLUDE_NON_PUBLIC_STORES must be unset). No .env is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BUDGETS: tuple[int, ...] = (1000, 1500, 2000, 3000, 5000, 8000)
OCCASIONS: tuple[str, ...] = ("sangeet", "wedding_guest", "office")
GENDERS: tuple[str, ...] = ("women", "men")


def run_grid(retriever: Any, df: Any) -> list[dict]:
    """Compose every cell; return one row per cell."""
    from src.agents.outfit.composer import compose_outfit

    rows = []
    for gender in GENDERS:
        for occasion in OCCASIONS:
            for budget in BUDGETS:
                look = compose_outfit(
                    df, retriever, occasion_slug=occasion, gender=gender, budget_inr=float(budget)
                )
                seed = look.get("seed_item")
                comps = look.get("complements") or []
                rows.append(
                    {
                        "gender": gender,
                        "occasion": occasion,
                        "budget": budget,
                        "cards": (1 if seed else 0) + len(comps),
                        "seed_price": (seed or {}).get("price_inr"),
                        "total": look.get("budget_total_inr"),
                        "slots": [c.get("_slot") for c in comps],
                        "empty_slots": list(look.get("empty_slots") or []),
                        "ids": ([str(seed["article_id"])] if seed else [])
                        + [str(c["article_id"]) for c in comps],
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument("--fractions", default="")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        sys.exit("INCLUDE_NON_PUBLIC_STORES is set; this grid is the public view only")

    from scripts import outfit_compose_snapshot as snap
    from src.agents.outfit import composer

    retriever, df = snap.load_retriever(args.index_dir)
    fractions: list[float | None] = [float(f) for f in args.fractions.split(",") if f] or [None]
    out: dict[str, list[dict]] = {}
    for frac in fractions:
        if frac is not None:
            composer.ANCHOR_BUDGET_FRACTION = frac
        rows = run_grid(retriever, df)
        out[str(frac)] = rows
        ok = sum(1 for r in rows if r["cards"] >= 3)
        over = [r for r in rows if r["total"] is not None and r["total"] > r["budget"]]
        print(
            f"fraction={frac}: {ok}/{len(rows)} cells with >=3 cards; over-budget cells={len(over)}"
        )
        for r in rows:
            print(
                f"  {r['gender']:5} {r['occasion']:14} {r['budget']:>5}: cards={r['cards']} "
                f"total={r['total']} seed={r['seed_price']} slots={r['slots']} "
                f"empty={r['empty_slots']}"
            )
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
