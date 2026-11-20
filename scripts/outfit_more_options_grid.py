"""72-cell 'More options' grid: 6 budgets x 6 occasions x 2 genders on the local public index.

For every cell composes the occasion-driven look (no LLM) and builds the 'more options' row, then
audits every card with an INDEPENDENT name/type keyword scan (not the builder's own predicate) so a
leak in the builder's gate cannot hide itself. Prints per-cell counts and an audit summary.

    python scripts/outfit_more_options_grid.py --index-dir <dir> [--json out]

The index is the PUBLIC view (INCLUDE_NON_PUBLIC_STORES must be unset). No .env is loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BUDGETS: tuple[int, ...] = (800, 1000, 1500, 2000, 3000, 5000)
OCCASIONS: tuple[str, ...] = ("sangeet", "wedding_guest", "office", "haldi", "mehendi", "cocktail")
GENDERS: tuple[str, ...] = ("women", "men")

# Independent audit vocabulary (deliberately separate from the builder's own gate).
_AUDIT: dict[str, re.Pattern[str]] = {
    "footwear": re.compile(
        r"\b(jutti|juttis|jutis|mojari|mojaris|kolhapuri|kolhapuris|shoes?|sandals?|sneakers?|"
        r"heels?|loafers?|slippers?|chappals?|flip\s?flops?|sliders?|boots?|footwear|wedges|"
        r"clogs?|mules?|pumps)\b",
        re.I,
    ),
    "swimwear": re.compile(r"\b(bikini|swim\w*|swimsuit|monokini|trunks)\b", re.I),
    "innerwear": re.compile(
        r"\b(panty|panties|bra|bras|brief|briefs|lingerie|boxers?|innerwear|camisole|shapewear|"
        r"thong|vest|undergarment)\b",
        re.I,
    ),
    "nightwear": re.compile(r"\b(nightwear|nightsuit|night\s?suit|nighty|night\s?dress|sleepwear|loungewear)\b", re.I),
    "jewellery_bag": re.compile(
        r"\b(necklace|earrings?|jhumka\w*|bangles?|bracelet|pendant|ring|clutch|handbag|potli|"
        r"watch|belt|cufflinks?|brooch|rakhi|hamper)\b",
        re.I,
    ),
}
_STANDALONE_BOTTOM = re.compile(r"\b(pant|pants|palazzo|palazzos|trouser|trousers|jeans|shorts|skirt|pyjama|pajama|churidar|salwar|leggings?)\b", re.I)
_SET_WORDS = re.compile(r"\b(set|kurta|kurti|suit|lehenga|saree|sari|dress|gown|anarkali|sherwani|shirt|top|jacket|blazer|dupatta)\b", re.I)


def audit_card(card: dict) -> list[str]:
    """Independent categories this card belongs to (empty = looks like a plain garment)."""
    text = f"{card.get('product_type') or ''} | {card.get('prod_name') or ''}"
    hits = [k for k, rx in _AUDIT.items() if rx.search(text)]
    name = card.get("prod_name") or ""
    if _STANDALONE_BOTTOM.search(name) and not _SET_WORDS.search(name):
        hits.append("standalone_bottom")
    return hits


def run_grid(retriever: Any, df: Any) -> list[dict]:
    from src.agents.outfit.composer import build_more_options, compose_outfit

    rows = []
    for gender in GENDERS:
        for occasion in OCCASIONS:
            for budget in BUDGETS:
                look = compose_outfit(
                    df, retriever, occasion_slug=occasion, gender=gender, budget_inr=float(budget)
                )
                row = build_more_options([look], float(budget))
                pool = look.get("anchor_alternates")
                rows.append(
                    {
                        "gender": gender,
                        "occasion": occasion,
                        "budget": budget,
                        "seed": bool(look.get("seed_item")),
                        "pool": None if pool is None else len(pool),
                        "row": None if row is None else row,
                    }
                )
    return rows


def summarise(rows: list[dict]) -> dict:
    cats: Counter[str] = Counter()
    cards = 0
    empty_cells: list[tuple] = []
    absent_cells: list[tuple] = []
    leaks: list[tuple] = []
    for c in rows:
        key = (c["gender"], c["occasion"], c["budget"])
        if c["row"] is None:
            absent_cells.append(key)
            continue
        if not c["row"]:
            empty_cells.append((*key, c["pool"]))
        for card in c["row"]:
            cards += 1
            for cat in audit_card(card):
                cats[cat] += 1
                if cat != "standalone_bottom":
                    leaks.append((*key, cat, card.get("store"), card.get("product_type"), card.get("prod_name")))
    return {
        "cells": len(rows),
        "cards": cards,
        "audit_counts": dict(cats),
        "empty_row_cells": empty_cells,
        "absent_row_cells": absent_cells,
        "leaks": leaks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        sys.exit("INCLUDE_NON_PUBLIC_STORES is set; this grid is the public view only")
    from scripts import outfit_compose_snapshot as snap

    retriever, df = snap.load_retriever(args.index_dir)
    rows = run_grid(retriever, df)
    s = summarise(rows)
    print(f"cells={s['cells']} cards={s['cards']} audit_counts={s['audit_counts']}")
    print(f"empty-row cells (gender, occasion, budget, pool): {len(s['empty_row_cells'])}")
    for e in s["empty_row_cells"]:
        print("  ", e)
    print(f"absent-row cells: {len(s['absent_row_cells'])}")
    print(f"leaks (non-bottom audit hits): {len(s['leaks'])}")
    for leak in s["leaks"][:40]:
        print("  ", leak)
    if args.json:
        args.json.write_text(json.dumps({"summary": s, "rows": rows}, ensure_ascii=False, default=str), "utf-8")


if __name__ == "__main__":
    main()
