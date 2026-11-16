#!/usr/bin/env python
"""Report what build-time catalogue hygiene (src/catalogue/hygiene.py) does to a unified catalogue.

Read-only: applies :func:`apply_hygiene` in memory and prints markdown tables:
  - per-store rows, category mix, unknown count, dropped (non-fashion) and quarantined counts
  - 20 random examples (seed 42) of dropped rows and of quarantined rows
Nothing is written unless --out is given.

Usage:
    python scripts/catalogue_hygiene_report.py --catalogue data/processed/unified/catalogue.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.catalogue.hygiene import CATEGORIES, apply_hygiene  # noqa: E402

N_EXAMPLES = 20
SEED = 42


def _md_table(df: pd.DataFrame) -> str:
    """GitHub markdown table of ``df`` (index first); avoids the optional tabulate dependency."""
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, r in df.iterrows():
        rows.append("| " + " | ".join([str(idx)] + [str(v) for v in r.tolist()]) + " |")
    return "\n".join(rows)


def build_report(cat: pd.DataFrame) -> str:
    """Markdown report for catalogue frame ``cat`` (needs store, prod_name, product_type_name)."""
    kept, rep = apply_hygiene(cat)
    dropped, quar = rep["dropped"], rep["quarantined"]
    lines = [f"rows in {len(cat)}, kept {len(kept)}, dropped (non_fashion) {len(dropped)}, "
             f"quarantined {len(quar)}", ""]

    # category mix per store is computed on the input rows (kept + dropped + quarantined)
    mix = pd.concat([kept[["store", "category"]], dropped[["store", "category"]],
                     quar[["store", "category"]]])
    tab = mix.groupby(["store", "category"]).size().unstack(fill_value=0)
    for c in CATEGORIES:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[list(CATEGORIES)]
    tab["rows_in"] = cat.groupby("store").size()
    tab["dropped"] = dropped.groupby("store").size()
    tab["quarantined"] = quar.groupby("store").size()
    tab = tab.fillna(0).astype(int).sort_values("rows_in", ascending=False)
    lines += ["## Per store", "", _md_table(tab), ""]

    for title, frame in (("Dropped (non_fashion)", dropped), ("Quarantined", quar)):
        lines += [f"## {title}: per store", "",
                  _md_table(frame.groupby("store").size().sort_values(ascending=False).to_frame("n")),
                  ""]
        sample = frame.sample(n=min(N_EXAMPLES, len(frame)), random_state=SEED)
        lines += [f"### {title}: {len(sample)} random examples (seed {SEED})", "",
                  "| store | product_type | title | reason |", "|---|---|---|---|"]
        for _, r in sample.iterrows():
            title_txt = str(r["prod_name"]).replace("|", "/")[:90]
            lines.append(f"| {r['store']} | {r['product_type_name']} | {title_txt} | {r['reason']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogue", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = build_report(pd.read_parquet(args.catalogue))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
