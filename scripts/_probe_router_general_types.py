#!/usr/bin/env python
"""One-off: with the fixed (catalogue-derived) router vocabulary live, measure
the RESIDUAL risk of the LLM emitting a product_type_name that isn't an exact
catalogue value at all (ignoring the given list, drawing on pretrained bias)
— the remaining risk after vocabulary staleness itself is closed. Supports
the type-pushdown safety re-audit. Not a permanent script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from src.agents.graph import (  # noqa: E402
    ROUTER_PROMPT,
    _build_facet_vocabulary_text,
    _format_items_brief,
    _format_messages,
    _parse_router_response,
)
from src.catalogue.loader import load_config  # noqa: E402
from src.llm.client import get_llm_client  # noqa: E402

_catalogue_df = pd.read_parquet(_ROOT / "data" / "processed" / "unified" / "catalogue.parquet")
_VALID_TYPES = set(_catalogue_df["product_type_name"].dropna().str.lower().unique())
ROUTER_PROMPT = ROUTER_PROMPT.replace("{facet_vocabulary}", _build_facet_vocabulary_text(_catalogue_df))

_QUERIES = [
    "red dress under 3000",
    "kurta for men",
    "black saree for a wedding",
    "formal blazer for men",
    "yellow anarkali for haldi",
    "denim jeans for women",
    "office shirt for men",
    "party gown for women",
    "sherwani for groom",
    "casual shorts for men",
    "ethnic palazzo pants for women",
    "gold earrings for women",
    "sneakers for women",
    "lehenga for sangeet",
    "cotton kurti under 1500",
]


def main() -> None:
    config = load_config()
    config["llm"]["provider"] = "groq"
    client = get_llm_client(config)

    exact = wrong = none_type = 0
    results = []
    for q in _QUERIES:
        prompt = ROUTER_PROMPT.format(
            last_action="none", items_retrieved=0, retrieved_summary=_format_items_brief([]),
            current_filters="{}", user_query=q,
            conversation=_format_messages([{"role": "user", "content": q}]),
        )
        raw = client.chat([{"role": "user", "content": prompt}])
        parsed = _parse_router_response(raw, q)
        ptype = (parsed.get("filters") or {}).get("product_type_name")
        if ptype is None:
            none_type += 1
            verdict = "no type filter set"
        elif ptype.lower() in _VALID_TYPES:
            exact += 1
            verdict = "EXACT catalogue match"
        else:
            wrong += 1
            verdict = "WRONG (not in catalogue at all)"
        results.append({"query": q, "product_type_name": ptype, "verdict": verdict})
        print(f"{q:<40} product_type_name={ptype!r:<20} {verdict}")

    print(f"\n{exact}/{len(_QUERIES)} exact catalogue match")
    print(f"{none_type}/{len(_QUERIES)} no type filter set (safe)")
    print(f"{wrong}/{len(_QUERIES)} WRONG value emitted (would break a hard pre-filter)")

    out_path = _ROOT / "reports" / "router_general_type_probe.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
