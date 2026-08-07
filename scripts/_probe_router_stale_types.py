#!/usr/bin/env python
"""One-off: probe the LIVE LLM router with realistic queries covering the 14
stale product_type_name example words in ROUTER_PROMPT, to measure how often
it actually emits one of them (vs. omitting the filter or using a real value).
Supports the router-vocabulary staleness wave. Not a permanent script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

import pandas as pd  # noqa: E402

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
ROUTER_PROMPT = ROUTER_PROMPT.replace("{facet_vocabulary}", _build_facet_vocabulary_text(_catalogue_df))

_STALE_WORDS = [
    "Coat", "Jacket", "Sweater", "Cardigan", "T-shirt", "Vest top",
    "Leggings/Tights", "Swimwear bottom", "Bikini top", "Swimsuit",
    "Pyjama set", "Night gown", "Hoodie", "Robe",
]

_QUERIES = [
    "jacket for men",
    "warm winter coat for women",
    "sweater for women under 2000",
    "cardigan for women",
    "plain t-shirt for men",
    "vest top for women",
    "leggings for women",
    "swimsuit for women",
    "bikini for women",
    "pyjama set for men",
    "night gown for women",
    "hoodie for men under 1500",
    "bathrobe for women",
    "black hoodie for women",
    "denim jacket for women",
]


def main() -> None:
    config = load_config()
    config["llm"]["provider"] = "groq"
    client = get_llm_client(config)

    stale_hits = 0
    no_type = 0
    real_type = 0
    results = []
    for q in _QUERIES:
        prompt = ROUTER_PROMPT.format(
            last_action="none",
            items_retrieved=0,
            retrieved_summary=_format_items_brief([]),
            current_filters="{}",
            user_query=q,
            conversation=_format_messages([{"role": "user", "content": q}]),
        )
        raw = client.chat([{"role": "user", "content": prompt}])
        parsed = _parse_router_response(raw, q)
        filters = parsed.get("filters") or {}
        ptype = filters.get("product_type_name")
        if ptype is None:
            no_type += 1
            verdict = "no type filter set"
        elif ptype in _STALE_WORDS:
            stale_hits += 1
            verdict = "STALE (dead value)"
        else:
            real_type += 1
            verdict = "other value"
        results.append({"query": q, "action": parsed.get("action"), "product_type_name": ptype, "verdict": verdict})
        print(f"{q:<35} action={parsed.get('action'):<10} product_type_name={ptype!r:<20} {verdict}")

    print(f"\n{stale_hits}/{len(_QUERIES)} queries got a STALE product_type_name value emitted by the LLM router")
    print(f"{no_type}/{len(_QUERIES)} got no type filter at all")
    print(f"{real_type}/{len(_QUERIES)} got some other (non-stale-list) value")

    out_path = _ROOT / "reports" / "router_stale_type_probe.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
