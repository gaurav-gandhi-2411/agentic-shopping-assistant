#!/usr/bin/env python
"""Would promoting fabric_score_delta's existing HALDI_LIGHTWEIGHT_KEYWORDS
vocabulary (src/agents/outfit/slots.py) into a HARD gate for the office
occasion safely catch the office-register misses this catalogue has
(athleisure items with formal-sounding names, casual co-ord sets)?

This is the third data point for eval/README.md's "text-formality signals
derived from marketing copy" standing rule, alongside the two already
measured and rejected in reports/compose_wave_final_20260807.md:
  - "embellishment vocabulary" (Cluster A's embellishment-vs-plain-pattern
    register signal): 60% FP (12/20).
  - "desc-marker" (the office-register broad-desc-scan precedent): 25% FP
    (2/8).

Method: gather every item ALREADY hand-labeled (relevant true or false) for
every office-context query in eval/fixtures/strict_gold_labels.yaml, check
each against the real catalogue row's prod_name + detail_desc for any
HALDI_LIGHTWEIGHT_KEYWORDS hit, and measure what fraction of the matched
items are currently-relevant (would be a false positive if hard-rejected)
vs currently-irrelevant (would be a correct reject).

Usage:
    python -m eval.diagnose_fabric_score_delta_promotion
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.outfit.slots import HALDI_LIGHTWEIGHT_KEYWORDS  # noqa: E402

_DATA_DIR = _ROOT / "data" / "processed" / "unified"
_QUERIES_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml"
_LABELS_PATH = _ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml"


def main() -> None:
    queries = yaml.safe_load(_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    office_qids = {q["id"] for q in queries if "office" in q["query"].lower()}
    print(f"office-context query ids ({len(office_qids)}): {sorted(office_qids)}")

    labels_raw = yaml.safe_load(_LABELS_PATH.read_text(encoding="utf-8"))["labels"]
    office_items: list[tuple[str, str, bool]] = [
        (entry["query_id"], str(item["article_id"]), bool(item["relevant"]))
        for entry in labels_raw
        if entry["query_id"] in office_qids
        for item in entry["items"]
    ]
    print(f"total office-context labeled (query, item) slots: {len(office_items)}")

    df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    df["article_id"] = df["article_id"].astype(str)
    cat = df.set_index("article_id")[["prod_name", "detail_desc"]]

    raw_true = raw_false = 0
    by_aid: dict[str, set[bool]] = {}
    matches: list[tuple[str, str, str, bool, list[str]]] = []
    for qid, aid, rel in office_items:
        if aid not in cat.index:
            continue
        row = cat.loc[aid]
        text = (str(row["prod_name"]) + " " + str(row["detail_desc"])).lower()
        matched = [kw for kw in HALDI_LIGHTWEIGHT_KEYWORDS if kw in text]
        if not matched:
            continue
        matches.append((qid, aid, str(row["prod_name"]), rel, matched))
        if rel:
            raw_true += 1
        else:
            raw_false += 1
        by_aid.setdefault(aid, set()).add(rel)

    print()
    print("=== per query-item-slot (raw) ===")
    print(f"would-be-wrongly-rejected (currently relevant=true): {raw_true}")
    print(f"would-be-correctly-rejected (currently relevant=false): {raw_false}")
    print(f"raw FP rate = {raw_true}/{raw_true + raw_false} = {raw_true / (raw_true + raw_false):.1%}")

    dedup_true = sum(1 for v in by_aid.values() if True in v)
    print()
    print("=== deduplicated by distinct article_id ===")
    print(f"distinct matched article_ids: {len(by_aid)}")
    print(f"at least one relevant=true occurrence (would be wrongly rejected): {dedup_true}")
    print(f"deduped FP rate = {dedup_true}/{len(by_aid)} = {dedup_true / len(by_aid):.1%}")

    print()
    print("=== sample false positives (currently relevant=true, would be hard-rejected) ===")
    for qid, aid, name, rel, matched in matches:
        if rel:
            print(f"  [{qid}] {name!r} matched={matched}")


if __name__ == "__main__":
    main()
