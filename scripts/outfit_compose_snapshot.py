"""Record / replay the outfit compose path (probe + variants) against the local unified index.

Mirrors the outfit node's composition in src/agents/graph.py (deterministic intent parse ->
probe compose_outfit_tool -> compose_outfit_variants) with NO LLM: rationales are not needed for
item selection, so none are generated.  Used to prove a latency change leaves selections
byte-identical (evals/fixtures/outfit_compose_snapshot.json) and to time before/after.

Usage:
    python scripts/outfit_compose_snapshot.py --index-dir <dir> --record  # write fixture
    python scripts/outfit_compose_snapshot.py --index-dir <dir> --check   # compare to fixture
    python scripts/outfit_compose_snapshot.py --index-dir <dir> --timing out.json [--runs 3]

The index is the PUBLIC view (INCLUDE_NON_PUBLIC_STORES must be unset). No .env is loaded.
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_PATH = REPO_ROOT / "evals" / "fixtures" / "outfit_compose_snapshot.json"

# 30 queries: occasions x genders x with/without budgets. The first two are the Sangeet
# strings from the latency diagnosis (the second is an assumed sibling, see PR notes).
QUERIES: list[str] = [
    "sangeet outfit under ₹3000 for women",
    "Sangeet outfit under ₹5000 for men",
    "sangeet outfit for women",
    "sangeet look for men",
    "sangeet outfit under ₹8000 for women",
    "haldi outfit for women",
    "haldi outfit under ₹2000 for women",
    "haldi outfit for men",
    "haldi look under ₹3000 for men",
    "mehendi outfit for women",
    "mehendi outfit under ₹4000 for women",
    "mehendi look for men",
    "wedding guest outfit for women",
    "wedding guest outfit under ₹6000 for women",
    "wedding guest outfit for men",
    "wedding guest outfit under ₹10000 for men",
    "office outfit for women",
    "office outfit under ₹3000 for women",
    "office look for men",
    "office outfit under ₹5000 for men",
    "cocktail outfit for women",
    "cocktail outfit under ₹7000 for women",
    "cocktail look for men",
    "cocktail party outfit under ₹6000 for men",
    "reception outfit for women",
    "reception outfit under ₹15000 for women",
    "eid outfit for men",
    "diwali outfit under ₹4000 for women",
    "date night outfit for women",
    "casual outfit under ₹2500 for men",
]


def load_retriever(index_dir: Path) -> tuple[Any, Any]:
    """Load the public-view retriever + catalogue exactly like the API startup path."""
    from src.catalogue.loader import load_config
    from src.retrieval.snapshot import load_index_parts

    config = load_config(str(REPO_ROOT / "config.yaml"))
    return load_index_parts(index_dir, config)


def compose_turn(query: str, df: Any, retriever: Any) -> dict:
    """Run the outfit node's composition for ``query`` (no LLM). Returns the variants."""
    from src.agents.intent_parser import parse_intent
    from src.agents.outfit.composer import compose_outfit_variants
    from src.agents.tools import compose_outfit_tool

    intent = parse_intent(query)
    occasion = intent.occasion or "casual"
    gender = intent.gender or "women"
    budget = intent.budget_max_inr
    kwargs: dict[str, Any] = dict(
        occasion_slug=occasion,
        gender=gender,
        budget_inr=budget,
        body_type=intent.body_type,
        body_modifiers=intent.body_modifiers or [],
        formality_override=intent.formality_softener,
    )
    probe = compose_outfit_tool(
        seed_article_id=None, catalogue_df=df, retriever=retriever, **kwargs
    )
    if probe.get("seed_item") is None:
        return {"occasion": occasion, "gender": gender, "budget": budget, "variants": []}
    vkwargs = dict(kwargs, brand_gender_default="women", pairing_stats=None)
    # "base_look" exists only after the probe-reuse change; same call site before and after.
    params = inspect.signature(compose_outfit_variants).parameters
    if "base_look" in params:
        vkwargs["base_look"] = probe
    variants = compose_outfit_variants(df, retriever, seed_article_id=None, **vkwargs)
    return {"occasion": occasion, "gender": gender, "budget": budget, "variants": variants}


def summarise(turn: dict) -> dict:
    """Selection-only view: ordered item IDs per variant + totals (look_id is a fresh uuid)."""
    out = []
    for v in turn["variants"]:
        seed = v.get("seed_item") or {}
        out.append(
            {
                "label": v.get("variant_label"),
                "item_ids": [str(seed.get("article_id"))]
                + [str(c["article_id"]) for c in (v.get("complements") or [])],
                "budget_total_inr": v.get("budget_total_inr"),
                "empty_slots": list(v.get("empty_slots") or []),
            }
        )
    return {
        "occasion": turn["occasion"],
        "gender": turn["gender"],
        "budget_inr": turn["budget"],
        "variants": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare against the fixture")
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import os

    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        sys.exit("INCLUDE_NON_PUBLIC_STORES is set; this snapshot is the public view only")

    retriever, df = load_retriever(args.index_dir)
    # Warm-up: loads models/lazy state so the first timed query is not an outlier.
    compose_turn("casual outfit for women", df, retriever)

    if args.record:
        snap = {q: summarise(compose_turn(q, df, retriever)) for q in QUERIES}
        # One query per line keeps the fixture diff-reviewable (30 lines, not ~1.2k).
        body = ",\n".join(
            f"{json.dumps(q, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
            for q, v in snap.items()
        )
        FIXTURE_PATH.write_text("{\n" + body + "\n}\n", "utf-8")
        print(f"wrote {FIXTURE_PATH} ({len(snap)} queries)")

    if args.check:
        want = json.loads(FIXTURE_PATH.read_text("utf-8"))
        bad = [q for q in QUERIES if summarise(compose_turn(q, df, retriever)) != want[q]]
        print(f"fixture check: {len(QUERIES) - len(bad)}/{len(QUERIES)} identical")
        for q in bad:
            print("MISMATCH:", q)
        if bad:
            sys.exit(1)

    if args.timing:
        try:
            import psutil

            _proc = psutil.Process()
            rss_before_mb = _proc.memory_info().rss / 2**20
        except ImportError:
            _proc, rss_before_mb = None, None
        timings: dict[str, list[float]] = {}
        for q in QUERIES:
            runs = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                compose_turn(q, df, retriever)
                runs.append((time.perf_counter() - t0) * 1000.0)
            timings[q] = runs
        meds = [statistics.median(v) for v in timings.values()]
        out = {
            "runs": timings,
            "median_ms": {q: statistics.median(v) for q, v in timings.items()},
            "overall_median_of_medians_ms": statistics.median(meds),
            "overall_mean_of_medians_ms": statistics.fmean(meds),
            "rss_mb_before_timing": rss_before_mb,
            "rss_mb_after_timing": _proc.memory_info().rss / 2**20 if _proc else None,
        }
        args.timing.write_text(json.dumps(out, indent=1, ensure_ascii=False), "utf-8")
        print(f"overall median-of-medians {out['overall_median_of_medians_ms']:.0f} ms")


if __name__ == "__main__":
    main()
