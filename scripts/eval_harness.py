#!/usr/bin/env python
"""Evaluation harness for the Agentic Shopping Assistant.

Usage:
    python scripts/eval_harness.py --dry-run           # validate YAML only
    python scripts/eval_harness.py                     # run all 32 queries live
    python scripts/eval_harness.py --query-id TB3      # run one query
"""
import argparse
import json
import os
import sys
import time
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ── path setup ───────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).parent
_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

_DATA_DIR = _ROOT / "data" / "processed"
_REPORTS_DIR = _ROOT / "reports"
_YAML_PATH = _SCRIPTS_DIR / "eval_queries.yaml"
_CONFIG_PATH = str(_ROOT / "config.yaml")

INTER_QUERY_DELAY = 2.0  # seconds between queries (Groq rate-limit buffer)

KNOWN_CHECK_KEYS = {
    "n_results_min",
    "colour_match",
    "colour_absent",
    "category_present",
    "category_present_or_empty_ack",
    "category_absent",
    "no_hallucination_keywords",
    "ooc_expected",
    "tool_expected",
    "filter_applied",
    "style_criteria",
}


# ── catalogue loading (fastparquet — pyarrow 19 has a histogram bug here) ────

def _load_catalogue_df(data_dir: Path):
    import fastparquet as fp
    pf = fp.ParquetFile(str(data_dir / "catalogue.parquet"))
    df = pf.to_pandas()
    # fastparquet expands nested-dict columns with dot notation; rebuild facets
    facet_keys = [
        "colour_group_name", "product_type_name", "department_name",
        "index_group_name", "garment_group_name",
    ]
    df["facets"] = [
        {k: df[f"facets.{k}"].iat[i] for k in facet_keys}
        for i in range(len(df))
    ]
    return df


# ── YAML loading ──────────────────────────────────────────────────────────────

def load_queries(yaml_path: Path) -> list[dict]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["queries"]


# ── dry-run validator ─────────────────────────────────────────────────────────

def dry_run(queries: list[dict], df) -> bool:
    """Validate YAML structure and catalogue value references. Returns True if clean."""
    all_colours = {c.lower() for c in df["colour_group_name"].dropna().unique()}
    all_types   = {t.lower() for t in df["product_type_name"].dropna().unique()}

    errors: list[str] = []

    print(f"{'ID':<6} {'Category':<16} Checks")
    print("-" * 80)

    for q in queries:
        qid      = q.get("id", "?")
        category = q.get("category", "?")
        checks   = q.get("checks", {})

        # Required fields
        for field in ("id", "category", "query", "checks"):
            if field not in q:
                errors.append(f"[{qid}] Missing required field: '{field}'")

        # Unknown check keys
        unknown = set(checks.keys()) - KNOWN_CHECK_KEYS
        if unknown:
            errors.append(f"[{qid}] Unknown check key(s): {unknown}")

        # Colour value validation
        for key in ("colour_match", "colour_absent"):
            for colour in checks.get(key, []):
                if colour.lower() not in all_colours:
                    errors.append(f"[{qid}] {key}: colour not in catalogue: {colour!r}")

        # Product-type value validation
        for key in ("category_present", "category_absent", "category_present_or_empty_ack"):
            for pt in checks.get(key, []):
                if pt and pt.lower() not in all_types:
                    errors.append(f"[{qid}] {key}: product_type not in catalogue: {pt!r}")

        # filter_applied: key must be a known facet key
        if "filter_applied" in checks:
            from src.agents.tools import VALID_FACET_KEYS
            for k in checks["filter_applied"]:
                if k not in VALID_FACET_KEYS:
                    errors.append(f"[{qid}] filter_applied: unknown facet key: {k!r}")

        # List active evaluators for this query
        active = sorted(k for k in KNOWN_CHECK_KEYS if k in checks)
        setup  = q.get("setup_turns", [])
        setup_note = f"  [{len(setup)}-turn setup]" if setup else ""
        print(f"{qid:<6} {category:<16} {', '.join(active)}{setup_note}")

    print()
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        return False

    print(f"CLEAN — {len(queries)} queries validated, 0 errors.")
    return True


# ── agent component loader ────────────────────────────────────────────────────

def build_components():
    from src.catalogue.loader import load_config
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.sparse_search import SparseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.llm.client import get_llm_client
    from src.memory.conversation import ConversationMemory
    from src.agents.graph import build_graph

    config = load_config(_CONFIG_PATH)
    config["llm"]["provider"] = os.environ.get("LLM_PROVIDER", "groq")

    print("Loading retrieval indices...")
    df = _load_catalogue_df(_DATA_DIR)
    dense  = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, df, config)

    llm    = get_llm_client(config)
    memory = ConversationMemory(llm, config)
    agent  = build_graph(retriever, df, llm, memory, config, streaming_mode=False)

    return agent, llm, config


# ── state helpers ─────────────────────────────────────────────────────────────

def _make_state(messages, user_query, retrieved_items=None, filters=None):
    return {
        "messages": list(messages),
        "user_query": user_query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": list(retrieved_items or []),
        "filters": dict(filters or {}),
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
    }


def _invoke(agent, state) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = agent.invoke(state)
    return result, time.perf_counter() - t0


# ── check evaluators ──────────────────────────────────────────────────────────

def evaluate_checks(checks: dict, result: dict, response_text: str) -> dict[str, Any]:
    """Evaluate all checks. Returns {check_name: True | False | "SKIP"}."""
    items      = result.get("retrieved_items", [])
    filters    = result.get("filters", {})
    tool_calls = result.get("tool_calls", [])
    ooc        = result.get("out_of_catalogue", False)

    ev: dict[str, Any] = {}

    # n_results_min
    if "n_results_min" in checks:
        ev["n_results_min"] = len(items) >= checks["n_results_min"]

    # colour_match: ≥50% of items must carry an acceptable colour
    if "colour_match" in checks:
        if not items:
            ev["colour_match"] = "SKIP"
        else:
            ok = {c.lower() for c in checks["colour_match"]}
            n_match = sum(1 for it in items if it.get("colour", "").lower() in ok)
            ev["colour_match"] = (n_match / len(items)) >= 0.5

    # colour_absent: no item may have any of these colours
    if "colour_absent" in checks:
        forbidden = {c.lower() for c in checks["colour_absent"]}
        ev["colour_absent"] = not any(
            it.get("colour", "").lower() in forbidden for it in items
        )

    # category_present: at least 1 item must have one of these product types
    if "category_present" in checks:
        ok = {pt.lower() for pt in checks["category_present"]}
        ev["category_present"] = any(
            it.get("product_type", "").lower() in ok for it in items
        )

    # category_present_or_empty_ack: item match OR response acknowledges via style_criteria
    if "category_present_or_empty_ack" in checks:
        ok = {pt.lower() for pt in checks["category_present_or_empty_ack"]}
        has_cat = any(it.get("product_type", "").lower() in ok for it in items)
        ack_words = [w.lower() for w in checks.get("style_criteria", [])]
        has_ack = bool(response_text) and any(w in response_text.lower() for w in ack_words)
        ev["category_present_or_empty_ack"] = has_cat or has_ack

    # category_absent: no item may have any of these product types (skip if list empty)
    if "category_absent" in checks:
        pts = checks["category_absent"]
        if not pts:
            ev["category_absent"] = "SKIP"
        else:
            forbidden = {pt.lower() for pt in pts}
            ev["category_absent"] = not any(
                it.get("product_type", "").lower() in forbidden for it in items
            )

    # no_hallucination_keywords: none must appear in response text
    if "no_hallucination_keywords" in checks:
        if not response_text:
            ev["no_hallucination_keywords"] = True
        else:
            text_lower = response_text.lower()
            violations = [kw for kw in checks["no_hallucination_keywords"]
                          if kw.lower() in text_lower]
            ev["no_hallucination_keywords"] = len(violations) == 0
            if violations:
                ev["_hallucination_violations"] = violations

    # ooc_expected: out_of_catalogue flag set AND zero items returned
    if "ooc_expected" in checks:
        ev["ooc_expected"] = bool(ooc) and len(items) == 0

    # tool_expected: the named tool must appear in tool_calls
    if "tool_expected" in checks:
        expected = checks["tool_expected"]
        ev["tool_expected"] = any(expected in tc for tc in tool_calls)

    # filter_applied: state["filters"] must contain all key-value pairs (case-insensitive)
    if "filter_applied" in checks:
        required = checks["filter_applied"]
        fil_lower = {k: str(v).lower() for k, v in filters.items()}
        ev["filter_applied"] = all(
            fil_lower.get(k, "") == str(v).lower()
            for k, v in required.items()
        )

    # style_criteria: any 1 term must appear in response text
    if "style_criteria" in checks:
        if not response_text:
            ev["style_criteria"] = False
        else:
            text_lower = response_text.lower()
            ev["style_criteria"] = any(w.lower() in text_lower for w in checks["style_criteria"])

    return ev


# ── single-query runner ───────────────────────────────────────────────────────

def run_query(agent, query_spec: dict) -> dict:
    setup_turns = query_spec.get("setup_turns", [])
    main_query  = query_spec["query"]
    checks      = query_spec.get("checks", {})

    messages        = []
    retrieved_items = []
    filters         = {}
    setup_latencies = []

    # Setup turns — build conversation context before the test query
    for turn_text in setup_turns:
        state = _make_state(
            messages + [{"role": "user", "content": turn_text}],
            turn_text,
            retrieved_items,
            filters,
        )
        res, lat = _invoke(agent, state)
        setup_latencies.append(round(lat, 2))
        messages        = res.get("messages", messages)
        retrieved_items = res.get("retrieved_items", retrieved_items)
        filters         = res.get("filters", filters)
        if setup_turns.index(turn_text) < len(setup_turns) - 1:
            time.sleep(0.5)  # brief gap between setup turns

    # Main test query
    state = _make_state(
        messages + [{"role": "user", "content": main_query}],
        main_query,
        retrieved_items,
        filters,
    )
    result, main_lat = _invoke(agent, state)

    response_text  = result.get("final_answer", "") or ""
    check_results  = evaluate_checks(checks, result, response_text)

    passed  = [k for k, v in check_results.items() if v is True  and not k.startswith("_")]
    failed  = [k for k, v in check_results.items() if v is False and not k.startswith("_")]
    skipped = [k for k, v in check_results.items() if v == "SKIP"]
    overall = "FAIL" if failed else "PASS"

    return {
        "id":             query_spec["id"],
        "category":       query_spec.get("category", ""),
        "query":          main_query,
        "setup_turns":    setup_turns,
        "status":         overall,
        "checks":         check_results,
        "passed":         passed,
        "failed":         failed,
        "skipped":        skipped,
        "n_items":        len(result.get("retrieved_items", [])),
        "response_text":  response_text[:600],
        "tool_calls":     [list(tc.keys())[0] for tc in result.get("tool_calls", [])],
        "filters":        result.get("filters", {}),
        "out_of_catalogue": bool(result.get("out_of_catalogue")),
        "latency_main":   round(main_lat, 2),
        "latency_setup":  setup_latencies,
        "latency_total":  round(main_lat + sum(setup_latencies), 2),
    }


# ── output path helper ────────────────────────────────────────────────────────

def _versioned_path(directory: Path, stem: str, suffix: str) -> Path:
    for v in range(1, 100):
        p = directory / f"{stem}_v{v}{suffix}"
        if not p.exists():
            return p
    raise RuntimeError("Too many existing result files")


# ── summary printer ───────────────────────────────────────────────────────────

def print_summary(results: list[dict], total_time: float):
    n      = len(results)
    pass_n = sum(1 for r in results if r["status"] == "PASS")
    fail_n = sum(1 for r in results if r["status"] == "FAIL")
    err_n  = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\n{'='*60}")
    print(f"RESULTS: {pass_n}/{n} PASS  |  {fail_n} FAIL  |  {err_n} ERROR  "
          f"|  {pass_n/n*100:.0f}%")
    print(f"Total elapsed: {total_time:.1f}s")

    # By category
    cats = defaultdict(list)
    for r in results:
        cats[r.get("category", "?")].append(r)
    print("\nBy category:")
    for cat in sorted(cats):
        recs = cats[cat]
        p = sum(1 for r in recs if r["status"] == "PASS")
        print(f"  {cat:<18}  {p}/{len(recs)}")

    # Latency stats (exclude ERROR rows)
    lats = sorted(r["latency_total"] for r in results if r.get("latency_total", 0) > 0)
    if lats:
        p95 = lats[max(0, int(len(lats) * 0.95) - 1)]
        print(f"\nLatency (total incl. setup): "
              f"median={statistics.median(lats):.1f}s  "
              f"min={lats[0]:.1f}s  max={lats[-1]:.1f}s  p95={p95:.1f}s")

    # Failed queries
    failed_recs = [r for r in results if r["status"] == "FAIL"]
    if failed_recs:
        print(f"\nFailed ({len(failed_recs)}):")
        for r in failed_recs:
            print(f"  ✗ {r['id']:<5}  failed: {r['failed']}")

    # Error queries
    err_recs = [r for r in results if r["status"] == "ERROR"]
    if err_recs:
        print(f"\nErrors ({len(err_recs)}):")
        for r in err_recs:
            print(f"  ✗ {r['id']:<5}  {r.get('error', '?')[:100]}")

    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load .env from repo root before any Groq client is initialised.
    # Shell-exported GROQ_API_KEY takes precedence (dotenv does not override).
    load_dotenv(_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Eval harness for the Shopping Assistant")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate YAML only — no agent or LLM calls")
    parser.add_argument("--query-id", metavar="ID",
                        help="Run a single query by id (e.g. C1, TB3)")
    parser.add_argument("--yaml", default=str(_YAML_PATH),
                        help="Path to eval_queries.yaml")
    args = parser.parse_args()

    # API key is only required for live runs
    if not args.dry_run and not os.environ.get("GROQ_API_KEY"):
        sys.exit(
            "GROQ_API_KEY missing — create a .env file with GROQ_API_KEY=your_key "
            "or export it in your shell.\n"
            "See .env.example for the expected format."
        )

    yaml_path = Path(args.yaml)
    queries   = load_queries(yaml_path)

    if args.query_id:
        queries = [q for q in queries if q["id"] == args.query_id]
        if not queries:
            print(f"No query with id {args.query_id!r}")
            sys.exit(1)

    # ── DRY RUN ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"DRY RUN — {yaml_path.name}  ({len(queries)} quer{'y' if len(queries) == 1 else 'ies'})\n")
        import fastparquet as fp
        pf     = fp.ParquetFile(str(_DATA_DIR / "catalogue.parquet"))
        df_chk = pf.to_pandas()
        ok     = dry_run(queries, df_chk)
        sys.exit(0 if ok else 1)

    # ── LIVE RUN ──────────────────────────────────────────────────────────────
    print("Building agent components...")
    agent, llm, config = build_components()
    print(f"Ready — {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} queued.\n")

    _REPORTS_DIR.mkdir(exist_ok=True)
    stem      = f"eval_results_{date.today().strftime('%Y%m%d')}"
    json_path = _versioned_path(_REPORTS_DIR, stem, ".json")
    md_path   = json_path.with_suffix(".md")

    results = []
    t_start = time.perf_counter()

    for i, q in enumerate(queries, 1):
        qid      = q["id"]
        category = q.get("category", "")
        setup_n  = len(q.get("setup_turns", []))
        setup_tag = f" [+{setup_n} setup]" if setup_n else ""
        print(f"[{i:2d}/{len(queries)}] {qid} ({category}){setup_tag}  {q['query'][:60]}")

        try:
            rec = run_query(agent, q)
            icon = "✓" if rec["status"] == "PASS" else "✗"
            fail_str = f"  FAILED: {rec['failed']}" if rec["failed"] else ""
            print(f"          {icon} {rec['status']}  {rec['latency_total']:.1f}s  "
                  f"items={rec['n_items']}{fail_str}")
        except Exception as exc:
            print(f"          ✗ ERROR: {exc!r}")
            rec = {
                "id": qid, "category": category, "query": q["query"],
                "setup_turns": q.get("setup_turns", []),
                "status": "ERROR", "error": repr(exc),
                "checks": {}, "passed": [], "failed": [], "skipped": [],
                "n_items": 0, "response_text": "", "tool_calls": [],
                "filters": {}, "out_of_catalogue": False,
                "latency_main": 0.0, "latency_setup": [], "latency_total": 0.0,
            }

        results.append(rec)

        if i < len(queries):
            time.sleep(INTER_QUERY_DELAY)

    total_time = time.perf_counter() - t_start

    # Save JSON
    payload = {
        "run_date":      date.today().isoformat(),
        "n_queries":     len(results),
        "total_time_s":  round(total_time, 1),
        "results":       results,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nJSON saved → {json_path}")

    # Summary to console
    print_summary(results, total_time)

    # Generate markdown report
    from eval_report import generate_markdown
    md = generate_markdown(payload)
    md_path.write_text(md, encoding="utf-8")
    print(f"MD   saved → {md_path}")


if __name__ == "__main__":
    main()
