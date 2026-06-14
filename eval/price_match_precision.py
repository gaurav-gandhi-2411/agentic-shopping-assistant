"""Precision eval for the cross-store same-product price matcher.

Verifies two critical properties:

  1. PRECISION >= 0.95 — the matcher must not false-fire on confusable
     brand-collision pairs (trust-critical: a false match destroys credibility).
  2. RECALL > 0 on synthetic true positives — the matcher can actually fire
     when a genuine cross-listing exists (sanity check against trivial reject-all).

Also runs a bounded scan over the real unified catalogue and reports how many
cross-store matches fire (expected: ~0 given the data reality documented in spec
and in entity_resolution.py).

Exit codes: 0 = PASS (both gates satisfied), 1 = FAIL.

Runnable as:
    python -m eval.price_match_precision
    python eval/price_match_precision.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Ensure repo root is on sys.path regardless of invocation style.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.catalogue.entity_resolution import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    TITLE_SIMILARITY_THRESHOLD,
    build_brand_index,
    build_brand_stores_map,
    find_cross_store_matches,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
_FIXTURE_PATH = _EVAL_DIR / "fixtures" / "price_match_precision.yaml"
_UNIFIED_DIR = _REPO_ROOT / "data" / "processed" / "unified"

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

# Trust-critical: a single false match destroys user trust.
# Failing to match (false negative) is fine — degrade to "available at {store}".
PRECISION_GATE: float = 0.95

# Sanity check: the matcher must be capable of firing (not a trivial reject-all).
# Even recall = 1/3 (one true positive out of three) passes this gate.
RECALL_GATE: float = 0.01  # any recall > 0

# Real-catalogue sample size for the live-data scan.
# Using 2000 items to bound runtime; brand-bucket guard makes the scan fast.
_REAL_CATALOGUE_SAMPLE: int = 2000
_RNG_SEED: int = 42

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PairDecision:
    """Result of running the matcher on one fixture pair."""

    pair_id: str
    label: str
    expected_match: bool
    actual_match: bool
    confidence: float | None  # None if no match found
    is_correct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_correct = self.expected_match == self.actual_match


@dataclass
class EvalReport:
    """Full precision eval report."""

    decisions: list[PairDecision]
    precision: float
    recall: float
    n_tp: int
    n_fp: int
    n_fn: int
    n_tn: int
    precision_pass: bool
    recall_pass: bool
    overall_pass: bool
    real_catalogue_matches: int
    real_catalogue_sample_size: int


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def _load_fixtures(path: Path) -> dict[str, Any]:
    """Load hard_negatives and true_positives fixture lists from YAML."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data


# ---------------------------------------------------------------------------
# Core: run matcher against one pair
# ---------------------------------------------------------------------------


def _run_pair(
    pair: dict[str, Any],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> PairDecision:
    """Run the matcher with query_item against a single-row catalogue of candidate_item.

    The candidate is wrapped in a minimal one-row DataFrame.  This is
    deterministic and needs no external data files.
    """
    query = pair["query_item"]
    candidate = pair["candidate_item"]
    expected = bool(pair["expected_match"])

    # Build a one-row catalogue from the candidate for the matcher to scan.
    cand_df = pd.DataFrame([candidate])

    # Ensure required columns exist (some fixture entries use synthetic data
    # without all catalogue columns — fill missing with None).
    for col in ("article_id", "prod_name", "product_type_name", "gender", "store",
                "price_inr", "pdp_handle"):
        if col not in cand_df.columns:
            cand_df[col] = None

    # Build a brand-stores map for this tiny one-row catalogue.
    # For the single-store brand-bucket guard to NOT prune incorrectly, we need
    # the query's brand to appear in the candidate's store.
    # The tiny-catalogue approach is correct: if the real guard would skip the
    # candidate (because the brand is in only one store globally), the guard
    # also fires on the fake catalogue — which is what we want to test.
    # For true positives, we override with a 2-store brand_stores_map so the
    # brand-bucket guard doesn't falsely prune the synthetic brand.
    query_store = str(query.get("store", "") or "")
    cand_store = str(candidate.get("store", "") or "")
    query_prod_name = str(query.get("prod_name", "") or "")

    from src.catalogue.entity_resolution import extract_brand

    query_brand = extract_brand(query_prod_name, query_store)

    # Build a permissive brand_stores_map that includes both stores for the query
    # brand (needed for synthetic TP pairs whose brand doesn't exist in the real data).
    permissive_brand_map: dict[str, set[str]] = {
        query_brand: {query_store, cand_store},
    }

    matches = find_cross_store_matches(
        query,
        cand_df,
        min_confidence=min_confidence,
        brand_stores_map=permissive_brand_map,
    )
    matched = len(matches) > 0
    confidence = matches[0]["confidence"] if matches else None

    return PairDecision(
        pair_id=str(pair.get("id", "")),
        label=str(pair.get("label", "")),
        expected_match=expected,
        actual_match=matched,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Real-catalogue scan
# ---------------------------------------------------------------------------


def _run_real_catalogue_scan(
    catalogue_df: pd.DataFrame,
    *,
    sample_size: int = _REAL_CATALOGUE_SAMPLE,
    seed: int = _RNG_SEED,
) -> tuple[int, int]:
    """Run the matcher over a bounded sample of the real catalogue.

    Returns (n_matches_found, sample_size_used).

    For each sampled item, find_cross_store_matches is run against the FULL
    catalogue (not just the sample) so that cross-store candidates can be found.
    The brand-bucket guard makes this fast: for ~99.9% of items the guard exits
    immediately (single-store brand), so the O(n) scan body rarely runs.
    """
    import numpy as np  # noqa: PLC0415
    rng_state = np.random.default_rng(seed)
    n = min(sample_size, len(catalogue_df))
    sampled_idx = rng_state.choice(len(catalogue_df), size=n, replace=False)
    sample_rows = catalogue_df.iloc[sampled_idx]

    brand_map = build_brand_stores_map(catalogue_df)
    b_index = build_brand_index(catalogue_df)
    n_matches = 0

    for _, row in sample_rows.iterrows():
        row_dict = row.to_dict()
        matches = find_cross_store_matches(
            row_dict,
            catalogue_df,
            brand_stores_map=brand_map,
            brand_index=b_index,
        )
        if matches:
            n_matches += 1

    return n_matches, n


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_LINE = "-" * 90


def _print_decisions(decisions: list[PairDecision]) -> None:
    """Print per-pair decision table."""
    print()
    print(f"{'ID':<12} {'Expected':<10} {'Actual':<10} {'Confidence':>11} {'OK?':<5} Label")
    print(_LINE)
    for d in decisions:
        conf_str = f"{d.confidence:.4f}" if d.confidence is not None else "  N/A  "
        ok_str = "PASS" if d.is_correct else "FAIL"
        label_trunc = d.label[:52] if len(d.label) > 52 else d.label
        print(
            f"{d.pair_id:<12} {'match' if d.expected_match else 'no-match':<10} "
            f"{'match' if d.actual_match else 'no-match':<10} {conf_str:>11} "
            f"{ok_str:<5} {label_trunc}"
        )
    print(_LINE)


def _print_report(report: EvalReport) -> None:
    """Print the full eval report to stdout."""
    print()
    print("Price-match precision eval  (Phase D)")
    print(_LINE)
    print(f"  Fixture pairs:  {len(report.decisions)} total")
    print(f"    Hard negatives (must reject):  {report.n_tn + report.n_fp}")
    print(f"    True positives (must match):   {report.n_tp + report.n_fn}")
    print()

    _print_decisions(report.decisions)

    print()
    print("Confusion matrix:")
    print(f"  True  Positives (TP): {report.n_tp}")
    print(f"  False Positives (FP): {report.n_fp}  <- false matches (trust-critical)")
    print(f"  True  Negatives (TN): {report.n_tn}")
    print(f"  False Negatives (FN): {report.n_fn}")
    print()

    prec_gate_str = f">= {PRECISION_GATE:.2f}"
    rec_gate_str = f"> {RECALL_GATE:.4f}"
    prec_result = "PASS" if report.precision_pass else "FAIL"
    rec_result = "PASS" if report.recall_pass else "FAIL"

    print("Gate results:")
    print(
        f"  Precision: {report.precision:.4f}  (gate {prec_gate_str})  [{prec_result}]"
        f"  {'<-- trust-critical gate' if not report.precision_pass else ''}"
    )
    print(
        f"  Recall:    {report.recall:.4f}  (gate {rec_gate_str})  [{rec_result}]"
        f"  {'<-- sanity: matcher must be capable of firing' if not report.recall_pass else ''}"
    )
    print()

    print("Real-catalogue scan:")
    print(
        f"  Scanned {report.real_catalogue_sample_size} items from the live unified catalogue."
    )
    print(
        f"  Cross-store matches found: {report.real_catalogue_matches}  "
        f"(expected ~0 - this catalogue has no genuine cross-store same-product overlap)"
    )
    print()
    print(_LINE)
    print(f"  OVERALL: {'PASS' if report.overall_pass else 'FAIL'}")
    print(_LINE)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_precision_recall(
    decisions: list[PairDecision],
) -> tuple[float, float, int, int, int, int]:
    """Compute precision, recall, TP, FP, TN, FN from pair decisions."""
    tp = sum(1 for d in decisions if d.expected_match and d.actual_match)
    fp = sum(1 for d in decisions if not d.expected_match and d.actual_match)
    tn = sum(1 for d in decisions if not d.expected_match and not d.actual_match)
    fn = sum(1 for d in decisions if d.expected_match and not d.actual_match)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # no matches at all = perfect precision
    n_positives = tp + fn
    recall = tp / n_positives if n_positives > 0 else 0.0
    return precision, recall, tp, fp, tn, fn


def main() -> None:
    """Run the price-match precision eval.  Exit 0 on PASS, 1 on FAIL."""
    print()
    print("Loading fixture:", _FIXTURE_PATH)
    fixtures = _load_fixtures(_FIXTURE_PATH)

    hard_negatives: list[dict[str, Any]] = fixtures.get("hard_negatives", [])
    true_positives: list[dict[str, Any]] = fixtures.get("true_positives", [])
    all_pairs = hard_negatives + true_positives

    print(
        f"  {len(hard_negatives)} hard negatives, {len(true_positives)} true positives "
        f"({len(all_pairs)} total pairs)"
    )
    print(
        f"  Matching threshold: TITLE_SIMILARITY >= {TITLE_SIMILARITY_THRESHOLD:.2f}, "
        f"min_confidence = {DEFAULT_MIN_CONFIDENCE:.2f}"
    )

    # Run matcher on all fixture pairs
    decisions = [_run_pair(p) for p in all_pairs]

    # Compute metrics
    precision, recall, n_tp, n_fp, n_tn, n_fn = _compute_precision_recall(decisions)
    precision_pass = precision >= PRECISION_GATE
    recall_pass = recall > RECALL_GATE

    # Real-catalogue scan
    cat_path = _UNIFIED_DIR / "catalogue.parquet"
    if cat_path.exists():
        print(f"Loading real catalogue: {cat_path}")
        cat_df = pd.read_parquet(cat_path)
        print(f"  {len(cat_df):,} items, {cat_df['store'].nunique()} stores")
        real_matches, real_sample_size = _run_real_catalogue_scan(cat_df)
    else:
        print(f"WARNING: catalogue not found at {cat_path} — skipping real-catalogue scan")
        real_matches, real_sample_size = 0, 0

    overall_pass = precision_pass and recall_pass

    report = EvalReport(
        decisions=decisions,
        precision=round(precision, 4),
        recall=round(recall, 4),
        n_tp=n_tp,
        n_fp=n_fp,
        n_fn=n_fn,
        n_tn=n_tn,
        precision_pass=precision_pass,
        recall_pass=recall_pass,
        overall_pass=overall_pass,
        real_catalogue_matches=real_matches,
        real_catalogue_sample_size=real_sample_size,
    )

    _print_report(report)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
