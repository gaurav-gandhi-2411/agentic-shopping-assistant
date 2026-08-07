#!/usr/bin/env python
"""Multi-family LLM-consensus labeling — verifiable, not just averaged.

Motivation: eval/judge_calibration.py found every candidate judge shares a
LENIENT bias (the rubric-decomposed variant scored 13/14 on unambiguous good/
bad looks but only 1/6 on borderline ones — it learned to default to "fine").
Consensus among similarly-biased judges would amplify, not correct, that
bias. This script instead:

  1. Uses judges from 4 DIFFERENT model families (not variants of one):
     qwen3:30b-a3b (Alibaba, local/free), gemma2:9b (Google DeepMind,
     local/free), gemini-2.5-flash (Google, API/free-tier),
     openai/gpt-oss-20b:free via OpenRouter (OpenAI, API/free-tier).
  2. Runs all 4 blind (no ground truth shown) against every look in
     eval/fixtures/coherence_calibration.yaml (not just the 5 flagged items —
     agreement statistics on n=5 would be too noisy to trust; n=20 gives a
     real number, and the anchor-validation step needs the bad/good anchors
     anyway).
  3. Maps each judge's (occasion_score, coherence_score) to a bucket verdict
     (good/bad/borderline) via the SAME bar as anchor_passes_calibration.
  4. Reports RAW pairwise agreement AND Krippendorff's alpha (nominal,
     chance-corrected) across the 4 primary judges — low agreement on
     borderline items is reported as a finding, not averaged away.
  5. VALIDATES the consensus mechanism against the fixture's own good/bad
     anchors before trusting it for anything: if majority-vote consensus
     fails to bucket bad_001_mixed_gender as "bad" (or any bad/good anchor
     into its expected bucket), the consensus is declared unusable.
  6. For items with no 3/4 majority among the primary 4, calls a 5th,
     independent-family tiebreaker (llama3.1:8b, Meta, local/free). If still
     no majority among all 5, the item is labeled "ambiguous" and excluded
     from scoring — never forced to a label.

Usage:
    python -m eval.consensus_labeling
"""
from __future__ import annotations

import itertools
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
load_dotenv(_ROOT / ".env")

from eval_model import anchor_passes_calibration, build_judge_prompt, parse_judge_json  # noqa: E402

from src.catalogue.loader import load_config  # noqa: E402
from src.llm.client import get_llm_client  # noqa: E402

_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "coherence_calibration.yaml"

# The 5 items originally flagged for GG's manual spot-check and never
# reviewed — the actual deliverable this script exists to unblock.
_TARGET_5 = {
    "borderline_002_haldi_velvet_dupatta",
    "borderline_004_reception_plain_flats",
    "borderline_006_gym_overdressed_hoodie",
    "good_003_wedding_reception",
    "bad_005_bridal_weight_at_mehendi",
}

# gemini-2.5-flash was in the original 4-judge plan (Alibaba/Google-DeepMind/
# Google/OpenAI) but its free-tier daily quota (20 req/day) was already
# exhausted before this run started (confirmed: the very first call of the
# run 429'd) and its reset time is opaque (no login-accessible dashboard —
# see DEPLOY.md's own note on this). Not worth burning the "free-tier only"
# budget waiting it out or paying to unblock it. Dropped for this run; the
# remaining 3 primary + 1 tiebreaker still span 4 distinct families.
_PRIMARY_JUDGES: dict[str, dict[str, Any]] = {
    "qwen3:30b-a3b": {"provider": "ollama", "model": "qwen3:30b-a3b", "max_tokens": 4096},
    "gemma2:9b": {"provider": "ollama", "model": "gemma2:9b", "max_tokens": 2048},
    "openrouter-gpt-oss-20b": {"provider": "openrouter"},
}
_TIEBREAKER = {"name": "llama3.1:8b (Meta, tiebreaker)", "provider": "ollama", "model": "llama3.1:8b"}


def _bucket(scored: dict[str, Any] | None) -> str | None:
    if scored is None:
        return None
    if anchor_passes_calibration(scored, "good"):
        return "good"
    if anchor_passes_calibration(scored, "bad"):
        return "bad"
    return "borderline"


def _client_for(spec: dict[str, Any]):
    config = load_config()
    config["llm"]["provider"] = spec["provider"]
    if spec["provider"] == "ollama":
        config["llm"]["model"] = spec["model"]
    elif spec["provider"] == "gemini":
        config["llm"]["gemini_model"] = spec["model"]
    return get_llm_client(config)


def _score_look(client, spec: dict[str, Any], look: dict[str, Any]) -> dict[str, Any] | None:
    messages = build_judge_prompt(look["occasion_slug"], look["items"])
    if spec.get("min_interval_s"):
        time.sleep(spec["min_interval_s"])
    for _attempt in range(2):
        try:
            raw = client.chat(messages, max_tokens=spec.get("max_tokens"))
        except Exception as exc:  # noqa: BLE001 — a failed call is a data point
            print(f"    [warn] call failed: {exc}")
            raw = None
        parsed = parse_judge_json(raw)
        if parsed is not None:
            return parsed
        if spec.get("min_interval_s"):
            time.sleep(spec["min_interval_s"])
    return None


def _krippendorff_alpha_nominal(votes_by_item: list[list[str]]) -> float | None:
    """Krippendorff's alpha for nominal data with missing values allowed.
    votes_by_item: one list of category labels per item (one entry per rater
    that actually produced a verdict for that item; raters need not all be
    present for all items). Returns None if there's no observed disagreement
    to correct for (fewer than 2 items with >=2 raters).
    """
    all_units = [u for u in votes_by_item if len(u) >= 2]
    if not all_units:
        return None

    def _pair_disagreements(unit: list[str]) -> list[int]:
        return [0 if a == b else 1 for a, b in itertools.combinations(unit, 2)]

    observed_pairs: list[int] = []
    for unit in all_units:
        observed_pairs.extend(_pair_disagreements(unit))
    if not observed_pairs:
        return 1.0  # perfect agreement everywhere there was overlap
    d_o = sum(observed_pairs) / len(observed_pairs)

    all_votes = [v for unit in all_units for v in unit]
    n = len(all_votes)
    counts = Counter(all_votes)
    # Expected disagreement under chance, from the overall category distribution.
    d_e = 1.0 - sum((c / n) * ((c - 1) / (n - 1)) for c in counts.values()) if n > 1 else 0.0
    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0
    return 1.0 - (d_o / d_e)


def main() -> None:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        looks = yaml.safe_load(f)["looks"]

    clients = {name: _client_for(spec) for name, spec in _PRIMARY_JUDGES.items()}
    tiebreak_client = _client_for(_TIEBREAKER)

    per_item_votes: dict[str, dict[str, str | None]] = {}
    for look in looks:
        print(f"\n=== {look['id']} [{look['bucket']}] ===")
        votes: dict[str, str | None] = {}
        for name, spec in _PRIMARY_JUDGES.items():
            scored = _score_look(clients[name], spec, look)
            bucket = _bucket(scored)
            votes[name] = bucket
            print(f"  {name:<25} -> {bucket}  ({scored})")
        per_item_votes[look["id"]] = votes

    # --- Agreement statistics across the 4 primary judges ---
    judge_names = list(_PRIMARY_JUDGES.keys())
    pairwise_agree = 0
    pairwise_total = 0
    votes_by_item_for_alpha: list[list[str]] = []
    for look_id, votes in per_item_votes.items():
        present = [v for v in votes.values() if v is not None]
        votes_by_item_for_alpha.append(present)
        for a, b in itertools.combinations(judge_names, 2):
            va, vb = votes.get(a), votes.get(b)
            if va is None or vb is None:
                continue
            pairwise_total += 1
            if va == vb:
                pairwise_agree += 1

    raw_agreement = pairwise_agree / pairwise_total if pairwise_total else None
    alpha = _krippendorff_alpha_nominal(votes_by_item_for_alpha)

    print("\n" + "=" * 70)
    print("INTER-JUDGE AGREEMENT (4 primary judges, all 20 calibration looks)")
    print(f"  raw pairwise agreement: {raw_agreement:.3f}" if raw_agreement is not None else "  raw pairwise agreement: n/a")
    print(f"  Krippendorff's alpha (nominal): {alpha:.3f}" if alpha is not None else "  Krippendorff's alpha: n/a")

    # Borderline-specific agreement (the bias this whole exercise is checking for).
    borderline_ids = {lk["id"] for lk in looks if lk["bucket"] == "borderline"}
    bl_agree = bl_total = 0
    for look_id in borderline_ids:
        votes = per_item_votes[look_id]
        present_names = [n for n in judge_names if votes.get(n) is not None]
        for a, b in itertools.combinations(present_names, 2):
            bl_total += 1
            if votes[a] == votes[b]:
                bl_agree += 1
    bl_rate = bl_agree / bl_total if bl_total else None
    print(f"  raw pairwise agreement, BORDERLINE items only: {bl_rate:.3f}" if bl_rate is not None else "  borderline agreement: n/a")

    # --- Anchor validation: does majority vote correctly bucket every good/bad anchor? ---
    print("\nANCHOR VALIDATION (majority vote among the 4 primary judges)")
    anchor_failures = []
    for look in looks:
        if look["bucket"] not in ("good", "bad"):
            continue
        votes = [v for v in per_item_votes[look["id"]].values() if v is not None]
        if not votes:
            continue
        majority_bucket, majority_count = Counter(votes).most_common(1)[0]
        passes = majority_bucket == look["bucket"] and majority_count > len(votes) / 2
        print(f"  {look['id']:<45} expected={look['bucket']:<10} majority={majority_bucket} ({majority_count}/{len(votes)}) {'PASS' if passes else 'FAIL'}")
        if not passes:
            anchor_failures.append(look["id"])

    consensus_usable = not anchor_failures
    print(f"\nCONSENSUS MECHANISM: {'USABLE' if consensus_usable else 'NOT USABLE'}"
          + (f" — failed anchors: {anchor_failures}" if anchor_failures else ""))

    # --- Resolve final label per item: majority vote, else tiebreaker, else ambiguous ---
    final_labels: dict[str, str] = {}
    for look_id, votes in per_item_votes.items():
        present = [v for v in votes.values() if v is not None]
        if not present:
            final_labels[look_id] = "ambiguous (no judge responded)"
            continue
        counts = Counter(present)
        top_bucket, top_count = counts.most_common(1)[0]
        if top_count > len(present) / 2:
            final_labels[look_id] = top_bucket
            continue
        # No majority among primary judges — call the tiebreaker.
        look = next(lk for lk in looks if lk["id"] == look_id)
        print(f"\n  [tiebreak] {look_id}: no majority among {dict(counts)}, calling {_TIEBREAKER['name']}...")
        tb_scored = _score_look(tiebreak_client, _TIEBREAKER, look)
        tb_bucket = _bucket(tb_scored)
        if tb_bucket is not None:
            present.append(tb_bucket)
        counts = Counter(present)
        top_bucket, top_count = counts.most_common(1)[0]
        if top_count > len(present) / 2:
            final_labels[look_id] = f"{top_bucket} (via tiebreaker)"
        else:
            final_labels[look_id] = f"ambiguous (no majority: {dict(counts)})"

    print("\n" + "=" * 70)
    print("FINAL LABELS — the 5 originally-flagged items (the deliverable)")
    for look_id in sorted(_TARGET_5):
        look = next(lk for lk in looks if lk["id"] == look_id)
        print(f"  {look_id:<45} hand-label={look['bucket']:<10} consensus={final_labels.get(look_id)}")

    import json
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _ROOT / "reports" / f"consensus_labeling_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "per_item_votes": per_item_votes,
                "raw_pairwise_agreement": raw_agreement,
                "krippendorff_alpha": alpha,
                "borderline_only_agreement": bl_rate,
                "anchor_validation": {"usable": consensus_usable, "failures": anchor_failures},
                "final_labels": final_labels,
                "target_5_results": {k: final_labels.get(k) for k in _TARGET_5},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
