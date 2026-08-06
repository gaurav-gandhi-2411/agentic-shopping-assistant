#!/usr/bin/env python
"""Coherence-judge calibration harness.

Runs multiple candidate LLM-judge backends against eval/fixtures/
coherence_calibration.yaml's hand-scored ground truth and applies a
MECHANICAL ship/no-ship decision rule (never eyeballed):

    - every good/bad look's derived (occasion_score, coherence_score) must
      fall in its expected bucket exactly (anchor_passes_calibration's bar:
      good >=4/4 both axes, bad <=2/2 both axes).
    - every borderline look's derived scores must land in {2,3,4} on BOTH
      axes (not at either pole 1 or 5) — plausibility, not exact-match.

A candidate SHIPS only if it clears every good/bad look AND every borderline
look. This mirrors (and generalizes to n=20, 3 buckets) the 4-anchor
calibration check already in scripts/eval_model.py's judge stage.

IMPORTANT — labeling circularity: the ground truth here was hand-scored by
Claude against a written rubric (see the fixture file header), and several
candidate judges below are also Claude-family or other LLMs. A candidate that
SHIPS by this harness has been shown to "agree with a rubric-following
labeler" — NOT "agree with a human stylist." Real-stylist validation of
either the ground truth or the winning judge remains unmeasured. State this
in every report this script produces; do not let a passing run be read as
more than that.

Usage:
    python -m eval.judge_calibration
    python -m eval.judge_calibration --candidates ollama:llama3.1:8b,gemini
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCRIPTS_DIR = _ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

load_dotenv(_ROOT / ".env")  # GEMINI_API_KEY etc. — matches scripts/eval_harness.py's convention

from eval_model import (  # noqa: E402 — path setup above must run first
    JUDGE_SYSTEM_PROMPT,
    anchor_passes_calibration,
    build_judge_prompt,
    parse_judge_json,
)
from src.catalogue.loader import load_config  # noqa: E402
from src.llm.client import get_llm_client  # noqa: E402

_FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "coherence_calibration.yaml"
_REPORTS_DIR = _ROOT / "reports"

# ---------------------------------------------------------------------------
# Candidate backend specs: (label, provider, model_override_key, model_value)
# ---------------------------------------------------------------------------
_CANDIDATES: dict[str, dict[str, Any]] = {
    "ollama:llama3.1:8b": {
        "provider": "ollama", "model": "llama3.1:8b", "decomposed": False,
        "note": "current production judge — known-broken baseline, rerun for before/after",
    },
    "ollama:qwen3:30b-a3b": {
        "provider": "ollama", "model": "qwen3:30b-a3b", "decomposed": False,
        "note": "strongest local model already pulled (18GB MoE), free",
        # Qwen3 is a hybrid-reasoning model — thinking mode is on by default and
        # spends the token budget on an internal <think> block before ever
        # emitting the JSON answer. config.yaml's production max_tokens=400 is
        # sized for non-reasoning llama3.1:8b and left qwen3 with nothing left
        # to answer with: a first attempt at 400 produced 8/20 (40%) UNPARSEABLE
        # (empty completions, confirmed via a direct raw-output probe — not
        # malformed JSON, zero output). 4096 gives the thinking block room to
        # finish before the JSON answer.
        "max_tokens": 4096,
    },
    "gemini-2.5-flash": {
        "provider": "gemini", "model": "gemini-2.5-flash", "decomposed": False,
        "note": "free-tier API",
        # Free tier is capped at 5 requests/minute (confirmed via live 429
        # RESOURCE_EXHAUSTED response, quotaValue=5) — GeminiClient's own
        # 1s/3s retry backoff is far too short for that cap and produced 100%
        # UNPARSEABLE (0% pass) on a first attempt that was a rate-limit
        # artifact, not a judge-quality result. 13s/call keeps every call
        # under the 5/min ceiling with margin.
        "min_interval_s": 13.0,
    },
    "ollama:qwen3:30b-a3b-decomposed": {
        "provider": "ollama", "model": "qwen3:30b-a3b", "decomposed": True,
        "note": "rubric-decomposed prompt (explicit binary defect questions) on the strongest local backbone",
        "max_tokens": 4096,  # see ollama:qwen3:30b-a3b's comment — same thinking-mode fix
    },
}


def _decomposed_prompt(occasion_slug: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    lines = [
        f"- {it.get('prod_name') or it.get('display_name') or '?'} "
        f"(colour: {it.get('colour') or '?'}, type: {it.get('product_type') or '?'}, "
        f"gender: {it.get('gender') or '?'})"
        for it in items
    ]
    item_block = "\n".join(lines) if lines else "(no items)"
    user = (
        f"Occasion: {occasion_slug}\n"
        f"Outfit items:\n{item_block}\n\n"
        "Answer FOUR yes/no questions about this outfit, then rate coherence:\n"
        "(a) gender_consistent — could a single wearer plausibly wear every item "
        "here (no mixing men's-only and women's-only pieces)?\n"
        "(b) occasion_register_ok — does the outfit's formality and vibe match the "
        "stated occasion (not too casual, not too dressy, not the wrong event type)?\n"
        "(c) formality_ok — do all items share one formality level (no gym wear with "
        "eveningwear, no loungewear with festive wear)?\n"
        "(d) palette_ok — do the colours work together rather than clash?\n"
        "(e) coherence_score (1-5) — overall, do these items work together as one look?\n\n"
        'Return STRICT JSON only: {"gender_consistent": <bool>, "occasion_register_ok": '
        '<bool>, "formality_ok": <bool>, "palette_ok": <bool>, "coherence_score": <1-5 int>, '
        '"reason": "<one sentence>"}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _parse_decomposed(text: str | None) -> dict[str, Any] | None:
    import json
    import re

    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    required_bools = ("gender_consistent", "occasion_register_ok", "formality_ok", "palette_ok")
    if any(not isinstance(obj.get(k), bool) for k in required_bools):
        return None
    coherence_score = obj.get("coherence_score")
    if isinstance(coherence_score, bool) or not isinstance(coherence_score, int):
        return None
    if not (1 <= coherence_score <= 5):
        return None
    return {k: obj[k] for k in required_bools} | {
        "coherence_score": coherence_score,
        "reason": str(obj.get("reason", "")),
    }


def _derive_scores_from_decomposed(parsed: dict[str, Any]) -> dict[str, int]:
    """Map the 4 binary defect flags + coherence_score onto the same
    (occasion_score, coherence_score) scale the holistic prompt/ground-truth
    uses, so decomposed and holistic candidates are scored by one predicate.
    Mirrors the ground-truth convention documented in the fixture header: a
    hard defect (gender/occasion/formality) caps BOTH axes low; a palette-only
    issue caps occasion_score at 2 but leaves coherence_score's own report.
    """
    hard_defect = not (
        parsed["gender_consistent"] and parsed["occasion_register_ok"] and parsed["formality_ok"]
    )
    if hard_defect:
        occasion_score = 1
        coherence_score = min(parsed["coherence_score"], 2)
    elif not parsed["palette_ok"]:
        occasion_score = 2
        coherence_score = min(parsed["coherence_score"], 2)
    else:
        occasion_score = 5 if parsed["coherence_score"] >= 4 else 4
        coherence_score = parsed["coherence_score"]
    return {"occasion_score": occasion_score, "coherence_score": coherence_score}


@dataclass
class LookResult:
    look_id: str
    bucket: str
    expected: dict[str, Any]
    scored: dict[str, Any] | None
    passed: bool
    detail: str = ""


@dataclass
class CandidateResult:
    name: str
    note: str
    results: list[LookResult] = field(default_factory=list)

    @property
    def ships(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0


def _borderline_plausible(scored: dict[str, Any] | None) -> bool:
    if scored is None:
        return False
    return scored["occasion_score"] in (2, 3, 4) and scored["coherence_score"] in (2, 3, 4)


def load_fixture() -> list[dict[str, Any]]:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["looks"]


def run_candidate(name: str, spec: dict[str, Any], looks: list[dict[str, Any]]) -> CandidateResult:
    config = load_config()
    config["llm"]["provider"] = spec["provider"]
    if spec["provider"] == "ollama":
        config["llm"]["model"] = spec["model"]
    elif spec["provider"] == "gemini":
        config["llm"]["gemini_model"] = spec["model"]
    client = get_llm_client(config)

    min_interval_s = spec.get("min_interval_s", 0.0)
    result = CandidateResult(name=name, note=spec["note"])
    for look in looks:
        occasion_slug = look["occasion_slug"]
        items = look["items"]
        expected = look["hand_score"]

        if spec["decomposed"]:
            messages = _decomposed_prompt(occasion_slug, items)
        else:
            messages = build_judge_prompt(occasion_slug, items)

        parsed = None
        for _attempt in range(2):
            if min_interval_s:
                time.sleep(min_interval_s)
            try:
                raw = client.chat(messages, max_tokens=spec.get("max_tokens"))
            except Exception as exc:  # noqa: BLE001 — a judge call failing is a data point, not a crash
                raw = None
                print(f"    [warn] {name} call failed on {look['id']}: {exc}")
            parsed = _parse_decomposed(raw) if spec["decomposed"] else parse_judge_json(raw)
            if parsed is not None:
                break

        if parsed is None:
            scored = None
        elif spec["decomposed"]:
            scored = _derive_scores_from_decomposed(parsed)
        else:
            scored = parsed

        bucket = look["bucket"]
        if bucket in ("good", "bad"):
            passed = anchor_passes_calibration(scored, bucket)
            detail = f"expected {bucket} bucket"
        else:
            passed = _borderline_plausible(scored)
            detail = "expected non-extreme (2-4 both axes)"

        result.results.append(
            LookResult(look_id=look["id"], bucket=bucket, expected=expected, scored=scored, passed=passed, detail=detail)
        )
        score_str = (
            f"occ={scored['occasion_score']} coh={scored['coherence_score']}"
            if scored else "UNPARSEABLE"
        )
        print(f"    {look['id']:<45} [{bucket:<10}] {score_str:<22} {'PASS' if passed else 'FAIL'}")
    return result


def render_report(candidates: list[CandidateResult], looks: list[dict[str, Any]]) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        f"# Coherence-judge calibration — {ts}",
        "",
        "**Circularity caveat (read this before the table):** ground truth was hand-scored "
        "by Claude against a written rubric (`eval/fixtures/coherence_calibration.yaml`), and "
        "several candidates below are also LLM judges. A candidate that SHIPS here has been "
        "shown to *agree with a rubric-following labeler* — NOT shown to agree with a human "
        "stylist. No real-stylist validation exists for either the ground truth or the winning "
        "judge; that remains an open gap.",
        "",
        f"Fixture: `eval/fixtures/coherence_calibration.yaml` — "
        f"{sum(1 for l in looks if l['bucket']=='good')} good / "
        f"{sum(1 for l in looks if l['bucket']=='bad')} bad / "
        f"{sum(1 for l in looks if l['bucket']=='borderline')} borderline "
        f"({len(looks)} total).",
        "",
        "## Decision rule",
        "A candidate SHIPS only if it passes every good/bad look (anchor bar: good >=4/4 both "
        "axes, bad <=2/2 both axes) AND lands every borderline look in {2,3,4} on both axes "
        "(non-extreme). Applied mechanically below — no eyeballing.",
        "",
        "## Results",
        "",
        "| Candidate | Pass rate | Good/Bad correct | Borderline plausible | SHIPS |",
        "|---|---|---|---|---|",
    ]
    for c in candidates:
        gb = [r for r in c.results if r.bucket in ("good", "bad")]
        bl = [r for r in c.results if r.bucket == "borderline"]
        gb_rate = f"{sum(r.passed for r in gb)}/{len(gb)}" if gb else "n/a"
        bl_rate = f"{sum(r.passed for r in bl)}/{len(bl)}" if bl else "n/a"
        lines.append(
            f"| {c.name} | {c.pass_rate:.0%} | {gb_rate} | {bl_rate} | "
            f"{'**YES**' if c.ships else 'no'} |"
        )
    lines.append("")
    for c in candidates:
        lines.append(f"### {c.name}")
        lines.append(f"_{c.note}_")
        lines.append("")
        lines.append("| Look | Bucket | Expected | Scored | Result |")
        lines.append("|---|---|---|---|---|")
        for r in c.results:
            exp = f"occ={r.expected['occasion_score']} coh={r.expected['coherence_score']}"
            got = (
                f"occ={r.scored['occasion_score']} coh={r.scored['coherence_score']}"
                if r.scored else "UNPARSEABLE"
            )
            lines.append(f"| {r.look_id} | {r.bucket} | {exp} | {got} | {'PASS' if r.passed else 'FAIL'} |")
        lines.append("")

    shippable = [c for c in candidates if c.ships]
    lines.append("## Verdict")
    if shippable:
        for c in shippable:
            lines.append(f"- **{c.name} SHIPS** — passes every good/bad look and every borderline look.")
    else:
        lines.append(
            "- **No candidate ships.** No LLM judge tested here reliably separates good from "
            "broken outfits at the required bar. Recommendation: do NOT wire any of these into "
            "the production eval as a scored metric — a broken judge reported as a number is "
            "worse than no number. Hand-score a small rotating sample periodically instead "
            "(see the wave's final report for a proposed cadence)."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=",".join(_CANDIDATES.keys()))
    args = parser.parse_args()

    looks = load_fixture()
    selected = [c.strip() for c in args.candidates.split(",") if c.strip()]

    candidate_results: list[CandidateResult] = []
    for name in selected:
        if name not in _CANDIDATES:
            print(f"[skip] unknown candidate {name!r}")
            continue
        spec = _CANDIDATES[name]
        print(f"\n=== {name} ({spec['note']}) ===")
        t0 = time.monotonic()
        result = run_candidate(name, spec, looks)
        elapsed = time.monotonic() - t0
        print(f"  -> pass_rate={result.pass_rate:.0%} ships={result.ships} ({elapsed:.1f}s)")
        candidate_results.append(result)

    report = render_report(candidate_results, looks)
    _REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _REPORTS_DIR / f"judge_calibration_{ts}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
