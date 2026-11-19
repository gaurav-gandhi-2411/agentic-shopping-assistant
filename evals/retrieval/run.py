"""Stage driver for the V4 retrieval eval.

    python -m evals.retrieval.run --stage candidates --index-dir <dir> [--limit N]
    python -m evals.retrieval.run --stage plan                       # token/day estimates
    python -m evals.retrieval.run --stage judge --mock [--limit N]   # calibration gate + labels
    python -m evals.retrieval.run --stage judge --export-packets     # Claude packets (no LLM)
    python -m evals.retrieval.run --stage judge --import-claude      # import Claude results
    python -m evals.retrieval.run --stage variants --mock [--limit N] [--reps 3]  # A A2 B C
    python -m evals.retrieval.run --stage factors --mock             # F1/F2 on the best two arms
    python -m evals.retrieval.run --stage report [--mock]
    python -m evals.retrieval.relabel --mock ...                     # consensus relabelling

Live Groq calls need BOTH `--live` and STAGE2_LIVE_OK=1; otherwise the run refuses. `--mock`
uses rule-based fakes and never touches the network. Everything is resumable (`--resume` is the
default behaviour: cached calls are never repeated).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable

from evals.retrieval import agreement as ag
from evals.retrieval.common import (
    CANDIDATE_DIR,
    JUDGE_DAILY_TOKEN_BUDGET,
    JUDGE_MODELS,
    MIN_REQUEST_INTERVAL_S,
    REPO,
    RERANKER_DAILY_TOKEN_BUDGET,
    RERANKER_MODEL,
    RUN_DIR,
    SEED,
    count_tokens,
    load_candidates,
    load_cases,
    sha,
)
from evals.retrieval.factors import (
    demote_sets,
    interleave_gender,
    is_gender_unspecified,
    is_set_demotion_query,
    pool_loss,
)
from evals.retrieval.groq_judge import (
    GroqRunner,
    Ledger,
    ModelSpec,
    ResultCache,
    assert_live_allowed,
    httpx_transport,
)
from evals.retrieval.judge import (
    JUDGE_MAX_TOKENS,
    build_judge_prompt,
    calibration_cases,
    labels_to_ids,
    load_rubric,
    parse_labels,
    score_calibration,
)
from evals.retrieval.mock import mock_transport
from evals.retrieval.packets import export_packets, import_results
from evals.retrieval.post_rerank import apply_post_rerank
from evals.retrieval.report import build_factor_report, build_report
from evals.retrieval.variants import (
    assert_no_length_hits,
    build_compact_prompt,
    run_variant_a,
    run_variant_a2,
    run_variant_b,
    run_variant_c,
)

JUDGE_PARAMS: dict[str, dict[str, Any]] = {
    "gpt-oss-120b": {"reasoning_effort": "low"},
    # UNVERIFIED: qwen3.8-27b is a Groq preview model; "none" is the documented switch for the
    # qwen3 family. If the first live call returns a 400 about this parameter, stage 2 stops.
    "qwen3.8-27b": {"reasoning_effort": "none"},
}


def judge_specs(budget: int = JUDGE_DAILY_TOKEN_BUDGET) -> dict[str, ModelSpec]:
    """Groq judge specs. gpt-oss-20b (the primary model) is never a judge."""
    return {
        name: ModelSpec(name, model, JUDGE_PARAMS[name], JUDGE_MAX_TOKENS, budget)
        for name, model in JUDGE_MODELS.items()
    }


def reranker_spec(budget: int = RERANKER_DAILY_TOKEN_BUDGET) -> ModelSpec:
    return ModelSpec("rerank-20b", RERANKER_MODEL, {"reasoning_effort": "low"}, 150, budget)


class GroqLLM:
    """`generate()` adapter over GroqRunner for the reranker variants (A and C).

    Differs from production's GroqClient in one documented way: no model-fallback chain. When
    the model returns empty content (reasoning ate the cap) production would retry gpt-oss-120b
    and qwen with the same cap and finally return the busy message; either way the reranker
    sees a non-JSON reply and falls back to retrieval order, which is what an empty reply does
    here too (the extra chain tokens/latency are not modelled)."""

    def __init__(self, runner: GroqRunner, spec: ModelSpec, variant: str) -> None:
        self.runner, self.spec, self.variant = runner, spec, variant
        self.query_id = ""
        self.rep = 1  # repeats of a query are separate cache entries (they must be real calls)
        self.last_meta: dict[str, Any] = {}

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        spec = dataclasses.replace(
            self.spec,
            name=f"{self.spec.name}-{self.variant}-r{self.rep}",
            max_tokens=int(kw.get("max_tokens", self.spec.max_tokens)),
        )
        res = self.runner.call(
            spec, self.query_id, system or "", prompt, sha((system or "") + prompt)
        )
        if res is None:
            raise RuntimeError("Groq call failed")
        self.last_meta = {
            "finish_reason": res["finish_reason"],
            "usage": res["usage"],
            "latency_ms": res["latency_ms"],
        }
        return str(res["text"])


class MockRerankLLM:
    """Deterministic fake reranker model (A: JSON, C: numbers).

    Every 5th JSON call truncates when the cap is the production 150 (a 600 cap never does, like
    the real model); every 7th call picks a different fifth item, so repeated runs of the same
    query can differ (exercises the stability code)."""

    def __init__(self) -> None:
        self.n = 0
        self.last_meta: dict[str, Any] = {}

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.n += 1
        cap = int(kw.get("max_tokens", 0))
        compact = (system or "").startswith("You rank fashion products")
        truncates = not compact and self.n % 5 == 0 and cap <= 150
        picks = "[3, 1, 2, 5, 6]" if self.n % 7 == 0 else "[3, 1, 2, 5, 4]"
        if compact:
            out = "3,1,2,5,4,6,7,8"
        elif truncates:
            out = '{"selected": [3, 1, 2, 5, 4], "reasoning": "elegant and on-bud'  # truncated
        else:
            out = '{"selected": ' + picks + ', "reasoning": "best fit"}'
        comp = cap if truncates else count_tokens(out) + 60
        self.last_meta = {
            "finish_reason": "length" if comp >= cap else "stop",
            "usage": {
                "prompt_tokens": count_tokens((system or "") + prompt),
                "completion_tokens": comp,
            },
            "latency_ms": 400 + self.n,
        }
        return out


def usable_cases(limit: int | None, cand_dir: Path) -> list[dict[str, Any]]:
    """Cases that have at least one candidate (queries the agent sends to clarify have none)."""
    out = []
    for c in load_cases(limit):
        if load_candidates(c["id"], cand_dir)["items"]:
            out.append(c)
        else:
            print(f"  skipping {c['id']}: no candidates (production routes it to clarify)")
    return out


def _runner(
    mock: bool, live: bool, out_dir: Path, mock_responder: Callable[..., Any] | None
) -> GroqRunner:
    if mock:
        transport = mock_transport(mock_responder) if mock_responder else mock_transport()  # noqa: E501
        return GroqRunner(
            transport,
            Ledger(out_dir / "ledger.jsonl"),
            ResultCache(out_dir / "cache.jsonl"),
            min_interval_s=0.0,
            sleep=lambda s: None,
        )
    assert_live_allowed(live)
    return GroqRunner(
        httpx_transport(), Ledger(out_dir / "ledger.jsonl"), ResultCache(out_dir / "cache.jsonl")
    )


def run_calibration(
    runner: GroqRunner,
    specs: dict[str, ModelSpec],
    seed: int = SEED,
) -> dict[str, dict[str, int]]:
    """Each judge labels the 20 calibration items; returns {judge: {article_id: label}}.

    A judge that misses the pass criterion (>=18/20 exact, none off by two) is reported and
    its labels are still returned so the caller can print them, but `calibration_ok` fails."""
    out: dict[str, dict[str, int]] = {name: {} for name in specs}
    for cal in calibration_cases():
        p = build_judge_prompt(cal["case"], cal["candidates"], seed)
        for name, spec in specs.items():
            res = runner.call(spec, cal["case"]["id"], p.system, p.user, p.prompt_hash)
            parsed = parse_labels(res["text"], len(p.order)) if res else None
            if parsed:
                out[name].update(
                    {
                        f"{cal['case']['id']}:{a}": v
                        for a, v in labels_to_ids(parsed, p.order).items()
                    }
                )
    return out


def _cal_gold() -> dict[str, int]:
    return {f"{c['case']['id']}:{a}": v for c in calibration_cases() for a, v in c["gold"].items()}


def calibration_ok(labels: dict[str, dict[str, int]]) -> tuple[bool, list[str]]:
    gold = _cal_gold()
    msgs, ok = [], True
    for judge, pred in labels.items():
        s = score_calibration(pred, gold)
        msgs.append(f"{judge}: exact {s['exact']}/{s['n']}, off-by-two {s['off_by_two']}")
        ok &= s["passed"]
    return ok, msgs


def stage_judge(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    cases = usable_cases(args.limit, Path(args.cand))
    rubric = load_rubric()
    if args.export_packets:
        prompts = [
            build_judge_prompt(c, load_candidates(c["id"], Path(args.cand))["items"], SEED)
            for c in cases
        ]
        cal = [build_judge_prompt(x["case"], x["candidates"], SEED) for x in calibration_cases()]
        pdir = out_dir / "claude_packets"
        n1 = len(export_packets(cal, pdir, "cal"))
        n2 = len(export_packets(prompts, pdir, "main"))
        print(f"wrote {n1} calibration + {n2} main packets to {pdir}")
        return 0
    if args.import_claude:
        imported, problems = import_results(out_dir / "claude_packets")
        cal_lab = {k: v for k, v in imported["claude"].items() if k.startswith("cal")}
        main_lab = {k: v for k, v in imported["claude"].items() if not k.startswith("cal")}
        (out_dir / "claude_packets" / "labels_cal_claude.json").write_text(
            json.dumps(cal_lab, indent=1)
        )
        (out_dir / "labels_claude.json").write_text(json.dumps(main_lab, indent=1))
        print(
            f"imported {len(main_lab)} main + {len(cal_lab)} calibration Claude labels; "
            f"problems: {problems or 'none'}"
        )
        return 1 if problems else 0
    specs = judge_specs(args.judge_budget)
    runner = _runner(args.mock, args.live, out_dir, None)
    cal_labels = run_calibration(runner, specs)
    ok, msgs = calibration_ok(cal_labels)
    print("calibration per judge:\n  " + "\n  ".join(msgs))
    if not ok:
        print("REFUSING to continue: a judge failed the calibration criterion.")
        return 2
    if args.mock:  # the mock stands in for the Claude judge's calibration labels too
        cal_labels["claude"] = _cal_gold()
    elif (out_dir / "claude_packets" / "labels_cal_claude.json").exists():
        cal_labels["claude"] = json.loads(
            (out_dir / "claude_packets" / "labels_cal_claude.json").read_text()
        )
    passed, msg = ag.calibration_gate(cal_labels)
    if len(cal_labels) < 3 and not args.allow_partial_gate:
        passed, msg = (
            False,
            "calibration gate needs all 3 judges (Claude calibration missing)\n" + msg,
        )
    print(msg)
    if not passed:
        print("\nRUBRIC:\n" + rubric)
        return 2
    for name, spec in specs.items():
        labels: dict[str, int] = {}
        for c in cases:
            cand = load_candidates(c["id"], Path(args.cand))["items"]
            p = build_judge_prompt(c, cand, SEED)
            res = runner.call(spec, c["id"], p.system, p.user, p.prompt_hash)
            parsed = parse_labels(res["text"], len(p.order)) if res else None
            if parsed is None:
                print(f"  {name} {c['id']}: unparseable/failed output (not labelled)")
                continue
            labels.update({f"{c['id']}:{a}": v for a, v in labels_to_ids(parsed, p.order).items()})
        (out_dir / f"labels_{name}.json").write_text(json.dumps(labels, indent=1))
        print(f"{name}: {len(labels)} labels")
    return 0


def _pool_for(case: dict[str, Any], cand_dir: Path) -> list[dict[str, Any]]:
    return list(load_candidates(case["id"], cand_dir)["items"])


def _gender_for_gate(case: dict[str, Any]) -> str:
    """Gender passed to the occasion coherence gate (search_node: 'unisex' when unstated)."""
    return "unisex" if case["gender"] == "either" else str(case["gender"])


def _with_cards(
    row: dict[str, Any], case: dict[str, Any], pool: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach the post-rerank card ids (replayed search_node tail) to a variant row."""
    by_id = {str(it["article_id"]): it for it in pool}
    shown = [by_id[a] for a in row["final_ids"] if a in by_id]
    cards = apply_post_rerank(case["text"], shown, pool, _gender_for_gate(case))
    return {**row, "cards": [str(it["article_id"]) for it in cards]}


def _make_llms(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """One adapter per LLM arm (A, A2, C); mock fakes need no network and no key."""
    if args.mock:
        return {k: MockRerankLLM() for k in ("A", "A2", "C")}
    runner = _runner(False, args.live, out_dir, None)
    spec = reranker_spec(args.rerank_budget)
    return {k: GroqLLM(runner, spec, k) for k in ("A", "A2", "C")}


def _run_arm(
    arm: str,
    case: dict[str, Any],
    pool: list[dict[str, Any]],
    llm: Any,
    rep: int,
) -> dict[str, Any]:
    if isinstance(llm, GroqLLM):
        llm.query_id, llm.rep = case["id"], rep
    if arm == "A":
        row = run_variant_a(case["text"], pool, llm)
    elif arm == "A2":
        row = run_variant_a2(case["text"], pool, llm)
    elif arm == "C":
        row = run_variant_c(case["text"], pool, llm)
    else:
        row = run_variant_b(pool)
    return {"query_id": case["id"], "rep": rep, **row}


def stage_variants(args: argparse.Namespace) -> int:
    """A, A2, B, C on the frozen pools. A and A2 run `--reps` times per query (stability); the
    order is rep-major so rep 1 of every arm finishes first (the main comparison needs only
    that; later reps can wait for the next day's token budget)."""
    out_dir = Path(args.out)
    cand = Path(args.cand)
    cases = usable_cases(args.limit, cand)
    llms = _make_llms(args, out_dir)
    results: dict[str, list[dict[str, Any]]] = {"A": [], "A2": [], "B": [], "C": []}
    for rep in range(1, args.reps + 1):
        arms = ("A", "A2", "B", "C") if rep == 1 else ("A", "A2")
        for arm in arms:
            for c in cases:
                pool = _pool_for(c, cand)
                row = _run_arm(arm, c, pool, llms.get(arm), rep)
                results[arm].append(_with_cards(row, c, pool))
    (out_dir / "variants.json").write_text(json.dumps(results, indent=1))
    check = _a2_length_check(results["A2"])
    print(f"variants A/A2/B/C over {len(cases)} queries x {args.reps} reps -> variants.json")
    print(check)
    return 0


def _a2_length_check(rows: list[dict[str, Any]]) -> str:
    """The A2 sizing assertion as a printed PASS/FAIL line (the data is kept either way)."""
    try:
        assert_no_length_hits(rows)
    except AssertionError as exc:
        return f"A2 LENGTH CHECK FAILED: {exc}"
    return "A2 length check passed: finish_reason was never 'length'"


def _load_aggregate(out_dir: Path, judges: list[str]) -> tuple[dict, dict]:
    labels = {
        j: json.loads((out_dir / f"labels_{j}.json").read_text())
        for j in judges
        if (out_dir / f"labels_{j}.json").exists()
    }
    return labels, ag.aggregate(labels)


def stage_factors(args: argparse.Namespace) -> int:
    """F1 / F2 on the two best arms (highest nDCG@5 point estimate in the main comparison).

    Needs labels and variants.json. Calls are only made for queries whose pool actually changes
    under the treatment (an unchanged pool reuses the rep-1 row)."""
    out_dir, cand = Path(args.out), Path(args.cand)
    variants = json.loads((out_dir / "variants.json").read_text())
    _, agg = _load_aggregate(out_dir, ["gpt-oss-120b", "qwen3.8-27b", "claude"])
    if not agg:
        print("no judge labels found; run --stage judge first")
        return 2
    cases = usable_cases(args.limit, cand)
    rep = build_report(cases, variants, agg)
    top2: list[str] = rep["top2_by_ndcg5_point_estimate"]
    llms = _make_llms(args, out_dir)
    rep1 = {
        v: {r["query_id"]: r for r in rows if int(r.get("rep", 1)) == 1}
        for v, rows in variants.items()
    }
    out: dict[str, Any] = {"top2": top2, "F1": {}, "F2": {}}
    for factor, flag, transform in (
        ("F1", is_set_demotion_query, demote_sets),
        ("F2", is_gender_unspecified, interleave_gender),
    ):
        flagged = [c for c in cases if flag(c)]
        for v in top2:
            base_rows, treated_rows = [], []
            loss = {"n_before": 0, "n_after": 0, "n_removed": 0, "n_removed_relevant": 0}
            for c in flagged:
                pool = _pool_for(c, cand)
                new_pool = transform(pool)
                base = rep1[v][c["id"]]
                if [it["article_id"] for it in new_pool] == [it["article_id"] for it in pool]:
                    treated = {**base, "reused": True}
                else:
                    treated = _with_cards(_run_arm(v, c, new_pool, llms.get(v), 1), c, new_pool)
                    treated["reused"] = False
                if factor == "F1":
                    lab = {a: x["label"] for a, x in _pool_labels_for(agg, c["id"]).items()}
                    for k, val in pool_loss(pool, new_pool, lab).items():
                        loss[k] += val
                    for r, p in ((base, pool), (treated, new_pool)):
                        by_id = {str(it["article_id"]): it for it in p}
                        r["n_sets_shown"] = sum(
                            1 for a in r["final_ids"] if a in by_id and _is_set_item(by_id[a])
                        )
                base_rows.append(base)
                treated_rows.append(treated)
            entry: dict[str, Any] = {"base": base_rows, "treated": treated_rows}
            if factor == "F1":
                entry["pool_loss"] = loss
            out[factor][v] = entry
    (out_dir / "factors.json").write_text(json.dumps(out, indent=1))
    print(f"factors F1/F2 on {top2} -> factors.json")
    return 0


def _pool_labels_for(agg: dict, qid: str) -> dict:
    prefix = qid + ":"
    return {k[len(prefix) :]: v for k, v in agg.items() if k.startswith(prefix)}


def _is_set_item(it: dict[str, Any]) -> bool:
    from src.agents.outfit.slots import is_multi_piece_set

    return bool(
        is_multi_piece_set(
            it.get("product_type") or "", it.get("prod_name") or it.get("display_name") or ""
        )
    )


def judge_warnings(labels: dict[str, dict[str, int]], expected: int = 3) -> list[str]:
    """Warnings when fewer than `expected` judges contributed labels (or contributed nothing).

    A 2-judge "majority" is a unanimity requirement and any disagreement becomes a tie, so
    every number in the report must carry this caveat."""
    n = sum(1 for lab in labels.values() if lab)
    out = []
    if n < expected:
        out.append(
            f"ONLY {n} of {expected} judges contributed labels ({sorted(k for k, v in labels.items() if v)}): "
            "majority labels are weaker than the designed 3-family consensus; do not quote as such"
        )
    return out


def stage_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    variants = json.loads((out_dir / "variants.json").read_text())
    labels, agg = _load_aggregate(out_dir, ["gpt-oss-120b", "qwen3.8-27b", "claude"])
    if not agg:
        print("no judge labels found; run --stage judge first")
        return 2
    rep = build_report(usable_cases(args.limit, Path(args.cand)), variants, agg)
    rep["agreement"] = ag.pairwise_report(labels)
    rep["warnings"] = judge_warnings(labels)
    rep["a2_length_check"] = _a2_length_check(variants.get("A2", []))
    if (out_dir / "factors.json").exists():
        rep["factors"] = build_factor_report(
            json.loads((out_dir / "factors.json").read_text()), agg
        )
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    keys = ("n_queries", "metrics", "paired", "usage", "rates", "stability")
    print(json.dumps({k: rep[k] for k in keys}, indent=1))
    print(rep["a2_length_check"])
    for w in rep["warnings"]:
        print("WARNING:", w)
    print("LLM-consensus labels (three model families), not human ground truth.")
    return 0


# Completion-token assumptions for the plan (visible output + hidden reasoning at low effort).
# Judges: ~120 visible + reasoning; A: observed prod p50 144 of the 150 cap (v4 log analysis);
# C: ~30 visible + reasoning. Stage 2 replaces these with ledger medians after the first calls.
EST_COMPLETION = {"judge": 400, "calibration": 400, "A": 150, "C": 100}
TPM_LIMIT = 8_000  # Groq free tier per model (architect brief)


def build_plan(
    jt: list[int],
    cal: list[int],
    a_prompt: list[int],
    c_prompt: list[int],
    judge_budget: int = JUDGE_DAILY_TOKEN_BUDGET,
    rerank_budget: int = RERANKER_DAILY_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Per-model token totals, days needed under the daily budget, and TPM-bound minutes."""
    import math

    def mean(x: list[int]) -> float:
        return sum(x) / len(x)

    n = len(jt)
    judge_cal = len(cal) * (mean(cal) + EST_COMPLETION["calibration"])
    judge_main = n * (mean(jt) + EST_COMPLETION["judge"])
    judge_worst = n * (mean(jt) + JUDGE_MAX_TOKENS)
    selfcons = 10 * (mean(jt) + EST_COMPLETION["judge"])
    a_tok = n * (mean(a_prompt) + EST_COMPLETION["A"])
    c_tok = n * (mean(c_prompt) + EST_COMPLETION["C"])
    probe = 3 * (mean(a_prompt) + EST_COMPLETION["A"])
    per_judge = judge_cal + judge_main
    plan: dict[str, Any] = {
        "n_queries_with_candidates": n,
        "mean_prompt_tokens": {
            "judge": mean(jt),
            "calibration": mean(cal),
            "A": mean(a_prompt),
            "C": mean(c_prompt),
        },
        "tokens": {
            "per_judge_calibration": judge_cal,
            "per_judge_main_expected": judge_main,
            "per_judge_main_worst_case_cap_hit_every_call": judge_worst,
            "per_judge_selfconsistency_10q_optional": selfcons,
            "A_all_queries": a_tok,
            "C_all_queries": c_tok,
            "caching_probe_per_model": probe,
        },
        "per_model": {
            "gpt-oss-120b (judge+probe)": per_judge + probe,
            "qwen3.8-27b (judge)": per_judge,
            "gpt-oss-20b (A+C)": a_tok + c_tok + probe,
        },
        "budgets": {"judge": judge_budget, "rerank_model": rerank_budget},
    }
    plan["days"] = {
        "gpt-oss-120b": math.ceil(plan["per_model"]["gpt-oss-120b (judge+probe)"] / judge_budget),
        "qwen3.8-27b": math.ceil(plan["per_model"]["qwen3.8-27b (judge)"] / judge_budget),
        "gpt-oss-120b_worst_case": math.ceil((judge_worst + judge_cal + probe) / judge_budget),
        "gpt-oss-20b": math.ceil(plan["per_model"]["gpt-oss-20b (A+C)"] / rerank_budget),
    }
    call_tok = mean(jt) + EST_COMPLETION["judge"]
    plan["tpm_bound_minutes_per_judge_main"] = n * call_tok / TPM_LIMIT
    plan["rpm_bound_minutes_per_judge_main"] = n * MIN_REQUEST_INTERVAL_S / 60
    return plan


def stage_plan(args: argparse.Namespace) -> int:
    """Token/day plan from the REAL built prompts (tiktoken o200k_base proxy, or heuristic)."""
    from src.agents.reranker import _SYSTEM, _format_candidates

    cases = usable_cases(args.limit, Path(args.cand))
    pools = [load_candidates(c["id"], Path(args.cand))["items"] for c in cases]
    jt = [
        count_tokens(p.system + p.user)
        for p in (build_judge_prompt(c, it, SEED) for c, it in zip(cases, pools, strict=True))
    ]
    ct = [
        sum(count_tokens(x) for x in build_compact_prompt(c["text"], it))
        for c, it in zip(cases, pools, strict=True)
    ]
    at = [
        count_tokens(_SYSTEM)
        + count_tokens(
            f'Query: "{c["text"]}"\n\nProducts:\n{_format_candidates(it)}\n\nSelect the best 5 products.'
        )
        for c, it in zip(cases, pools, strict=True)
    ]
    cal = [
        count_tokens(p.system + p.user)
        for p in (build_judge_prompt(x["case"], x["candidates"], SEED) for x in calibration_cases())
    ]
    plan = build_plan(jt, cal, at, ct, args.judge_budget, args.rerank_budget)
    plan["tokenizer"] = "tiktoken o200k_base" if count_tokens("a b") == 2 else "heuristic"
    (Path(args.out) / "plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage",
        required=True,
        choices=["candidates", "judge", "variants", "factors", "report", "plan"],
    )
    ap.add_argument(
        "--reps", type=int, default=3, help="variants: runs per query of the stochastic arms A, A2"
    )
    ap.add_argument("--mock", action="store_true", help="rule-based fakes, no network")
    ap.add_argument(
        "--live", action="store_true", help="allow live Groq (also needs STAGE2_LIVE_OK=1)"
    )
    ap.add_argument("--resume", action="store_true", help="accepted for clarity; always resumes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument("--cand", default=str(CANDIDATE_DIR))
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument(
        "--all-stores", action="store_true", help="candidates: no public-store restriction"
    )
    ap.add_argument(
        "--sparse-only", action="store_true", help="candidates: BM25 only (equality proof)"
    )
    ap.add_argument("--only", default=None, help="candidates: comma-separated case ids to refresh")
    ap.add_argument("--export-packets", action="store_true")
    ap.add_argument("--import-claude", action="store_true")
    ap.add_argument("--allow-partial-gate", action="store_true")
    ap.add_argument("--judge-budget", type=int, default=JUDGE_DAILY_TOKEN_BUDGET)
    ap.add_argument("--rerank-budget", type=int, default=RERANKER_DAILY_TOKEN_BUDGET)
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.stage == "candidates":
        from evals.retrieval import candidates

        if not args.index_dir:
            print("--index-dir is required for --stage candidates")
            return 2
        recs = candidates.generate(
            Path(args.index_dir),
            Path(args.config),
            Path(args.cand),
            args.limit,
            not args.all_stores,
            args.sparse_only,
            set(args.only.split(",")) if args.only else None,
        )
        print(json.dumps(candidates.summarize(recs), indent=1))
        return 0
    stages = {
        "judge": stage_judge,
        "variants": stage_variants,
        "factors": stage_factors,
        "report": stage_report,
        "plan": stage_plan,
    }
    try:
        return stages[args.stage](args)
    except Exception as exc:  # noqa: BLE001 - report guard/budget/429 stops as a clean refusal
        print(f"STOPPED: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
