"""Stage driver for the V4 retrieval eval.

    python -m evals.retrieval.run --stage candidates --index-dir <dir> [--limit N]
    python -m evals.retrieval.run --stage plan                       # token/day estimates
    python -m evals.retrieval.run --stage judge --mock [--limit N]   # calibration gate + labels
    python -m evals.retrieval.run --stage judge --export-packets     # Claude packets (no LLM)
    python -m evals.retrieval.run --stage judge --import-claude      # import Claude results
    python -m evals.retrieval.run --stage variants --mock [--limit N] [--reps 2]  # A A2 B C
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
    DEFAULT_REPS,
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
    resolve_pools_dir,
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

    Labels are returned as-is; `calibration_ok` reports exact-vs-hand counts and fails closed
    on missing labels, the pairwise-kappa gate decides whether the main run proceeds."""
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
    """Per-judge exact-vs-hand-labels count, REPORTED only (GG ruling 2026-11-19: the stage-1
    ">= 18/20 exact" per-judge criterion was our own addition; the architect's gate is the
    pairwise kappa >= 0.4 in `ag.calibration_gate`). The one blocker kept here is fail-closed:
    a judge that did not label every calibration item (failed call or unparseable output)."""
    gold = _cal_gold()
    msgs, ok = [], True
    for judge, pred in labels.items():
        s = score_calibration(pred, gold)
        missing = sum(1 for k in gold if k not in pred)
        note = f", MISSING {missing} labels" if missing else ""
        msgs.append(
            f"{judge}: exact {s['exact']}/{s['n']}, off-by-two {s['off_by_two']}"
            f" (reported, not a gate){note}"
        )
        ok &= missing == 0
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
        print("REFUSING to continue: a judge did not label every calibration item.")
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
    """A, A2, B, C on the pools regenerated on master (`--pools`). A and A2 run `--reps` times
    (default 2: rep 1 plus one repeat) per query (stability); the order is rep-major so rep 1 of every arm finishes first (the main comparison needs only
    that; later reps can wait for the next day's token budget)."""
    out_dir = Path(args.out)
    cand = resolve_pools_dir(Path(args.cand), args.pools)  # master-regenerated pools
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
    out_dir, cand = Path(args.out), Path(args.cand)  # cand = frozen pools (judged superset)
    pools_dir = resolve_pools_dir(cand, args.pools)
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
                pool = _pool_for(c, pools_dir)  # what the main comparison (rep 1) ran on
                base_pool, new_pool = pool, transform(pool)
                if factor == "F1" and pools_dir != cand:
                    # Regenerated pools already carry #129 (sets demoted before rerank): the
                    # treated arm is the main run, the "sets not demoted" arm is the frozen pool.
                    base_pool, new_pool = _pool_for(c, cand), pool
                lab = {a: x["label"] for a, x in _pool_labels_for(agg, c["id"]).items()}
                unlabelled = {str(it["article_id"]) for it in base_pool + new_pool} - set(lab)
                if unlabelled:
                    raise RuntimeError(
                        f"{c['id']}: {len(unlabelled)} pool items have no judge label "
                        f"(judge the frozen pool, a superset of the regenerated one): "
                        f"{sorted(unlabelled)[:3]}"
                    )
                main_ids = [it["article_id"] for it in pool]
                arm_rows = []
                for arm_pool in (base_pool, new_pool):
                    if [it["article_id"] for it in arm_pool] == main_ids:
                        arm_rows.append({**rep1[v][c["id"]], "reused": True})
                    else:
                        row = _with_cards(_run_arm(v, c, arm_pool, llms.get(v), 1), c, arm_pool)
                        arm_rows.append({**row, "reused": False})
                base, treated = arm_rows
                if factor == "F1":
                    for k, val in pool_loss(base_pool, new_pool, lab).items():
                        loss[k] += val
                    for r, p in ((base, base_pool), (treated, new_pool)):
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
EST_COMPLETION = {"judge": 400, "calibration": 400, "A": 150, "A2": 250, "C": 100}
# A2 completion: visible JSON ~60 tokens + hidden reasoning 96-148 measured in prod logs, +margin.
TPM_LIMIT = 8_000  # Groq free tier per model (architect brief)
F1_QUERIES, F2_QUERIES = (
    8,
    12,
)  # flagged fixtures (factors.is_set_demotion_query / is_gender_unspecified)
FACTOR_ARMS = 2  # factors run on the two best arms; both LLM arms is the upper bound


def schedule(
    tasks: list[tuple[str, int, float]], budget: int, per_call_cap: float | None = None
) -> list[dict[str, Any]]:
    """Greedy day-by-day packing of (name, n_calls, tokens_per_call) tasks, in order, under a
    daily token budget. A call is never split across days; `per_call_cap` (the runner's worst-case
    pre-check, prompt + completion cap) must also fit in the remaining day budget, which is how
    the live run will actually stop. Returns [{"day", "tokens", "calls", "work": {name: calls}}]."""
    days: list[dict[str, Any]] = []
    cur: dict[str, Any] = {"day": 1, "tokens": 0.0, "calls": 0, "work": {}}
    for name, n, each in tasks:
        for _ in range(n):
            need = max(each, per_call_cap or 0.0)
            if cur["tokens"] + need > budget and cur["calls"]:
                days.append(cur)
                cur = {"day": cur["day"] + 1, "tokens": 0.0, "calls": 0, "work": {}}
            cur["tokens"] += each
            cur["calls"] += 1
            cur["work"][name] = cur["work"].get(name, 0) + 1
    if cur["calls"]:
        days.append(cur)
    return days


def build_plan(
    jt: list[int],
    cal: list[int],
    a_prompt: list[int],
    c_prompt: list[int],
    judge_budget: int = JUDGE_DAILY_TOKEN_BUDGET,
    rerank_budget: int = RERANKER_DAILY_TOKEN_BUDGET,
    reps: int = DEFAULT_REPS,
    relabel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-model token totals, day-by-day schedules and wall-clock under the global 1 request /
    8 s limit. `jt`/`cal` cover every query / calibration call; `a_prompt` and `c_prompt` only
    the pools with more than 5 candidates (smaller pools skip the reranker). `relabel` is the
    consensus-relabel plan (`relabel.plan`) or None (then NOT included)."""
    import math

    def mean(x: list[int]) -> float:
        return sum(x) / len(x)

    n, n_llm = len(jt), len(a_prompt)
    judge_cal = len(cal) * (mean(cal) + EST_COMPLETION["calibration"])
    judge_each = mean(jt) + EST_COMPLETION["judge"]
    a_each = mean(a_prompt) + EST_COMPLETION["A"]
    a2_each = mean(a_prompt) + EST_COMPLETION["A2"]
    c_each = mean(c_prompt) + EST_COMPLETION["C"]
    probe = 3 * a_each
    f_calls = (F1_QUERIES + F2_QUERIES) * FACTOR_ARMS  # upper bound: every pool changes
    r_tok = relabel["expected_tokens_per_judge"] if relabel else 0
    r_calls = relabel["n_calls_per_judge"] if relabel else 0
    r_each = r_tok / r_calls if r_calls else 0
    plan: dict[str, Any] = {
        "n_queries_with_candidates": n,
        "n_pools_over_5_candidates": n_llm,
        "reps": reps,
        "mean_prompt_tokens": {
            "judge": mean(jt),
            "calibration": mean(cal),
            "A_A2": mean(a_prompt),
            "C": mean(c_prompt),
        },
        "est_completion_tokens_ASSUMED": EST_COMPLETION,
        "relabel_included": bool(relabel),
        "tokens": {
            "per_judge_calibration": judge_cal,
            "per_judge_main_expected": n * judge_each,
            "per_judge_main_worst_case_cap_hit_every_call": n * (mean(jt) + JUDGE_MAX_TOKENS),
            "per_judge_relabel_expected": r_tok,
            "A_all_reps": n_llm * reps * a_each,
            "A2_all_reps": n_llm * reps * a2_each,
            "C_one_rep": n_llm * c_each,
            "factors_upper_bound": f_calls * a2_each,
            "caching_probe_per_model": probe,
        },
    }
    plan["per_model"] = {
        "gpt-oss-120b (judge + relabel + probe)": judge_cal + n * judge_each + r_tok + probe,
        "qwen3.8-27b (judge + relabel)": judge_cal + n * judge_each + r_tok,
        "gpt-oss-20b (A, A2, C, factors, probe)": (
            plan["tokens"]["A_all_reps"]
            + plan["tokens"]["A2_all_reps"]
            + plan["tokens"]["C_one_rep"]
            + plan["tokens"]["factors_upper_bound"]
            + probe
        ),
    }
    judge_tasks: list[tuple[str, int, float]] = [
        ("calibration", len(cal), judge_cal / len(cal)),
        ("relabel_chunks", r_calls, r_each),
        ("judge_queries", n, judge_each),
    ]
    rerank_tasks: list[tuple[str, int, float]] = [
        ("A_rep1", n_llm, a_each),
        ("A2_rep1", n_llm, a2_each),
        ("C_rep1", n_llm, c_each),
        ("factors", f_calls, a2_each),
    ] + [
        (f"{arm}_rep{r}", n_llm, e)
        for r in range(2, reps + 1)
        for arm, e in (("A", a_each), ("A2", a2_each))
    ]
    plan["schedule_per_judge_model"] = schedule(
        judge_tasks, judge_budget, JUDGE_MAX_TOKENS + mean(jt)
    )
    plan["schedule_gpt_oss_20b"] = schedule(rerank_tasks, rerank_budget, 600 + mean(a_prompt))
    plan["days"] = {
        "each_groq_judge": len(plan["schedule_per_judge_model"]),
        "gpt-oss-20b": len(plan["schedule_gpt_oss_20b"]),
        "gpt-oss-20b_without_repeats_rep1_only": math.ceil(
            (n_llm * (a_each + a2_each + c_each) + f_calls * a2_each) / rerank_budget
        ),
    }
    both_judges_calls = 2 * (len(cal) + r_calls + n)
    plan["wall_clock_global_1_req_per_8s"] = {
        "judge_requests_total_both_models": both_judges_calls,
        "judge_hours_rate_limit_only": both_judges_calls * MIN_REQUEST_INTERVAL_S / 3600,
        "rerank_requests_total": n_llm * reps * 2 + n_llm + f_calls,
        "rerank_hours_rate_limit_only": (n_llm * reps * 2 + n_llm + f_calls)
        * MIN_REQUEST_INTERVAL_S
        / 3600,
        "tpm_note": "8K TPM per model: ~4 judge calls/min, ~1-2 relabel calls/min; the TPM "
        "limit, not the 8 s spacing, binds the judge runs",
    }
    return plan


def stage_plan(args: argparse.Namespace) -> int:
    """Token/day plan from the REAL built prompts (tiktoken o200k_base proxy, or heuristic)."""
    from src.agents.reranker import _SYSTEM, _format_candidates

    cases = usable_cases(args.limit, Path(args.cand))
    pools = [load_candidates(c["id"], Path(args.cand))["items"] for c in cases]  # judged: frozen
    jt = [
        count_tokens(p.system + p.user)
        for p in (build_judge_prompt(c, it, SEED) for c, it in zip(cases, pools, strict=True))
    ]
    rr_dir = resolve_pools_dir(Path(args.cand), args.pools)  # reranker arms: regenerated pools
    rr_pools = [load_candidates(c["id"], rr_dir)["items"] for c in cases]
    big = [(c, it) for c, it in zip(cases, rr_pools, strict=True) if len(it) > 5]
    ct = [sum(count_tokens(x) for x in build_compact_prompt(c["text"], it)) for c, it in big]
    at = [
        count_tokens(_SYSTEM)
        + count_tokens(
            f'Query: "{c["text"]}"\n\nProducts:\n{_format_candidates(it)}\n\nSelect the best 5 products.'
        )
        for c, it in big
    ]
    cal = [
        count_tokens(p.system + p.user)
        for p in (build_judge_prompt(x["case"], x["candidates"], SEED) for x in calibration_cases())
    ]
    relabel_path = Path(args.out) / "relabel" / "plan.json"
    relabel = json.loads(relabel_path.read_text("utf-8")) if relabel_path.exists() else None
    plan = build_plan(jt, cal, at, ct, args.judge_budget, args.rerank_budget, args.reps, relabel)
    plan["tokenizer"] = "tiktoken o200k_base" if count_tokens("a b") == 2 else "heuristic"
    (Path(args.out) / "plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan, indent=1))
    return 0


def candidate_provenance(index_dir: Path, config_path: Path, public_only: bool) -> dict[str, Any]:
    """Provenance of a regenerated pool directory: code commit, catalogue hash, config, scope."""
    import hashlib
    import subprocess

    head = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    cat = index_dir / "catalogue.parquet"
    digest = hashlib.sha256(cat.read_bytes()).hexdigest() if cat.exists() else None
    return {
        "role": "main-comparison pools, regenerated on master after #129 (sets demoted before "
        "rerank); the frozen candidates/ pools are the F1 'sets not demoted' arm",
        "code_commit": head or "unknown",
        "index": f"{index_dir.name} (read-only)",
        "catalogue_parquet_sha256": digest,
        "config": config_path.name,
        "restriction": "public_stores.json pushed into BM25+FAISS fetch" if public_only else "none",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage",
        required=True,
        choices=["candidates", "judge", "variants", "factors", "report", "plan"],
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help="variants: runs per query of the stochastic arms A, A2",
    )
    ap.add_argument("--mock", action="store_true", help="rule-based fakes, no network")
    ap.add_argument(
        "--live", action="store_true", help="allow live Groq (also needs STAGE2_LIVE_OK=1)"
    )
    ap.add_argument("--resume", action="store_true", help="accepted for clarity; always resumes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument(
        "--cand",
        default=str(CANDIDATE_DIR),
        help="frozen pools: judged by every judge, and the F1 'sets not demoted' arm",
    )
    ap.add_argument(
        "--pools",
        default=None,
        help="pools the A/A2/B/C comparison runs on (default: candidates_master/ when --cand is "
        "the default frozen dir and it exists, else --cand)",
    )
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
        summary = candidates.summarize(recs)
        out_cand = Path(args.cand)
        if not args.only and not args.limit and out_cand.resolve() != CANDIDATE_DIR.resolve():
            # A full regeneration into a directory other than the frozen one: record provenance
            # and how it differs from the frozen pools (the frozen _summary.json is never touched).
            summary = candidates.summary_with_provenance(
                recs,
                candidate_provenance(Path(args.index_dir), Path(args.config), not args.all_stores),
                candidates.compare_pools(CANDIDATE_DIR, out_cand),
            )
            (out_cand / "_summary.json").write_text(
                json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
            )
        print(json.dumps(summary, indent=1))
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
