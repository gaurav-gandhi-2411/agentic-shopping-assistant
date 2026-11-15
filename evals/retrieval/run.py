"""Stage driver for the V4 retrieval eval.

    python -m evals.retrieval.run --stage candidates --index-dir <dir> [--limit N]
    python -m evals.retrieval.run --stage plan                       # token/day estimates
    python -m evals.retrieval.run --stage judge --mock [--limit N]   # calibration gate + labels
    python -m evals.retrieval.run --stage judge --export-packets     # Claude packets (no LLM)
    python -m evals.retrieval.run --stage judge --import-claude      # import Claude results
    python -m evals.retrieval.run --stage variants --mock [--limit N]
    python -m evals.retrieval.run --stage report [--mock]

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
from evals.retrieval import metrics as mt
from evals.retrieval.common import (
    CANDIDATE_DIR,
    JUDGE_DAILY_TOKEN_BUDGET,
    JUDGE_MODELS,
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
from evals.retrieval.variants import (
    C_MAX_TOKENS,
    build_compact_prompt,
    run_variant_a,
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
    return ModelSpec("rerank-20b", RERANKER_MODEL, {"reasoning_effort": "low"}, 150, budget, 150)


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
        self.last_meta: dict[str, Any] = {}

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        spec = dataclasses.replace(
            self.spec,
            name=f"{self.spec.name}-{self.variant}",
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
    """Deterministic fake reranker model (A: JSON, C: numbers); every 5th A call truncates."""

    def __init__(self) -> None:
        self.n = 0
        self.last_meta: dict[str, Any] = {}

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        self.n += 1
        compact = (system or "").startswith("You rank fashion products")
        if compact:
            out = "3,1,2,5,4,6,7,8"
        elif self.n % 5 == 0:
            out = '{"selected": [3, 1, 2, 5, 4], "reasoning": "elegant and on-bud'  # truncated
        else:
            out = '{"selected": [3, 1, 2, 5, 4], "reasoning": "best fit"}'
        cap = int(kw.get("max_tokens", 0))
        comp = cap if (not compact and self.n % 5 == 0) else count_tokens(out) + 60
        self.last_meta = {
            "finish_reason": "length" if comp >= cap else "stop",
            "usage": {
                "prompt_tokens": count_tokens((system or "") + prompt),
                "completion_tokens": comp,
            },
            "latency_ms": 400 + self.n,
        }
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
    cases = load_cases(args.limit)
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
        labels, problems = import_results(out_dir / "claude_packets")
        cal = {k: v for k, v in labels["claude"].items() if k.startswith("cal")}
        main = {k: v for k, v in labels["claude"].items() if not k.startswith("cal")}
        (out_dir / "claude_packets" / "labels_cal_claude.json").write_text(
            json.dumps(cal, indent=1)
        )
        (out_dir / "labels_claude.json").write_text(json.dumps(main, indent=1))
        print(
            f"imported {len(main)} main + {len(cal)} calibration Claude labels; "
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


def stage_variants(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    cases = load_cases(args.limit)
    spec = reranker_spec(args.rerank_budget)
    if args.mock:
        llm_a: Any = MockRerankLLM()
        llm_c: Any = MockRerankLLM()
    else:
        runner = _runner(False, args.live, out_dir, None)
        llm_a, llm_c = GroqLLM(runner, spec, "A"), GroqLLM(runner, spec, "C")
    results: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for c in cases:
        items = load_candidates(c["id"], Path(args.cand))["items"]
        for llm in (llm_a, llm_c):
            if isinstance(llm, GroqLLM):
                llm.query_id = c["id"]
        results["A"].append({"query_id": c["id"], **run_variant_a(c["text"], items, llm_a)})
        results["B"].append({"query_id": c["id"], **run_variant_b(items)})
        results["C"].append({"query_id": c["id"], **run_variant_c(c["text"], items, llm_c)})
    (out_dir / "variants.json").write_text(json.dumps(results, indent=1))
    print(f"variants A/B/C over {len(cases)} queries written to {out_dir / 'variants.json'}")
    return 0


def _load_aggregate(out_dir: Path, judges: list[str]) -> tuple[dict, dict]:
    labels = {
        j: json.loads((out_dir / f"labels_{j}.json").read_text())
        for j in judges
        if (out_dir / f"labels_{j}.json").exists()
    }
    return labels, ag.aggregate(labels)


def build_report(
    cases: list[dict],
    variants: dict[str, list[dict]],
    agg: dict[str, dict],
) -> dict[str, Any]:
    """Metrics per variant (with bootstrap CIs), paired differences, tie-excluded sensitivity."""
    per: dict[str, dict[str, list[float]]] = {}
    sens: dict[str, dict[str, list[float]]] = {}
    for v, rows in variants.items():
        per[v] = {"ndcg5": [], "p5_ge1": [], "p5_ge2": [], "p5_ge1_5": []}
        sens[v] = {"ndcg5": [], "p5_ge1": []}
        for row in rows:
            qid = row["query_id"]
            pool_ids = [k.split(":", 1)[1] for k in agg if k.startswith(qid + ":")]
            lab = {a: agg[f"{qid}:{a}"] for a in pool_ids}
            shown = [lab[a]["label"] for a in row["final_ids"] if a in lab]
            pool = [x["label"] for x in lab.values()]
            per[v]["ndcg5"].append(mt.ndcg_at_k(shown, pool))
            per[v]["p5_ge1"].append(mt.precision_at_k(shown, 1.0))
            per[v]["p5_ge2"].append(mt.precision_at_k(shown, 2.0))
            per[v]["p5_ge1_5"].append(mt.precision_at_k(shown, 1.5))
            keep = {a for a, x in lab.items() if not x["tie"]}
            s_shown = [lab[a]["label"] for a in row["final_ids"] if a in keep]
            s_pool = [lab[a]["label"] for a in keep]
            sens[v]["ndcg5"].append(mt.ndcg_at_k(s_shown, s_pool))
            sens[v]["p5_ge1"].append(
                sum(1 for x in s_shown if x >= 1) / len(s_shown) if s_shown else float("nan")
            )
    summary = {v: {m: mt.bootstrap_mean_ci(vals) for m, vals in d.items()} for v, d in per.items()}
    sens_summary = {
        v: {m: mt.bootstrap_mean_ci(vals) for m, vals in d.items()} for v, d in sens.items()
    }
    pairs = {}
    for a, b in (("A", "B"), ("C", "B"), ("C", "A")):
        if a in per and b in per:
            pairs[f"{a}-{b}"] = {m: mt.paired_difference(per[a][m], per[b][m]) for m in per[a]}
    usage = {v: mt.usage_summary(rows) for v, rows in variants.items()}
    outcomes = {
        v: {
            o: sum(1 for r in rows if r.get("outcome") == o)
            for o in {r.get("outcome") for r in rows}
        }
        for v, rows in variants.items()
    }
    caps = {v: sum(1 for r in rows if r.get("hit_cap")) for v, rows in variants.items()}
    return {
        "n_queries": len(cases),
        "metrics": summary,
        "sensitivity_excluding_ties": sens_summary,
        "paired": pairs,
        "usage": usage,
        "outcomes": outcomes,
        "hit_cap": caps,
        "n_tied_items": sum(1 for x in agg.values() if x["tie"]),
        "n_items": len(agg),
    }


def stage_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    variants = json.loads((out_dir / "variants.json").read_text())
    labels, agg = _load_aggregate(out_dir, ["gpt-oss-120b", "qwen3.8-27b", "claude"])
    if not agg:
        print("no judge labels found; run --stage judge first")
        return 2
    rep = build_report(load_cases(args.limit), variants, agg)
    rep["agreement"] = ag.pairwise_report(labels)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_queries", "metrics", "paired", "usage")}, indent=1))
    print("LLM-consensus labels (three model families), not human ground truth.")
    return 0


def stage_plan(args: argparse.Namespace) -> int:
    """Token estimates from the real built prompts (tiktoken o200k_base proxy, or heuristic)."""
    cases = load_cases(args.limit)
    jt = [
        count_tokens(p.system + p.user)
        for p in (
            build_judge_prompt(c, load_candidates(c["id"], Path(args.cand))["items"], SEED)
            for c in cases
        )
    ]
    ct = [
        sum(
            count_tokens(x)
            for x in build_compact_prompt(
                c["text"], load_candidates(c["id"], Path(args.cand))["items"]
            )
        )
        for c in cases
    ]
    cal = [
        count_tokens(p.system + p.user)
        for p in (build_judge_prompt(x["case"], x["candidates"], SEED) for x in calibration_cases())
    ]
    est = {
        "judge_prompt_mean": sum(jt) / len(jt),
        "judge_prompt_max": max(jt),
        "compact_prompt_mean": sum(ct) / len(ct),
        "calibration_prompt_mean": sum(cal) / len(cal),
        "n_cases": len(cases),
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "c_max_tokens": C_MAX_TOKENS,
    }
    print(json.dumps(est, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage", required=True, choices=["candidates", "judge", "variants", "report", "plan"]
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
        )
        print(json.dumps(candidates.summarize(recs), indent=1))
        return 0
    stages = {
        "judge": stage_judge,
        "variants": stage_variants,
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
