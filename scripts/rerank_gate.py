"""Pre-deploy gate for the production reranker (stage-2 variant C promoted to production).

Runs the PRODUCTION code (`src.agents.reranker.rerank`, through the real `GroqClient.complete`)
on the stage-2 frozen pools, scores it on the consensus labels and compares it with the eval's
variant C rows. Gate criteria (architect, 2026-11-21):

  1. nDCG@5 on consensus labels: lower bound of the paired bootstrap 95% CI of
     (production - eval C) >= -0.03 (10,000 resamples, seed 42, percentile CI over queries);
  2. clean outcome rate >= 90% over the primary queries (see CLEAN_OUTCOMES);
  3. reported, not gated: tokens/turn, latency p50/p95, n, excluded queries.
Then (optional, `--chips-pools`) the chip queries: outcomes/tokens/latency and the shown titles
for review; they have no consensus labels, so they are never scored.

Live safety (the stage-2 rules, kept): refuses live unless `--live` AND RERANK_GATE_LIVE_OK=1
(and today's UTC date == `--expect-utc-day` when given); at most one request per 8 s; a token
cap (default 60,000, prompt estimate + completion cap reserved before every call, counting this
gate's own ledger for today so a resume cannot overspend); stops at once on a busy signal
(5xx/timeout/transport) or the daily quota, and after 3 consecutive 429s; every real response is
cached (resume = rerun the same command, nothing is requested twice). GROQ_API_KEY is read by
name only (from the process env, else from the `--env-file` line of that exact name); the rest of
the .env is never loaded and no value is printed.

Offline modes (no network, no key): `--mock` (deterministic fake model) and
`--replay-eval-cache PATH` (re-serves the stage-2 eval's variant C replies from its cache.jsonl:
the prompt is byte-identical, so production's parse/validate/fill logic is exercised on the real
model replies of the eval; expect production ~ eval C up to the strict-parser differences).

    python scripts/rerank_gate.py --replay-eval-cache <v4e>/cache.jsonl --out <scratch>   # offline
    RERANK_GATE_LIVE_OK=1 python scripts/rerank_gate.py --live --expect-utc-day 2026-11-22
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as `python scripts/rerank_gate.py` from any cwd
    sys.path.insert(0, str(REPO))

from src.agents import reranker as rr  # noqa: E402
from src.llm.client import LLMCallError, LLMCompletion  # noqa: E402

STAGE2 = Path("C:/Users/gaura/ml-projects/agentic-shopping-assistant-wt-work")
DEFAULT_POOLS = STAGE2 / "evals/retrieval/candidates_master"
DEFAULT_RUN_DIR = STAGE2 / "reports/wave1b/v4e"
DEFAULT_OUT = REPO / "reports" / "rerank_gate"  # reports/ is gitignored
JUDGES = ("gpt-oss-120b", "qwen3.8-27b", "claude")  # consensus = majority, else median

TOP_K = 5
N_BOOT = 10_000
SEED = 42
MARGIN = -0.03
CLEAN_RATE_MIN = 0.90
# Outcomes that are a valid answer (the model answered; nothing failed). `ok` alone is the strict
# reading; `partial` (1-4 valid picks, rest filled) and `empty_selection` (model: nothing fits,
# finish=stop) are valid answers too. The gate reports both rates; see main()'s verdict.
CLEAN_OUTCOMES = ("ok", "partial", "empty_selection")
MIN_INTERVAL_S = 8.0
TOKEN_CAP = 60_000
MAX_CONSECUTIVE_429 = 3
LIVE_ENV_FLAG = "RERANK_GATE_LIVE_OK"
KEY_NAME = "GROQ_API_KEY"


class GateStop(RuntimeError):
    """Stop the whole run (busy signal, quota, repeated 429, budget, fatal): never a row."""


class CompleteLike(Protocol):
    def complete(self, messages: list[dict], **kw: Any) -> LLMCompletion: ...


# --- metrics (same definitions as evals/retrieval/metrics.py, vendored: that package lives on
# --- feat/v4-retrieval-eval, not on master) ------------------------------------------------

def dcg(labels: Sequence[float], k: int = TOP_K) -> float:
    return sum((2.0**lab - 1.0) / math.log2(i + 2) for i, lab in enumerate(labels[:k]))


def ndcg_at_k(shown: Sequence[float], pool: Sequence[float], k: int = TOP_K) -> float:
    """nDCG@k vs the ideal ordering of ALL labelled candidates; NaN when the ideal is 0."""
    ideal = dcg(sorted(pool, reverse=True), k)
    return float("nan") if ideal == 0 else dcg(shown, k) / ideal


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile 95% CI of the mean over query resamples (NaNs dropped)."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_label(labels: Sequence[int]) -> float:
    """Strict-majority label, else the median (the eval's `aggregate_item`)."""
    counts = Counter(labels)
    top_lab, top_n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return float(top_lab) if top_n * 2 > len(labels) else float(statistics.median(labels))


def load_consensus(run_dir: Path) -> dict[str, dict[str, float]]:
    """qid -> {article_id: consensus label} from labels_<judge>.json (keys 'qid:article_id')."""
    per_judge = [
        json.loads((run_dir / f"labels_{j}.json").read_text("utf-8"))
        for j in JUDGES
        if (run_dir / f"labels_{j}.json").exists()
    ]
    if len(per_judge) < len(JUDGES):
        raise SystemExit(f"need labels_<judge>.json for {JUDGES} in {run_dir}; found {len(per_judge)}")
    keys = sorted({k for lab in per_judge for k in lab})
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        qid, _, aid = k.partition(":")
        out.setdefault(qid, {})[aid] = aggregate_label([lab[k] for lab in per_judge if k in lab])
    return out


def query_ndcg(final_ids: Sequence[str], labels: Mapping[str, float]) -> float:
    shown = [labels[a] for a in final_ids if a in labels]
    return ndcg_at_k(shown, list(labels.values()))


# --- token budget, throttle, cache, ledger ---------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic (3 chars/token) prompt-size estimate for the pre-call reserve."""
    return len(text) // 3 + 1


def request_key(messages: Sequence[Mapping[str, str]], kw: Mapping[str, Any]) -> str:
    stable = {k: v for k, v in kw.items() if k not in ("timeout_s", "retry")}  # per-call, not identity
    blob = json.dumps({"m": list(messages), "kw": stable}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


class JsonlStore:
    """Append-only JSONL file; `cache` keeps the latest record per key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        if path.exists():
            self.records = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]

    def append(self, rec: dict[str, Any]) -> None:
        self.records.append(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dataclass
class GateLLM:
    """The `complete()` the production reranker is handed: cache + throttle + token cap + stop
    policy around `inner` (the real GroqClient live, a fake offline). Typed errors from `inner`
    never reach rerank(): every provider failure either is retried (429 < 3 in a row) or ends the
    run with GateStop, so no outcome row is a swallowed failure."""

    inner: CompleteLike | None
    cache: JsonlStore
    ledger: JsonlStore
    token_cap: int = TOKEN_CAP
    min_interval_s: float = MIN_INTERVAL_S
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)  # noqa: E731
    query_id: str = ""
    last_cached: bool = False
    _last_request: float | None = None
    _consecutive_429: int = 0

    def tokens_today(self) -> int:
        today = self.now().date().isoformat()
        return sum(
            int(r.get("total_tokens") or 0)
            for r in self.ledger.records
            if str(r.get("ts", ""))[:10] == today
        )

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval_s - (self.clock() - self._last_request)
            if wait > 0:
                self.sleep(wait)
        self._last_request = self.clock()

    def complete(self, messages: list[dict], **kw: Any) -> LLMCompletion:
        key = request_key(messages, kw)
        for rec in reversed(self.cache.records):
            if rec["key"] == key and rec.get("query_id") == self.query_id:
                self.last_cached = True
                return LLMCompletion(
                    rec["text"], rec["finish_reason"], rec["prompt_tokens"],
                    rec["completion_tokens"], rec["reasoning_tokens"], rec["latency_ms"],
                    rec.get("model", ""),
                )
        self.last_cached = False
        if self.inner is None:
            raise GateStop(f"{self.query_id}: not in the replay/mock source")
        est = sum(estimate_tokens(m["content"]) for m in messages) + int(kw["max_tokens"])
        used = self.tokens_today()
        if used + est > self.token_cap:
            raise GateStop(
                f"token cap: {used} used today + ~{est} for this call > {self.token_cap}; "
                "rerun the same command later (the cache resumes)"
            )
        while True:
            self._throttle()
            try:
                # retry=False: the gate owns pacing and the 429 policy; one request per attempt
                reply = self.inner.complete(messages, **{**kw, "retry": False})
            except LLMCallError as exc:
                self.ledger.append({"ts": self.now().isoformat(), "event": exc.kind,
                                    "query_id": self.query_id, "status": exc.status,
                                    "total_tokens": 0})
                if exc.kind == "rate_limit":
                    self._consecutive_429 += 1
                    if self._consecutive_429 >= MAX_CONSECUTIVE_429:
                        raise GateStop(f"{MAX_CONSECUTIVE_429} consecutive 429s") from exc
                    self.sleep(min(max(exc.retry_after or 0.0, 4.0), 60.0))
                    continue
                # daily_quota, server/timeout/transport (busy), fatal, malformed: stop the run
                raise GateStop(f"{exc.kind} (status={exc.status}): stopping") from exc
            self._consecutive_429 = 0
            total = (reply.prompt_tokens or 0) + (reply.completion_tokens or 0)
            self.ledger.append({
                "ts": self.now().isoformat(), "event": "ok", "query_id": self.query_id,
                "finish_reason": reply.finish_reason, "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "reasoning_tokens": reply.reasoning_tokens, "total_tokens": total,
                "latency_ms": reply.latency_ms,
            })
            self.cache.append({
                "key": key, "query_id": self.query_id, "text": reply.text,
                "finish_reason": reply.finish_reason, "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "reasoning_tokens": reply.reasoning_tokens, "latency_ms": reply.latency_ms,
                "model": reply.model,
            })
            return reply


# --- offline sources -------------------------------------------------------------------------

class MockComplete:
    """Deterministic fake model: answers 8 numbers (the first 8 candidates, reversed pairs) with
    usage figures shaped like the real ones. Zero network."""

    def __init__(self) -> None:
        self.n = 0

    def complete(self, messages: list[dict], **kw: Any) -> LLMCompletion:
        self.n += 1
        text = "2,1,4,3,6,5,8,7"
        return LLMCompletion(text, "stop", 650, 120, 100, 800 + self.n, "mock")


class ReplayEvalCache:
    """Serves the stage-2 eval's variant C replies (cache.jsonl, key
    'rerank-20b-C-r1|<qid>|<sha16(system+user)>') for the SAME prompts: a miss means the prompt
    is not byte-identical to eval C and aborts the replay."""

    def __init__(self, path: Path) -> None:
        self.recs: dict[str, dict[str, Any]] = {}
        for line in path.read_text("utf-8").splitlines():
            rec = json.loads(line)
            if rec["key"].startswith("rerank-20b-C-r1|"):
                self.recs[rec["key"]] = rec
        self.query_id = ""

    def complete(self, messages: list[dict], **kw: Any) -> LLMCompletion:
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        h = hashlib.sha256((system + user).encode("utf-8")).hexdigest()[:16]
        rec = self.recs.get(f"rerank-20b-C-r1|{self.query_id}|{h}")
        if rec is None:
            raise GateStop(f"{self.query_id}: prompt not found in the eval cache (not byte-equal?)")
        meta = rec.get("meta") or {}
        usage = meta.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return LLMCompletion(
            rec["text"], meta.get("finish_reason"), usage.get("prompt_tokens"),
            usage.get("completion_tokens"), details.get("reasoning_tokens"),
            int(meta.get("latency_ms") or 0), "replay",
        )


# --- the run ---------------------------------------------------------------------------------

class _EventCapture(logging.Handler):
    """Collects the reranker's one-line JSON `rerank` events (the logging contract under test)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ev = json.loads(record.getMessage())
        except ValueError:
            return
        if isinstance(ev, dict) and ev.get("event") == "rerank":
            self.events.append(ev)


def run_queries(
    cases: Sequence[Mapping[str, Any]], llm: GateLLM, source: Any, say: Callable[[str], None]
) -> list[dict[str, Any]]:
    """Production `rerank()` on each case's pool; one row per case that reached the LLM."""
    cap = _EventCapture()
    lg = logging.getLogger("src.agents.reranker")
    prev_level = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.INFO)
    rows: list[dict[str, Any]] = []
    try:
        for c in cases:
            pool = c["items"]
            llm.query_id = str(c["id"])
            if hasattr(source, "query_id"):
                source.query_id = str(c["id"])
            cap.events.clear()
            shown = rr.rerank(str(c["text"]), list(pool), cast(Any, llm), top_k=TOP_K)
            if len(cap.events) != 1:  # <= top_k candidates: no LLM call, no event
                say(f"  {c['id']}: no LLM call ({len(pool)} candidates)")
                continue
            ev = cap.events[0]
            rows.append({
                "query_id": c["id"], "text": c["text"], "final_ids": [str(i["article_id"]) for i in shown],
                "shown_titles": [str(i.get("display_name")) for i in shown],
                "outcome": ev["outcome"], "finish_reason": ev["finish_reason"],
                "prompt_tokens": ev["prompt_tokens"], "completion_tokens": ev["completion_tokens"],
                "reasoning_tokens": ev["reasoning_tokens"], "latency_ms": ev["latency_ms"],
                "n_candidates": ev["n_candidates"], "n_selected": ev["n_selected"],
                "cached": llm.last_cached,
            })
            say(f"  {c['id']}: {ev['outcome']} finish={ev['finish_reason']} "
                f"tok={ev['prompt_tokens']}+{ev['completion_tokens']} {ev['latency_ms']}ms"
                f"{' (cached)' if llm.last_cached else ''}")
    finally:
        lg.removeHandler(cap)
        lg.setLevel(prev_level)
    return rows


def pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def summarize(
    rows: Sequence[Mapping[str, Any]],
    eval_c: Mapping[str, Mapping[str, Any]],
    consensus: Mapping[str, Mapping[str, float]],
    n_skipped_pool: int,
) -> dict[str, Any]:
    """Gate numbers from production rows vs eval C rows (same queries, rep 1)."""
    prod_nd, c_nd, nan_qids = [], [], []
    for r in rows:
        lab = consensus.get(str(r["query_id"]), {})
        p = query_ndcg(r["final_ids"], lab)
        c = query_ndcg(eval_c[str(r["query_id"])]["final_ids"], lab)
        if math.isnan(p) or math.isnan(c):
            nan_qids.append(r["query_id"])
            continue
        prod_nd.append(p)
        c_nd.append(c)
    diffs = [p - c for p, c in zip(prod_nd, c_nd, strict=True)]
    mean, lo, hi = bootstrap_mean_ci(diffs)
    outcomes = Counter(str(r["outcome"]) for r in rows)
    n = len(rows)
    clean = sum(outcomes[o] for o in CLEAN_OUTCOMES) / n if n else float("nan")
    strict = outcomes["ok"] / n if n else float("nan")
    tok = [(r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows]
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    return {
        "n_llm_queries": n,
        "n_ndcg": len(diffs),
        "excluded": {"pool_le_top_k_or_empty": n_skipped_pool, "ndcg_undefined_all_zero_pool": nan_qids},
        "ndcg5_production": float(np.mean(prod_nd)) if prod_nd else float("nan"),
        "ndcg5_eval_c": float(np.mean(c_nd)) if c_nd else float("nan"),
        "diff_prod_minus_c": {"mean": mean, "ci_lo": lo, "ci_hi": hi, "margin": MARGIN,
                              "n_boot": N_BOOT, "seed": SEED},
        "outcomes": dict(outcomes),
        "length_hits": outcomes["length"],
        "clean_rate_valid": clean,
        "clean_rate_strict_ok": strict,
        "tokens_per_turn_mean": float(np.mean(tok)) if tok else float("nan"),
        "latency_ms_p50": pct(lat, 50),
        "latency_ms_p95": pct(lat, 95),
        "pass_ndcg": bool(not math.isnan(lo) and lo >= MARGIN),
        "pass_clean_valid": bool(clean >= CLEAN_RATE_MIN),
        "pass_clean_strict_ok": bool(strict >= CLEAN_RATE_MIN),
    }


def load_pool_file(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text("utf-8"))
    return {"id": d.get("query_id") or path.stem, "text": d["text"], "items": d["items"]}


def load_pools(pools_dir: Path) -> list[dict[str, Any]]:
    return [load_pool_file(p) for p in sorted(pools_dir.glob("*.json")) if not p.name.startswith("_")]


def load_eval_c(run_dir: Path) -> dict[str, dict[str, Any]]:
    vp = json.loads((run_dir / "variants_partial.json").read_text("utf-8"))
    return {str(r["query_id"]): r for r in vp["C"] if int(r.get("rep", 1)) == 1}


def render_md(res: Mapping[str, Any], chips: Sequence[Mapping[str, Any]]) -> str:
    d = res["diff_prod_minus_c"]
    lines = [
        "# Reranker gate (production rerank() vs eval variant C)",
        "",
        f"n LLM queries {res['n_llm_queries']}, n with defined nDCG {res['n_ndcg']}; "
        f"excluded: {json.dumps(res['excluded'])}",
        f"nDCG@5 production {res['ndcg5_production']:.3f} vs eval C {res['ndcg5_eval_c']:.3f}; "
        f"diff {d['mean']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] "
        f"(paired bootstrap, {d['n_boot']} resamples, seed {d['seed']}); need lower bound >= {d['margin']}"
        f" -> {'PASS' if res['pass_ndcg'] else 'FAIL'}",
        f"outcomes {json.dumps(res['outcomes'])}; length hits {res['length_hits']}",
        f"clean rate (ok+partial+empty_selection) {res['clean_rate_valid']:.3f} "
        f"-> {'PASS' if res['pass_clean_valid'] else 'FAIL'} (>= {CLEAN_RATE_MIN}); "
        f"strict ok-only rate {res['clean_rate_strict_ok']:.3f} "
        f"-> {'PASS' if res['pass_clean_strict_ok'] else 'FAIL'}",
        f"tokens/turn {res['tokens_per_turn_mean']:.0f}; latency p50 {res['latency_ms_p50']:.0f} ms, "
        f"p95 {res['latency_ms_p95']:.0f} ms",
        "Labels are LLM consensus (three model families), not human ground truth.",
    ]
    if chips:
        lines += ["", "## Chip queries (live, unlabelled: review the shown titles)"]
        for r in chips:
            lines.append(f"- {r['text']!r}: {r['outcome']}, tok {r['prompt_tokens']}+"
                         f"{r['completion_tokens']}, {r['latency_ms']} ms -> {r['shown_titles']}")
    return "\n".join(lines) + "\n"


# --- credentials and live construction -------------------------------------------------------

def read_named_key(env_file: Path, name: str = KEY_NAME) -> str | None:
    """Value of exactly `name` from a dotenv file, or None. No other line is parsed or kept."""
    if not env_file.exists():
        return None
    for line in env_file.read_text("utf-8").splitlines():
        m = re.match(rf"\s*{re.escape(name)}\s*=\s*(.*?)\s*$", line)
        if m:
            return m.group(1).strip("'\"") or None
    return None


def assert_live_allowed(
    live: bool, expect_day: str | None, environ: Mapping[str, str], now: dt.datetime
) -> None:
    """Live needs --live, RERANK_GATE_LIVE_OK=1 and (when given) the expected UTC date."""
    if not live:
        raise SystemExit("refusing a live call: pass --live (offline: --mock / --replay-eval-cache)")
    if environ.get(LIVE_ENV_FLAG) != "1":
        raise SystemExit(f"refusing a live call: set {LIVE_ENV_FLAG}=1")
    today = now.astimezone(dt.timezone.utc).date().isoformat()
    if expect_day and today != expect_day:
        raise SystemExit(f"refusing a live call: UTC day is {today}, expected {expect_day}")


def build_live_client(env_file: Path) -> CompleteLike:
    """The real production GroqClient (config.yaml llm section), key read by name only."""
    import yaml

    if not os.environ.get(KEY_NAME):
        key = read_named_key(env_file)
        if not key:
            raise SystemExit(f"{KEY_NAME} is not set and not found in {env_file}")
        os.environ[KEY_NAME] = key  # in-process only; never printed
    from src.llm.client import GroqClient

    cfg = yaml.safe_load((REPO / "config.yaml").read_text("utf-8"))
    return cast(CompleteLike, GroqClient(cfg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", default=str(DEFAULT_POOLS))
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="labels_*.json + variants_partial.json")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--chips-pools", default=None, help="dir of chip pool files (same format as --pools)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--replay-eval-cache", default=None)
    ap.add_argument("--expect-utc-day", default=None)
    ap.add_argument("--token-cap", type=int, default=TOKEN_CAP)
    ap.add_argument("--env-file", default=str(REPO / ".env"))
    ap.add_argument("--limit", type=int, default=None, help="first N queries only (smoke tests)")
    args = ap.parse_args(argv)

    modes = [bool(args.live), bool(args.mock), bool(args.replay_eval_cache)]
    if sum(modes) != 1:
        raise SystemExit("choose exactly one of --live, --mock, --replay-eval-cache")
    out = Path(args.out)
    mode = "live" if args.live else "mock" if args.mock else "replay"
    source: Any
    if args.live:
        assert_live_allowed(True, args.expect_utc_day, os.environ, dt.datetime.now(dt.timezone.utc))
        source = build_live_client(Path(args.env_file))
    elif args.mock:
        source = MockComplete()
    else:
        source = ReplayEvalCache(Path(args.replay_eval_cache))
    # separate cache/ledger per mode: a mock or replay run can never be mistaken for a live resume
    llm = GateLLM(source, JsonlStore(out / f"cache_{mode}.jsonl"), JsonlStore(out / f"ledger_{mode}.jsonl"),
                  token_cap=args.token_cap,
                  min_interval_s=MIN_INTERVAL_S if args.live else 0.0)  # pacing only matters live

    def say(msg: str) -> None:
        print(msg, flush=True)

    cases = load_pools(Path(args.pools))
    if args.limit:
        cases = cases[: args.limit]
    big = [c for c in cases if len(c["items"]) > TOP_K]
    consensus = load_consensus(Path(args.run_dir))
    eval_c = load_eval_c(Path(args.run_dir))
    say(f"mode={mode}: {len(big)} queries with pool > {TOP_K} (of {len(cases)} pools)")
    try:
        rows = run_queries(big, llm, source, say)
        chips: list[dict[str, Any]] = []
        if args.chips_pools:
            say("chip queries:")
            chips = run_queries(load_pools(Path(args.chips_pools)), llm, source, say)
    except GateStop as exc:
        say(f"STOPPED: {exc}\nrerun the same command to resume from the cache ({out}).")
        return 3
    res = summarize(rows, eval_c, consensus, n_skipped_pool=len(cases) - len(big))
    res["mode"] = mode
    (out / f"gate_{mode}.json").write_text(json.dumps({"summary": res, "rows": rows, "chips": chips}, indent=1))
    md = render_md(res, chips)
    (out / f"gate_{mode}.md").write_text(md, encoding="utf-8")
    say(md)
    return 0 if res["pass_ndcg"] and res["pass_clean_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
