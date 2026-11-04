"""W10 grounding eval harness: prompts -> real routes (REST /chat and WS /chat/stream) -> checker.

    python -m evals.grounding.run --llm mock --agent stub --smoke
    python -m evals.grounding.run --llm mock --agent real --scratch <dir> [--sample 10,10]
    python -m evals.grounding.run --llm groq --agent real --scratch <dir> --sample 10,10 \
        --path ws --max-llm-calls 60

--agent stub : offline. Keyword retrieval over evals/fixtures/grounding_catalogue.json, then the
               REAL respond prompt (graph.RESPOND_PROMPT / _format_items_for_response), the REAL
               routes, ItemSummary serialisation and (REST only) validate_response. No index.
--agent real : the compiled LangGraph agent over the local index (data/processed/unified), roots
               redirected into --scratch (hermetic, same recipe as scripts/ops/count_llm_calls.py).
--llm mock   : evals.grounding.mock_llm (grounded / ungrounded / mixed templates), zero cost.
--llm groq   : real Groq via the repo's GroqClient; key read from GROQ_API_KEY (env, else the
               repo .env, name only, value never printed). Hard-capped by --max-llm-calls.

Writes reports/grounding_eval_<UTC>_<llm>_<agent>.json (commit SHA, prompt-set hash, model, n,
violations, per-category counts). Exit code 1 only for `--llm mock` when the checker gate fails.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from evals.grounding.checker import check
from evals.grounding.mock_llm import SEED, MockLLM

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "evals" / "fixtures"
MINIMAL_CONFIG = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}
# 5 regular + 5 adversarial (one per adversarial category), chosen to cover price echo, plain
# search, occasion, budget, and every bait kind. Fixed list: the smoke subset must never drift.
SMOKE_IDS = (
    "r001", "r011", "r021", "r061", "r098",
    "a001", "a012", "a021", "a031", "a041",
)


# --------------------------------------------------------------------------- fixtures


def load_cases() -> list[dict[str, Any]]:
    """All 150 prompts (regular then adversarial), each tagged with its set."""
    cases: list[dict[str, Any]] = []
    for name in ("regular", "adversarial"):
        data = json.loads((FIXTURES / f"grounding_{name}.json").read_text(encoding="utf-8"))
        cases += [{**c, "set": data["set"]} for c in data["cases"]]
    return cases


def select(
    cases: list[dict[str, Any]], smoke: bool, sample: str | None, ids: str | None = None
) -> list[dict[str, Any]]:
    """Smoke list, a seeded stratified sample ('R,A' per set, round-robin over categories), or all."""
    by_id = {c["id"]: c for c in cases}
    if ids:
        return [by_id[i] for i in ids.split(",")]
    if smoke:
        return [by_id[i] for i in SMOKE_IDS]
    if not sample:
        return cases
    want = dict(zip(("regular", "adversarial"), (int(x) for x in sample.split(","))))
    rng = random.Random(SEED)
    out: list[dict[str, Any]] = []
    for set_name, n in want.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in cases:
            if c["set"] == set_name:
                groups[c["category"]].append(c)
        pools = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
        picked: list[dict[str, Any]] = []
        while len(picked) < n and any(pools.values()):
            for k in sorted(pools):
                if pools[k] and len(picked) < n:
                    picked.append(pools[k].pop())
        out += sorted(picked, key=lambda c: c["id"])
    return out


def prompt_set_hash(cases: list[dict[str, Any]]) -> str:
    """sha256 over the canonical JSON of the selected cases (id, category, prompt)."""
    canon = json.dumps(
        [(c["id"], c["category"], c["prompt"]) for c in cases], ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- agents


class StubAgent:
    """Keyword-retrieval agent that reuses the app's real respond prompt and serialisation."""

    def __init__(self, llm: Any, catalogue: list[dict[str, Any]], streaming: bool) -> None:
        self.llm, self.catalogue, self.streaming = llm, catalogue, streaming

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        import re

        words = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for it in self.catalogue:
            tags = set(it["tags"].split()) | set(re.findall(r"[a-z]+", it["display_name"].lower()))
            score = len(words & tags)
            if score:
                scored.append((-score, it["article_id"], it))
        return [{**it, "image_url": f"https://example.invalid/{it['article_id']}.jpg"}
                for _, _, it in sorted(scored, key=lambda t: t[:2])[:3]]

    def invoke(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Mirror of respond_node's streaming / non-streaming branches (graph.py ~3965-4000)."""
        from src.agents.graph import RESPOND_PROMPT, _format_items_for_response
        from src.agents.grounding import validate_response

        q = state["user_query"]
        items = self._retrieve(q)
        prompt = RESPOND_PROMPT.format(
            user_query=q, items=_format_items_for_response(items),
            conversation="(no prior conversation)",
        )
        base: dict[str, Any] = {
            "retrieved_items": items, "filters": {}, "iteration": 1, "out_of_catalogue": False,
            "new_items_this_turn": bool(items), "excluded_colours": None,
            "tool_calls": [{"router_decision": {"action": "search", "query": q}},
                           {"search": {"query": q, "filters": {}}}],
        }
        if self.streaming:
            return {**base, "final_answer": None, "messages": state["messages"],
                    "current_plan": json.dumps({"action": "pending_respond", "prompt": prompt})}
        answer = self.llm.generate(prompt)
        answer, _flags = validate_response(answer, items, allow_price_mentions=True)
        return {**base, "final_answer": answer, "current_plan": None,
                "messages": state["messages"] + [{"role": "assistant", "content": answer}]}


@contextlib.contextmanager
def stub_env(llm: Any) -> Iterator[None]:
    """Patch api.deps + env so the real routes run against StubAgent; restore on exit."""
    import api.deps as deps

    catalogue = json.loads((FIXTURES / "grounding_catalogue.json").read_text(encoding="utf-8"))["items"]

    def get_factory() -> Any:
        def factory(memory: Any, streaming: bool = False) -> Any:
            return StubAgent(llm, catalogue, streaming)
        return factory

    saved = {k: getattr(deps, k) for k in ("_llm", "_config", "_session_store", "get_agent_factory")}
    env_keys = ("JWT_VERIFICATION_DISABLED", "RATE_LIMIT_PER_MINUTE")
    saved_env = {k: os.environ.get(k) for k in env_keys}
    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"
    from api.session import InMemorySessionStore

    deps._llm, deps._config = llm, MINIMAL_CONFIG
    deps._session_store = InMemorySessionStore()
    deps.get_agent_factory = get_factory  # type: ignore[assignment]
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(deps, k, v)
        for k, ev in saved_env.items():
            if ev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = ev


def redirect_roots(scratch: Path, hf_home: Path) -> None:
    """Point every writable root at `scratch`, print and assert them (hermetic probe rule)."""
    for sub in ("tmp", "home", "local", "roaming"):
        (scratch / sub).mkdir(parents=True, exist_ok=True)
    roots = {
        "TEMP": scratch / "tmp", "TMP": scratch / "tmp", "TMPDIR": scratch / "tmp",
        "USERPROFILE": scratch / "home", "HOME": scratch / "home",
        "LOCALAPPDATA": scratch / "local", "APPDATA": scratch / "roaming",
    }
    os.environ.update({k: str(v) for k, v in roots.items()})
    os.environ.update({
        "HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TQDM_DISABLE": "1", "JWT_VERIFICATION_DISABLED": "true", "DEMO_MODE": "",
        "DATABASE_URL": "", "RATE_LIMIT_PER_MINUTE": "100000", "PYTHONDONTWRITEBYTECODE": "1",
    })
    for key in roots:
        print("root", key, os.environ[key])
        assert os.path.realpath(os.environ[key]).startswith(os.path.realpath(scratch)), key
    print("root HF_HOME (read-only model cache)", hf_home)


class CountingLLM:
    """Proxy that counts every logical LLM call and refuses to exceed a hard cap."""

    def __init__(self, inner: Any, cap: int) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_cap", cap)
        object.__setattr__(self, "n_calls", 0)

    def _tick(self) -> None:
        if self.n_calls >= self._cap:
            raise RuntimeError(f"LLM call cap {self._cap} reached; refusing further calls")
        object.__setattr__(self, "n_calls", self.n_calls + 1)

    def generate(self, *a: Any, **k: Any) -> Any:
        self._tick()
        return self._inner.generate(*a, **k)

    def chat(self, *a: Any, **k: Any) -> Any:
        self._tick()
        return self._inner.chat(*a, **k)

    def generate_stream(self, *a: Any, **k: Any) -> Any:
        self._tick()
        return self._inner.generate_stream(*a, **k)

    def chat_stream(self, *a: Any, **k: Any) -> Any:
        self._tick()
        return self._inner.chat_stream(*a, **k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


def load_groq_key_from_repo_env() -> bool:
    """Ensure GROQ_API_KEY is set (env, else repo .env). Returns False if unavailable. Never prints."""
    if os.environ.get("GROQ_API_KEY"):
        return True
    env_file = REPO / ".env"
    if not env_file.exists():
        return False
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "GROQ_API_KEY" and value.strip().strip("'\""):
            os.environ["GROQ_API_KEY"] = value.strip().strip("'\"")
            return True
    return False


def init_real_agent(llm: Any, cfg: dict[str, Any]) -> None:
    """Load the local unified index (read-only) and compile the real graphs into api.deps."""
    import pandas as pd

    import api.deps as deps
    from api.session import InMemorySessionStore
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    index_dir = REPO / "data" / "processed" / "unified"
    if not (index_dir / "dense.faiss").exists():
        sys.exit(f"index not found at {index_dir}")
    df = pd.read_parquet(index_dir / "catalogue.parquet")
    retriever = HybridRetriever(
        DenseRetriever.load(cfg, index_dir), SparseRetriever.load(cfg, index_dir), df, cfg
    )
    deps._init(retriever, df, llm, cfg, session_store=InMemorySessionStore())


# --------------------------------------------------------------------------- one case


def _collect_items(src: dict[str, Any]) -> list[dict[str, Any]]:
    """Items rendered as cards: top-level, every outfit variant, the partner look."""
    items = list(src.get("items") or [])
    for v in src.get("outfit_variants") or []:
        items += v.get("items") or []
    items += (src.get("partner_look") or {}).get("items") or []
    return items


def _collect_text_and_amounts(src: dict[str, Any], response: str) -> tuple[str, list[float]]:
    """Assistant-visible strings (reply + every rationale) and code-computed money figures."""
    parts = [response, src.get("outfit_rationale") or ""]
    amounts = [src.get("budget_total_inr")]
    for v in src.get("outfit_variants") or []:
        parts.append(v.get("rationale") or "")
        amounts.append(v.get("budget_total_inr"))
    pl = src.get("partner_look") or {}
    parts.append(pl.get("outfit_rationale") or "")
    amounts.append(pl.get("budget_total_inr"))
    return "\n".join(p for p in parts if p), [float(a) for a in amounts if a]


def run_rest(client: Any, prompt: str) -> tuple[str, list[dict[str, Any]], list[float], int]:
    """POST /chat. Returns (assistant text, shown items, code amounts, http status)."""
    r = client.post("/chat", json={"message": prompt})
    if r.status_code != 200:
        return "", [], [], r.status_code
    body = r.json()
    text, amounts = _collect_text_and_amounts(body, body.get("response") or "")
    return text, _collect_items(body), amounts, 200


def run_ws(client: Any, prompt: str) -> tuple[str, list[dict[str, Any]], list[float], int]:
    """WS /chat/stream. Text is the concatenated token frames (what the browser renders)."""
    tokens: list[str] = []
    items: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": prompt})
        while True:
            msg = ws.receive_json()
            kind = msg.get("type")
            if kind == "items":
                items += msg["items"]
            elif kind == "token":
                tokens.append(msg["text"])
            elif kind == "done":
                final = msg["final_state"]
                break
            elif kind in ("error", "cancelled"):
                return "", items, [], 500
    streamed = "".join(tokens)
    text, amounts = _collect_text_and_amounts(final, streamed)
    return text, items + _collect_items(final), amounts, 200


def run_case(client: Any, case: dict[str, Any], path: str, llm: Any) -> dict[str, Any]:
    """Run one prompt on one path and check the result."""
    if isinstance(llm, MockLLM):
        llm.events.clear()
    runner = run_rest if path == "rest" else run_ws
    text, items, amounts, status = runner(client, case["prompt"])
    violations = check(text, items, user_text=case["prompt"], extra_amounts=amounts)
    rec: dict[str, Any] = {
        "id": case["id"], "set": case["set"], "category": case["category"], "path": path,
        "prompt": case["prompt"], "http_ok": status == 200, "n_items": len(items),
        "text": text,
        "shown": [{"id": i.get("article_id"), "name": i.get("display_name"),
                   "price_inr": i.get("price_inr")} for i in items],
        "violations": [{"kind": v.kind, "text": v.text, "reason": v.reason} for v in violations],
    }
    if isinstance(llm, MockLLM):
        evs = list(llm.events)
        rec["mock"] = {
            "templates": [e.template for e in evs],
            "injected": any(e.injected for e in evs),
            "neutralised": any(e.injected and e.raw_text.strip() != text.strip() for e in evs)
            and not violations,
        }
    return rec


# --------------------------------------------------------------------------- reporting


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-category / per-kind / per-path counts."""
    per_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"runs": 0, "runs_with_violation": 0, "violations": 0})
    by_kind: Counter[str] = Counter()
    by_path: dict[str, dict[str, int]] = defaultdict(lambda: {"runs": 0, "runs_with_violation": 0, "violations": 0})
    for r in results:
        n = len(r["violations"])
        for d in (per_cat[f"{r['set']}/{r['category']}"], by_path[r["path"]]):
            d["runs"] += 1
            d["runs_with_violation"] += bool(n)
            d["violations"] += n
        by_kind.update(v["kind"] for v in r["violations"])
    return {"per_category": dict(sorted(per_cat.items())), "by_kind": dict(by_kind),
            "by_path": dict(by_path)}


def mock_gate(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Grounded templates must produce 0 violations; injected ones must be caught.

    On the REST path validate_response may scrub an injected sentence before the checker sees it
    ("neutralised"); that counts as caught. On WS nothing scrubs, so a miss there is a real miss.
    """
    grounded_flagged = [f"{r['id']}/{r['path']}" for r in results
                        if not r["mock"]["injected"] and r["violations"]]
    missed = [f"{r['id']}/{r['path']}" for r in results
              if r["mock"]["injected"] and not r["violations"] and not r["mock"]["neutralised"]]
    injected = [r for r in results if r["mock"]["injected"]]
    return {
        "mode": mode,
        "injected_runs": len(injected),
        "injected_flagged": sum(bool(r["violations"]) for r in injected),
        "injected_neutralised_by_validate_response": sum(r["mock"]["neutralised"] for r in injected),
        "injected_missed": missed,
        "grounded_runs": len(results) - len(injected),
        "grounded_runs_flagged": grounded_flagged,
        "http_failures": [f"{r['id']}/{r['path']}" for r in results if not r["http_ok"]],
        "ok": not missed and not grounded_flagged and all(r["http_ok"] for r in results),
    }


def git_info() -> dict[str, Any]:
    """Commit SHA and whether tracked eval/app sources differ from it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty_tracked_or_eval": bool(git("status", "--porcelain", "--", "api", "src", "evals"))}


def run_all(cases: list[dict[str, Any]], llm: Any, paths: list[str]) -> list[dict[str, Any]]:
    """Run every case on every path with a fresh session each time."""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app, raise_server_exceptions=True)  # no lifespan: no GCS, no real LLM init
    results = []
    for case in cases:
        for path in paths:
            results.append(run_case(client, case, path, llm))
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--llm", choices=("mock", "groq"), default="mock")
    ap.add_argument("--agent", choices=("stub", "real"), default="stub")
    ap.add_argument("--mock-mode", choices=("grounded", "ungrounded", "mixed"), default="mixed")
    ap.add_argument("--path", choices=("rest", "ws", "both"), default="both")
    ap.add_argument("--smoke", action="store_true", help="the fixed 10-prompt CI subset")
    ap.add_argument("--sample", default=None, help="stratified sample 'R,A', e.g. 10,10")
    ap.add_argument("--ids", default=None, help="comma-separated case ids (debugging)")
    ap.add_argument("--max-llm-calls", type=int, default=60)
    ap.add_argument("--scratch", type=Path, default=None, help="required for --agent real")
    ap.add_argument("--hf-home", type=Path,
                    default=Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"))
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports")
    args = ap.parse_args(argv)

    if args.agent == "real":
        if not args.scratch:
            sys.exit("--agent real needs --scratch <empty dir you own>")
        hf_home = args.hf_home.resolve()  # resolve BEFORE HOME is redirected
        redirect_roots(args.scratch.resolve(), hf_home)
        os.chdir(args.scratch.resolve())
    sys.path.insert(0, str(REPO))

    from src.catalogue.loader import load_config

    cfg = load_config(str(REPO / "config.yaml"))
    paths = ["rest", "ws"] if args.path == "both" else [args.path]
    cases = select(load_cases(), args.smoke, args.sample, args.ids)

    if args.llm == "mock":
        llm: Any = MockLLM(mode=args.mock_mode)
        model = f"mock/{args.mock_mode}"
    else:
        if not load_groq_key_from_repo_env():
            sys.exit("no GROQ_API_KEY in env or repo .env; live run skipped")
        from src.llm.client import GroqClient

        cfg["llm"]["provider"] = "groq"
        llm = CountingLLM(GroqClient(cfg), args.max_llm_calls)
        model = f"groq/{cfg['llm'].get('groq_model')}"

    results: list[dict[str, Any]] = []
    status = "ok"
    try:
        if args.agent == "stub":
            with stub_env(llm):
                results = run_all(cases, llm, paths)
        else:
            init_real_agent(llm, cfg)
            results = run_all(cases, llm, paths)
    except RuntimeError as exc:  # LLM cap reached: keep partial results, say so
        status = f"aborted: {exc}"

    summary = summarise(results)
    report: dict[str, Any] = {
        **git_info(),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "llm": args.llm, "model": model, "agent": args.agent, "paths": paths,
        "status": status, "seed": SEED,
        "n_prompts": len(cases), "n_runs": len(results),
        "prompt_set_hash": prompt_set_hash(cases),
        "llm_calls": getattr(llm, "n_calls", None),
        "total_violations": sum(len(r["violations"]) for r in results),
        "n_runs_with_text": sum(bool(r["text"].strip()) for r in results),
        # 0 violations only means something when every run produced text; empty or failed
        # runs (seen live: Groq streams returning no content) are NOT clean runs.
        "measurement_valid": bool(results) and all(r["http_ok"] and r["text"].strip() for r in results),
        "runs_with_violation": sum(bool(r["violations"]) for r in results),
        **summary,
        # a run that failed or produced no text would otherwise read as "0 violations"
        "http_failures": [f"{r['id']}/{r['path']}" for r in results if not r["http_ok"]],
        "empty_text_runs": [f"{r['id']}/{r['path']}" for r in results if not r["text"].strip()],
        "runs": [{k: r[k] for k in ("id", "path", "category", "http_ok", "n_items", "shown", "text")}
                 for r in results],
        "violations": [
            {"id": r["id"], "path": r["path"], "category": r["category"], "prompt": r["prompt"],
             "assistant_text": r["text"], "kind": v["kind"], "span": v["text"],
             "reason": v["reason"]}
            for r in results for v in r["violations"]
        ],
    }
    if args.llm == "mock" and results:
        report["mock_gate"] = mock_gate(results, args.mock_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"grounding_eval_{stamp}_{args.llm}_{args.agent}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "commit", "model", "agent", "paths", "status", "n_prompts", "n_runs", "llm_calls",
        "total_violations", "runs_with_violation", "n_runs_with_text", "measurement_valid",
        "by_kind", "by_path", "http_failures", "empty_text_runs")}, indent=2))
    if "mock_gate" in report:
        print("mock_gate", json.dumps(report["mock_gate"]))
    print("report", out)
    return 1 if args.llm == "mock" and not report.get("mock_gate", {}).get("ok", False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
