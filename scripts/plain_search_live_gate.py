"""Live gate for plain-search post-processing (rule: evals/retrieval/results/plain_search_gate_rule.md).

Runs the real graph in-process (real index, real GroqClient on gpt-oss-20b ONLY) for the 5 landing
chips, each in a fresh session, `--runs` times, and prints every shown card (title, store, price,
gender) with the per-turn criteria:

  zero duplicate products, <= 2 per store, both genders on the gender-neutral chips,
  price <= budget on a budget chip that reaches plain search.

Shared Groq key rules (CLAUDE.md "SHARED GROQ KEY"): the production fallback tier is switched off
here (no groq_fallback_models, the reranker's 120b fallback disabled), so a failed 20b call ends in
retrieval order / the busy message and never reaches gpt-oss-120b or qwen. A hard cap on live calls
stops the run. GROQ_API_KEY is read by name only (rerank_gate.read_named_key); the rest of the
.env is never loaded and no value is printed.

    python scripts/plain_search_live_gate.py --runs 2 --max-calls 60 --out <scratch>/live.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rerank_gate as rg  # noqa: E402

from src.agents import search_postprocess as pp  # noqa: E402

GATE_MODEL = "openai/gpt-oss-20b"
CHIPS = [
    "Sangeet outfit under ₹3,000",
    "Haldi kurta set",
    "Wedding guest saree",
    "Office wear under ₹1,500",
    "Men's kurta for a wedding",
]
DEFAULT_INDEX = Path("C:/Users/gaura/ml-projects/agentic-shopping-assistant/data/processed/unified")


class CallBudgetExceeded(RuntimeError):
    """The hard cap on live LLM calls was reached."""


class CountingLLM:
    """Wraps the real client: counts calls/tokens, enforces the model and the call cap."""

    def __init__(self, inner: Any, max_calls: int) -> None:
        self._inner = inner
        self.max_calls = max_calls
        self.calls = 0
        self.tokens = 0
        self.models: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name not in ("complete", "generate", "chat", "generate_stream", "chat_stream"):
            return attr

        def wrapped(*a: Any, **kw: Any) -> Any:
            if self.calls >= self.max_calls:
                raise CallBudgetExceeded(f"{self.max_calls} live calls reached")
            self.calls += 1
            res = attr(*a, **kw)
            model = getattr(res, "model", None)
            if model:
                self.models.add(str(model))
                if model != GATE_MODEL:
                    raise RuntimeError(f"foreign model {model!r}: only {GATE_MODEL!r} is allowed")
            self.tokens += int(getattr(res, "prompt_tokens", 0) or 0) + int(
                getattr(res, "completion_tokens", 0) or 0
            )
            return res

        return wrapped


def initial_state(query: str, llm: Any, config: dict) -> dict[str, Any]:
    from src.memory.conversation import ConversationMemory

    return {
        "messages": [{"role": "user", "content": query}], "user_query": query,
        "current_plan": None, "tool_calls": [], "retrieved_items": [], "filters": {},
        "final_answer": None, "iteration": 0, "new_items_this_turn": False,
        "out_of_catalogue": False, "excluded_colours": None, "anchor_article_id": None,
        "anchor_is_owned": False, "outfit_rationale": None, "outfit_variants": None,
        "more_options": None, "more_options_budget_inr": None,
        "_memory": ConversationMemory(llm, config),
    }


def criteria(items: list[dict], cap: float | None, neutral: bool, plain: bool) -> dict[str, Any]:
    dups = sum(
        1 for i in range(len(items)) for j in range(i)
        if pp.same_product(items[i], items[j]) or pp.legacy_key(items[i]) == pp.legacy_key(items[j])
    )
    stores = [str(i.get("store")) for i in items]
    mx = max((stores.count(s) for s in set(stores)), default=0)
    genders = {pp.gender_of(i) for i in items}
    return {
        "plain_search": plain,
        "n_cards": len(items),
        "duplicates": dups, "pass_no_duplicates": dups == 0,
        "max_per_store": mx, "pass_store_cap": mx <= pp.MAX_PER_STORE,
        "both_genders": {"women", "men"} <= genders,
        "pass_both_genders": (not neutral) or ({"women", "men"} <= genders),
        "over_budget": sum(1 for i in items if cap is not None and not pp.within_budget(i, cap)),
        "pass_budget": cap is None or all(pp.within_budget(i, cap) for i in items),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-dir", default=str(DEFAULT_INDEX))
    ap.add_argument("--env-file", default=str(DEFAULT_INDEX.parents[2] / ".env"))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-s", type=float, default=8.0, help="pause between turns (shared key)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    key = os.environ.get(rg.KEY_NAME) or rg.read_named_key(Path(args.env_file))
    if not key:
        raise SystemExit(f"{rg.KEY_NAME} not found by name")
    os.environ[rg.KEY_NAME] = key  # in-process only; never printed
    os.environ.pop("DATABASE_URL", None)

    import yaml

    config = yaml.safe_load((REPO / "config.yaml").read_text("utf-8"))
    config["llm"]["provider"] = "groq"
    config["llm"]["groq_model"] = GATE_MODEL
    config["llm"]["groq_fallback_models"] = []  # never the 120b / qwen tier on the shared key
    from src.agents import reranker as rr

    rr.FALLBACK_KINDS = ()  # the reranker's own one-shot 120b fallback: off
    from src.agents.graph import build_graph
    from src.llm.client import GroqClient
    from src.retrieval.snapshot import load_index_parts

    llm = CountingLLM(GroqClient(config), args.max_calls)
    retriever, df = load_index_parts(Path(args.index_dir), config)
    agent = build_graph(retriever, df, llm, config, streaming_mode=False)

    turns: list[dict[str, Any]] = []
    for run in range(1, args.runs + 1):
        for chip in CHIPS:
            t0 = time.monotonic()
            res = agent.invoke(initial_state(chip, llm, config))
            items = res.get("retrieved_items") or []
            tc = res.get("tool_calls") or []
            plain = any("search" in c for c in tc)
            f = pp.budget_cap((res.get("filters") or {}), None)
            neutral = plain and pp.query_gender(res.get("filters") or {}) is None
            crit = criteria(items, f, neutral, plain)
            turns.append({
                "run": run, "chip": chip, "route": [next(iter(c)) for c in tc],
                "price_max": f, "gender_neutral": neutral, "criteria": crit,
                "gender_refine": res.get("gender_refine"),
                "latency_s": round(time.monotonic() - t0, 1),
                "cards": [{"title": i.get("display_name"), "store": i.get("store"),
                           "price": i.get("price_inr"), "gender": i.get("gender")} for i in items],
            })
            print(f"\n== run {run} | {chip} | route={turns[-1]['route']} | cap={f} | neutral={neutral}"
                  f" | refine={'yes' if res.get('gender_refine') else 'no'}")
            for c in turns[-1]["cards"]:
                print(f"   {c['store']:<14} {str(c['gender']):<8} {c['price']!s:>9}  {c['title']}")
            print("   criteria:", json.dumps(crit))
            time.sleep(args.gap_s)
    out = {"model": GATE_MODEL, "live_calls": llm.calls, "tokens": llm.tokens,
           "models_seen": sorted(llm.models), "turns": turns}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nlive calls={llm.calls} tokens={llm.tokens} models={sorted(llm.models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
