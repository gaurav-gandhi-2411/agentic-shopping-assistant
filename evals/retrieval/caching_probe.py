"""Groq prompt-caching probe (stage 2 only; built and mock-tested in stage 1, NOT run).

Sends the SAME >= 1,100-token prompt (production reranker system prompt + a real 20-candidate
block) repeatedly to gpt-oss-20b and gpt-oss-120b: 3 calls each, 6 live calls in total (a
hard-coded cap, `MAX_LIVE_CALLS`). For each call it records usage.prompt_tokens_details
.cached_tokens, total tokens and the x-ratelimit-remaining-tokens header, and from the deltas
tells whether cached tokens are counted against the TPM limit.

    python -m evals.retrieval.caching_probe --live      # also needs STAGE2_LIVE_OK=1

Caveat built into the verdict: the TPM bucket refills continuously, and the 8 s inter-request
throttle lets ~limit/60*8 tokens back in between calls, so the consumption is corrected by the
refill estimate and the verdict says INCONCLUSIVE when the corrected delta cannot separate
"cached tokens counted" from "not counted".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.retrieval.common import (
    CANDIDATE_DIR,
    RERANKER_MODEL,
    RUN_DIR,
    count_tokens,
    load_candidates,
)
from evals.retrieval.groq_judge import (
    GroqRunner,
    Ledger,
    ModelSpec,
    assert_live_allowed,
    httpx_transport,
)

MAX_LIVE_CALLS = 6
CALLS_PER_MODEL = 3
MIN_PREFIX_TOKENS = 1100  # Groq caches prefixes of >= 1,024 tokens
MODELS = {"gpt-oss-20b": RERANKER_MODEL, "gpt-oss-120b": "openai/gpt-oss-120b"}


def build_probe_prompt(cand_dir: Path = CANDIDATE_DIR) -> tuple[str, str]:
    """(system, user) = production reranker prompt over a real candidate block (>= 1,100 tokens)."""
    from src.agents.reranker import _SYSTEM, _format_candidates

    items = load_candidates("q01", cand_dir)["items"]
    user = (
        f'Query: "probe"\n\nProducts:\n{_format_candidates(items)}\n\nSelect the best 5 products.'
    )
    if count_tokens(_SYSTEM + user) < MIN_PREFIX_TOKENS:
        raise RuntimeError("probe prompt is shorter than the 1,100-token minimum")
    return _SYSTEM, user


def _num(headers: dict[str, str], key: str) -> float | None:
    v = headers.get(key)
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verdict for one model from its ordered call rows (see `probe`)."""
    later = rows[1:]
    cached = [r["cached_tokens"] or 0 for r in later]
    if not later or max(cached) == 0:
        return {"verdict": "NOT VERIFIED", "reason": "no cached_tokens reported on repeat calls"}
    evidence = []
    for prev, cur in zip(rows, rows[1:], strict=False):
        if None in (prev["remaining"], cur["remaining"], cur["limit"]):
            continue
        refill = cur["limit"] / 60.0 * cur["gap_s"]
        consumed = prev["remaining"] + refill - cur["remaining"]
        evidence.append(
            {
                "consumed_corrected": consumed,
                "total": cur["total_tokens"],
                "cached": cur["cached_tokens"] or 0,
            }
        )
    if not evidence:
        return {"verdict": "NOT VERIFIED", "reason": "rate-limit headers missing"}
    counted = sum(
        1
        for e in evidence
        if e["cached"] and e["consumed_corrected"] >= e["total"] - 0.1 * e["cached"]
    )
    not_counted = sum(
        1
        for e in evidence
        if e["cached"] and e["consumed_corrected"] <= e["total"] - 0.5 * e["cached"]
    )
    if not_counted and not counted:
        return {
            "verdict": "VERIFIED",
            "reason": "caching active; cached tokens NOT counted vs TPM",
            "evidence": evidence,
        }
    if counted and not not_counted:
        return {
            "verdict": "VERIFIED",
            "reason": "caching active; cached tokens STILL counted vs TPM",
            "evidence": evidence,
        }
    return {
        "verdict": "NOT VERIFIED",
        "reason": "INCONCLUSIVE rate-limit deltas",
        "evidence": evidence,
    }


def probe(
    runner: GroqRunner, system: str, user: str, models: dict[str, str] = MODELS
) -> dict[str, Any]:
    """Run <= 6 calls (3 per model) and return raw rows plus per-model verdicts."""
    if len(models) * CALLS_PER_MODEL > MAX_LIVE_CALLS:
        raise RuntimeError(f"probe would exceed the {MAX_LIVE_CALLS}-call cap")
    out: dict[str, Any] = {"calls": 0, "models": {}}
    for name, model in models.items():
        spec = ModelSpec(f"probe-{name}", model, {"reasoning_effort": "low"}, 150, 20_000)
        rows: list[dict[str, Any]] = []
        prev_t: float | None = None
        for i in range(CALLS_PER_MODEL):
            if out["calls"] >= MAX_LIVE_CALLS:
                raise RuntimeError("live-call cap reached")
            res = runner.call(spec, f"probe{i}", system, user, f"probe-{name}-{i}")
            out["calls"] += 1
            if res is None:
                raise RuntimeError(f"probe call failed for {name}")
            h, u = res["headers"], res["usage"]
            now = runner._clock()  # noqa: SLF001 - same clock the runner throttles with
            rows.append(
                {
                    "call": i + 1,
                    "prompt_tokens": u.get("prompt_tokens"),
                    "total_tokens": u.get("total_tokens"),
                    "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "remaining": _num(h, "x-ratelimit-remaining-tokens"),
                    "limit": _num(h, "x-ratelimit-limit-tokens"),
                    "gap_s": 0.0 if prev_t is None else now - prev_t,
                }
            )
            prev_t = now
        out["models"][name] = {"rows": rows, **analyse(rows)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument("--cand", default=str(CANDIDATE_DIR))
    args = ap.parse_args(argv)
    assert_live_allowed(args.live)
    system, user = build_probe_prompt(Path(args.cand))
    runner = GroqRunner(httpx_transport(), Ledger(Path(args.out) / "probe_ledger.jsonl"), None)
    res = probe(runner, system, user)
    print(json.dumps(res, indent=1))
    for name, m in res["models"].items():
        print(f"{name}: {m['verdict']} ({m['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
