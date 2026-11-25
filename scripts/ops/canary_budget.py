"""Worst-case daily token budget of every SCHEDULED check (docs/ops/canary-budget.md).

Rule (GG, 2026-11-25): all scheduled checks combined must stay <= 25% of the primary model's
tokens per day. The 200,000 TPD figure is UNVERIFIED (docs table, not an API header), so the
ceiling below, 25% of it = 50,000, is only as good as that figure.

Pure arithmetic, stdlib only, no network: ``python scripts/ops/canary_budget.py`` prints the
table. The per-turn costs are INPUTS, taken from the 2026-11-18..21 Cloud Logging ``llm_call``
records (outfit: n=11 Sangeet canary turns) and from PR #136's reranker variant C; this script
does not re-measure them. ``MAX_RETRIES_PER_RUN`` in scripts/ops/canary.py is the cap this
arithmetic yields; tests/test_canary_budget.py fails if that cap pushes the worst case past 25%.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

# ---- inputs -----------------------------------------------------------------------------------
TPD_PRIMARY = 200_000  # primary model tokens/day: UNVERIFIED (Groq docs table, not a header)
SHARE_LIMIT = 0.25  # all scheduled checks combined
OUTFIT_TURN = 1_390  # tokens: 1 LLM call; MEASURED, n=11 canary Sangeet turns
SEARCH_TURN_NOW = 3_345  # tokens: reranker ~2,000 + respond ~1,350; MEASURED
SEARCH_TURN_C = 2_150  # tokens: with reranker variant C (752 per rerank), PR #136, NOT merged yet
READYZ_PROBE = 100  # tokens per /readyz LLM probe: ESTIMATE (max_tokens=1 prompt "ping")
OUTFIT_QUERIES = 2  # of the 11 per run: the "sangeet outfit" query and the Sangeet chip
SEARCH_QUERIES = 9  # the other 9
DEPLOYS_PER_DAY = 3
READYZ_PROBES_PER_DEPLOY = 2  # tag URL + main URL
UPTIME_PROBES_PER_DAY = 96  # uptime-alert.yml cron */15, nominal worst case
# api/routes/canary.py MAX_RUNS_PER_DAY (tied by a test): the public endpoint's hard daily cap.
# Nothing calls it on a schedule today; this is the worst case if it is hammered.
STATUS_RUNS_PER_DAY = 3


@dataclass(frozen=True)
class Row:
    """One line of the budget: ``tokens`` per day under the stated worst-case assumption."""

    name: str
    tokens: int
    note: str


def ws_canary(search_turn: int) -> int:
    """The daily WS canary without retries: 2 outfit + 9 search turns."""
    return OUTFIT_QUERIES * OUTFIT_TURN + SEARCH_QUERIES * search_turn


def other_checks(search_turn: int) -> list[Row]:
    """Everything scheduled except the WS canary."""
    smoke = DEPLOYS_PER_DAY * (search_turn + READYZ_PROBES_PER_DEPLOY * READYZ_PROBE)
    return [
        Row("deploy smokes", smoke,
            f"{DEPLOYS_PER_DAY}/day x (1 chat {search_turn} + {READYZ_PROBES_PER_DEPLOY} readyz x {READYZ_PROBE})"),
        Row("/canary/status", STATUS_RUNS_PER_DAY * search_turn,
            f"<= {STATUS_RUNS_PER_DAY} runs/day x {search_turn} (the cap; no scheduled caller exists)"),
        Row("/readyz uptime-alert", UPTIME_PROBES_PER_DAY * READYZ_PROBE,
            f"{UPTIME_PROBES_PER_DAY}/day x {READYZ_PROBE} (cron */15, nominal = worst case)"),
    ]


def limit_tokens() -> int:
    """25% of the primary model's tokens/day (50,000 with the unverified 200,000)."""
    return int(TPD_PRIMARY * SHARE_LIMIT)


def worst_case(max_retries: int, search_turn: int = SEARCH_TURN_C) -> int:
    """All scheduled checks per day with at most ``max_retries`` failed queries retried.

    A retried query is priced as a search turn: the cap cannot know which query will fail, and a
    search turn is the more expensive kind.
    """
    return (ws_canary(search_turn) + max_retries * search_turn
            + sum(r.tokens for r in other_checks(search_turn)))


def whole_run_retry(search_turn: int = SEARCH_TURN_C) -> int:
    """Alternative design: every query fails and the WHOLE run is retried (2x the WS canary)."""
    return 2 * ws_canary(search_turn) + sum(r.tokens for r in other_checks(search_turn))


def max_retries_that_fit(search_turn: int = SEARCH_TURN_C) -> int:
    """Largest retry cap whose worst case is still <= the limit."""
    n = 0
    while worst_case(n + 1, search_turn) <= limit_tokens():
        n += 1
    return n


def _pct(tokens: int) -> str:
    return f"{100 * tokens / TPD_PRIMARY:.1f}%"


def table(max_retries: int) -> str:
    """Markdown table of the worst case, for the current search cost and for variant C."""
    out = [f"Limit: {int(SHARE_LIMIT * 100)}% of {TPD_PRIMARY:,} TPD (UNVERIFIED) = "
           f"{limit_tokens():,} tokens/day", ""]
    for label, st in (("reranker variant C (PR #136, not merged)", SEARCH_TURN_C),
                      ("current reranker", SEARCH_TURN_NOW)):
        out += [f"Search turn = {st:,} tokens, {label}", "",
                "| Check | Tokens/day | Assumption |", "|---|---:|---|",
                f"| WS canary, no retry | {ws_canary(st):,} | {OUTFIT_QUERIES} outfit x {OUTFIT_TURN:,}"
                f" + {SEARCH_QUERIES} search x {st:,} |"]
        out += [f"| {r.name} | {r.tokens:,} | {r.note} |" for r in other_checks(st)]
        base = worst_case(0, st)
        out += [f"| **Total, no retry** | **{base:,}** ({_pct(base)} of TPD) | |"]
        for k in sorted({max_retries, max_retries_that_fit(st)}):
            tot = worst_case(k, st)
            ok = "fits" if tot <= limit_tokens() else "EXCEEDS"
            out.append(f"| Total, <= {k} failed queries retried | {tot:,} ({_pct(tot)}) | "
                       f"{k} x {st:,}; {ok} {limit_tokens():,} |")
        whole = whole_run_retry(st)
        out += [f"| Total, whole run retried (all {OUTFIT_QUERIES + SEARCH_QUERIES} fail) | {whole:,} "
                f"({_pct(whole)}) | 2 x {ws_canary(st):,}; "
                f"{'fits' if whole <= limit_tokens() else 'EXCEEDS'} {limit_tokens():,} |", ""]
        out.append(f"Largest retry cap that fits at this search cost: {max_retries_that_fit(st)}")
        out.append("")
    return "\n".join(out)


def _canary_max_retries() -> int:
    spec = importlib.util.spec_from_file_location("canary", Path(__file__).with_name("canary.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("canary", mod)  # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return int(mod.MAX_RETRIES_PER_RUN)


def main() -> int:
    """Print the table for the retry cap currently set in canary.py."""
    print(table(_canary_max_retries()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
