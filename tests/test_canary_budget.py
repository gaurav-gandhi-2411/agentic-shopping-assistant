"""Gate: all scheduled checks, worst case, stay <= 25% of the primary model's TPD. Pure maths."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "scripts" / "ops"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_budget_t", OPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


budget = _load("canary_budget")
canary = _load("canary")


def test_limit_is_25_percent_of_200k() -> None:
    assert budget.limit_tokens() == 50_000


def test_worst_case_with_the_shipped_retry_cap_fits() -> None:
    assert budget.worst_case(canary.MAX_RETRIES_PER_RUN) <= 50_000
    # Pinned so it is not forgotten: with today's reranker (before PR #136 / variant C) the worst
    # case is over 25% even with zero retries (docs/ops/canary-budget.md). Update when #136 lands.
    assert budget.worst_case(0, budget.SEARCH_TURN_NOW) > 50_000


def test_the_cap_is_not_looser_than_the_budget_allows() -> None:
    assert canary.MAX_RETRIES_PER_RUN <= budget.max_retries_that_fit()
    assert budget.worst_case(budget.max_retries_that_fit() + 1) > 50_000


def test_whole_run_retry_would_not_fit() -> None:
    # Why the retry is capped per run instead of repeating the whole run.
    assert budget.whole_run_retry() > 50_000


def test_exact_numbers_from_the_inputs() -> None:
    assert budget.ws_canary(2_150) == 2 * 1_390 + 9 * 2_150 == 22_130
    assert budget.worst_case(0) == 22_130 + 7_050 + 6_450 + 9_600 == 45_230
    assert budget.whole_run_retry() == 2 * 22_130 + 23_100 == 67_360


def test_query_mix_matches_the_canary_run() -> None:
    assert budget.OUTFIT_QUERIES + budget.SEARCH_QUERIES == canary.QUERIES_PER_RUN


def test_table_renders_every_scenario() -> None:
    out = budget.table(canary.MAX_RETRIES_PER_RUN)
    assert "whole run retried" in out and "50,000" in out and "22,130" in out


def test_status_run_cap_is_tied_to_the_endpoint_constant() -> None:
    import re

    src = (OPS.parents[1] / "api" / "routes" / "canary.py").read_text(encoding="utf-8")
    m = re.search(r"^MAX_RUNS_PER_DAY = (\d+)", src, re.M)
    assert m is not None and int(m.group(1)) == budget.STATUS_RUNS_PER_DAY
