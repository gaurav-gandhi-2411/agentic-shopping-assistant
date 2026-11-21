"""Before/after report for the deterministic rerank rules on the special-case fixture.

For each query in evals/fixtures/rerank_special_cases.json (hand-written pools, rule violators
FIRST in retrieval order) it counts hard-rule violations in the shown top-5 under three arms:

  no_rules        the model's order is shown as is. The simulated model accepts retrieval order
                  (answers 1..8): the worst case of "the prompt rules were not followed", and
                  also exactly what the retrieval-order fallback shows.
  old_master_code the only deterministic part of the old master reranker: its colour-diversity
                  swap (ported verbatim below); its other rules were prompt text, so how well
                  gpt-oss obeyed them is NOT measurable offline (it needs live calls).
  new_rules       production rerank() (rerank_filters.apply_rules) with the same simulated model.

A violation is `rerank_filters.rule_violations` (the same definition the filters use, so
new_rules = 0 wherever enough compliant candidates exist; that is by construction, the point of
the report is the size of the before column and that the filters leave compliant results alone).
Also reported: shown count, and distinct colours in the top-5 of date-night queries.

    python scripts/rerank_rules_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, cast

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.agents.rerank_filters import rule_violations  # noqa: E402
from src.agents.reranker import rerank  # noqa: E402
from src.llm.client import LLMCompletion  # noqa: E402

FIXTURE = REPO / "evals" / "fixtures" / "rerank_special_cases.json"
_DATE_NIGHT_RE = re.compile(
    r"\b(date\s+night|date-night|evening\s+out|evening|cocktail|wedding|gala)\b", re.IGNORECASE)


def old_enforce_colour_diversity(selected: list[dict], candidates: list[dict], query: str) -> list[dict]:
    """Verbatim logic of `_enforce_colour_diversity` in src/agents/reranker.py @ ad17f47."""
    if not _DATE_NIGHT_RE.search(query):
        return selected
    colours = [(it.get("colour") or "").lower() for it in selected]
    if len(set(colours)) >= 2:
        return selected
    dominant = colours[0] if colours else ""
    seen_ids = {it["article_id"] for it in selected}
    for cand in candidates:
        if cand["article_id"] in seen_ids:
            continue
        if (cand.get("colour") or "").lower() != dominant:
            return selected[:-1] + [cand]
    return selected


class RetrievalOrderModel:
    """Simulated model that answers 1..8 (accepts retrieval order)."""

    def complete(self, messages: list[dict], **kw: Any) -> LLMCompletion:
        return LLMCompletion("1,2,3,4,5,6,7,8", "stop", 600, 100, 80, 500, "sim")


def _count(query: str, shown: list[dict]) -> int:
    return sum(len(rule_violations(query, it)) for it in shown)


def _colours(query: str, shown: list[dict]) -> int | None:
    if not _DATE_NIGHT_RE.search(query):
        return None
    return len({(it.get("colour") or "").lower() for it in shown})


def evaluate(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for c in cases:
        q, items = c["text"], c["items"]
        before = items[:5]
        old = old_enforce_colour_diversity(items[:5], items, q)
        new = rerank(q, items, cast(Any, RetrievalOrderModel()), top_k=5)
        rows.append({
            "id": c["id"], "query": q,
            "no_rules": {"violations": _count(q, before), "n": len(before), "colours": _colours(q, before)},
            "old_master_code": {"violations": _count(q, old), "n": len(old), "colours": _colours(q, old)},
            "new_rules": {"violations": _count(q, new), "n": len(new), "colours": _colours(q, new)},
        })
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["| case | query | violations no_rules / old_master_code / new_rules | shown (new) | "
             "date-night colours no_rules / old / new |", "|---|---|---|---|---|"]
    tot = {k: 0 for k in ("no_rules", "old_master_code", "new_rules")}
    for r in rows:
        for k in tot:
            tot[k] += r[k]["violations"]
        col = [r[k]["colours"] for k in ("no_rules", "old_master_code", "new_rules")]
        lines.append(
            f"| {r['id']} | {r['query']} | {r['no_rules']['violations']} / "
            f"{r['old_master_code']['violations']} / {r['new_rules']['violations']} | "
            f"{r['new_rules']['n']} | {' / '.join('-' if x is None else str(x) for x in col)} |")
    lines.append(f"| total | {len(rows)} queries | {tot['no_rules']} / {tot['old_master_code']} / "
                 f"{tot['new_rules']} | | |")
    return "\n".join(lines)


def main() -> int:
    cases = json.loads(FIXTURE.read_text("utf-8"))["cases"]
    print(render(evaluate(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
