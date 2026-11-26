"""ID-format compliance of RAW model output (V5 live-eval metric, amendment 1).

The respond prompt asks the model to refer to items only as ``[item:ID]``. This measures how often
it does, from the raw text (before rendering): item mentions written as ``[item:ID]`` divided by
all item mentions (those tokens plus plain-name mentions found by the claim checker's name
matcher). A diagnostic, with no pass/fail threshold; a run with no mention at all is excluded.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from src.agents.claim_checker import _mentions, _sentences

_ITEM_TOKEN = re.compile(r"\[\s*item\s*:\s*[^\]\s]{1,80}\s*\]", re.IGNORECASE)
_OTHER_TOKEN = re.compile(r"\[\s*[A-Za-z_]+\s*:\s*[^\]\s]{1,80}\s*\]")


def count_mentions(raw: str, items: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """(id_tokens, plain_name_mentions) in one raw reply."""
    tokens = len(_ITEM_TOKEN.findall(raw))
    stripped = _OTHER_TOKEN.sub(" ", raw)  # remove every tag so a name inside one is not counted
    plain = 0
    for start, end in _sentences(stripped):
        groups, _ = _mentions(stripped[start:end], list(items))
        plain += len(groups)
    return tokens, plain


def summarise(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run ``id_tokens`` / ``plain_mentions`` into the report block."""
    total: dict[str, dict[str, int]] = defaultdict(
        lambda: {"runs": 0, "runs_with_mentions": 0, "id_tokens": 0, "plain_mentions": 0}
    )
    for r in runs:
        tokens, plain = int(r.get("id_tokens", 0)), int(r.get("plain_mentions", 0))
        for key in ("all", str(r.get("path", ""))):
            d = total[key]
            d["runs"] += 1
            d["runs_with_mentions"] += bool(tokens + plain)
            d["id_tokens"] += tokens
            d["plain_mentions"] += plain
    out: dict[str, Any] = {}
    for key, d in sorted(total.items()):
        mentions = d["id_tokens"] + d["plain_mentions"]
        out[key] = {**d, "rate": round(d["id_tokens"] / mentions, 4) if mentions else None}
    return out
