"""split_accessory_query_by_family must not depend on PYTHONHASHSEED.

Regression: it iterated frozensets of str when joining each family's matched words into the
sub-query text. Iteration order varies per process, the dense embedding is word-order
sensitive, so "reception outfit for women" picked a different last item in ~1 of 3 processes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from src.agents.outfit.slots import split_accessory_query_by_family

_QUERY = "bag sling handbag earrings women accessory embellished formal evening"

_PROGRAM = (
    "import json,sys;"
    "from src.agents.outfit.slots import split_accessory_query_by_family as f;"
    f"print(json.dumps(f({_QUERY!r})))"
)


def test_words_within_a_family_are_in_sorted_order() -> None:
    parts = split_accessory_query_by_family(_QUERY)
    assert len(parts) == 2  # bag family + jewellery family
    for part in parts:
        head = [w for w in part.split() if w in {"bag", "sling", "handbag", "earrings"}]
        assert head == sorted(head)


def test_output_identical_across_hash_seeds() -> None:
    outs = set()
    for seed in ("0", "1", "2", "3", "4", "5", "6", "7"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _PROGRAM],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outs.add(res.stdout.strip().splitlines()[-1])
    assert len(outs) == 1, f"order depends on hash seed: {outs}"
    assert json.loads(outs.pop()) == split_accessory_query_by_family(_QUERY)
