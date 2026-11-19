"""Tests that pin behaviour previously unguarded (surviving mutants of the stage-1 review)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from evals.retrieval import judge as jd
from evals.retrieval import metrics as mt
from evals.retrieval import packets as pk
from evals.retrieval import run

CASE = {
    "id": "q99",
    "text": "men's black kurta set under ₹3000",
    "constraints": {"gender": "men", "colour": "black", "category": "kurta set", "max_price": 3000},
}


def _cands(n: int = 20) -> list[dict]:
    return [
        {
            "article_id": f"a{i}",
            "display_name": f"Black Festive Kurta Number {i}",
            "store": "SECRETSTORE",
            "gender": "zzgenderzz",
            "department": "zzdeptzz",
            "price_inr": 1000.0 + i,
            "product_type": "Kurta set",
            "colour": "Black",
            "detail_desc": f"Soft cotton piece {i}.",
            "rank": i + 1,
            "rrf_score": 0.9,
        }
        for i in range(n)
    ]


def _prompts(n_q: int = 3) -> list[jd.JudgePrompt]:
    return [
        jd.build_judge_prompt({**CASE, "id": f"q{i:02d}"}, _cands(), 42) for i in range(1, n_q + 1)
    ]


def test_manifest_order_matches_numbering_printed_in_packet_text(tmp_path: Path) -> None:
    prompts = _prompts(7)
    paths = pk.export_packets(prompts, tmp_path, "main")
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    for path in paths:
        pid = path.stem
        blocks = re.split(r"=== QUERY (q\d+) ===\n", path.read_text("utf-8"))[1:]
        for qid, body in zip(blocks[::2], blocks[1::2], strict=True):
            order = manifest[pid]["orders"][qid]
            printed = {
                int(m.group(1)): f"a{int(m.group(2))}"
                for m in re.finditer(r"^(\d+)\. Black Festive Kurta Number (\d+) ", body, re.M)
            }
            assert len(printed) == 20
            assert [printed[i] for i in range(1, 21)] == order  # number n IS order[n-1]
            assert order != [f"a{i}" for i in range(20)]  # and the shuffle is real


def test_packet_text_has_no_variant_rank_or_store_hints(tmp_path: Path) -> None:
    (path,) = pk.export_packets(_prompts(2), tmp_path, "main")
    text = path.read_text("utf-8")
    header = text.split("=== QUERY")[0]
    for banned in ("variant", "rerank", "rrf", "rank", "retriev", "system a", "baseline"):
        assert banned not in header.lower(), banned
    body = text.lower()
    assert "rrf" not in body and "secretstore" not in body
    assert not re.search(r"\brank", body.split("rubric", 1)[1].split("=== query", 1)[1])


def test_gender_and_department_columns_are_never_shown_to_judges(tmp_path: Path) -> None:
    p = _prompts(1)[0]
    assert "zzgenderzz" not in p.user and "zzdeptzz" not in p.user
    (path,) = pk.export_packets([p], tmp_path, "main")
    assert "zzgenderzz" not in path.read_text("utf-8")
    # only the query-side constraint (what the shopper asked for) mentions gender
    assert p.user.count("gender:") == 1


def test_bootstrap_ci_is_a_95_percent_interval() -> None:
    vals = [1.0] * 50 + [0.0] * 50  # p=.5, n=100: SE = 0.05 -> 95% half-width ~ 0.098
    mean, lo, hi = mt.bootstrap_mean_ci(vals)
    assert mean == pytest.approx(0.5)
    assert 0.085 < mean - lo < 0.11 and 0.085 < hi - mean < 0.11  # 90% would be ~0.082, 99% ~0.13


def test_ndcg_ideal_comes_from_the_pool_not_from_the_shown_list() -> None:
    pool = [2, 2, 2, 1, 1, 0]
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
    ideal += 1 / math.log2(6)
    shown = [1, 1, 0, 0, 0]  # sorted-shown would look "ideal" (=1.0); the pool says it is not
    expected = (1 / math.log2(2) + 1 / math.log2(3)) / ideal
    assert mt.ndcg_at_k(shown, pool) == pytest.approx(expected)
    assert mt.ndcg_at_k(shown, pool) < 0.4


def test_wilcoxon_tie_correction_hand_value() -> None:
    # |d| = 1,1,2,2,3,3 (all positive): ranks 1.5,1.5,3.5,3.5,5.5,5.5, W+ = 21, mean = 10.5,
    # var = 6*7*13/24 - tie_term/48 with tie_term = 3 * (2^3 - 2) = 18 -> 22.75 - 0.375
    z = (21 - 10.5 - 0.5) / math.sqrt(22.75 - 18 / 48)
    assert mt.wilcoxon_p([1, 1, 2, 2, 3, 3]) == pytest.approx(math.erfc(z / math.sqrt(2)))
    assert mt.wilcoxon_p([1, 1, 2, 2, 3, 3]) != pytest.approx(
        math.erfc((10.0 / math.sqrt(22.75)) / math.sqrt(2))
    )


def test_usage_summary_excludes_no_llm_call_rows() -> None:
    rows = [
        {"outcome": "ok", "prompt_tokens": 1000, "completion_tokens": 100, "latency_ms": 900},
        {"outcome": "ok", "prompt_tokens": 2000, "completion_tokens": 200, "latency_ms": 1100},
        {"outcome": "no_llm_call", "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0},
    ]
    s = mt.usage_summary(rows)
    assert s["n"] == 2 and s["n_no_llm_call"] == 1
    assert s["mean_prompt_tokens"] == 1500 and s["mean_completion_tokens"] == 150
    assert s["latency_p50_ms"] == 1000
    assert mt.usage_summary([{"outcome": "no_llm_call"}]) == {"n": 0, "n_no_llm_call": 1}


def test_report_warns_when_fewer_than_three_judges_contribute() -> None:
    two = {"gpt-oss-120b": {"q1:a": 2}, "qwen3.8-27b": {"q1:a": 2}}
    (w,) = run.judge_warnings(two)
    assert "ONLY 2 of 3" in w
    assert run.judge_warnings({**two, "claude": {"q1:a": 1}}) == []
    assert "ONLY 2 of 3" in run.judge_warnings({**two, "claude": {}})[0]  # empty != contributed
