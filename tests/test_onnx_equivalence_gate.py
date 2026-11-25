"""Unit tests for the pure helpers of scripts/onnx_equivalence_gate.py (no models, no index)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import faiss
import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "onnx_equivalence_gate", Path(__file__).parents[1] / "scripts" / "onnx_equivalence_gate.py"
)
gate = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
sys.modules["onnx_equivalence_gate"] = gate
_SPEC.loader.exec_module(gate)  # type: ignore[union-attr]


def test_cosines_ignores_scale_and_detects_orthogonal() -> None:
    a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    b = np.array([[5.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(gate.cosines(a, b), [1.0, 0.0], atol=1e-7)


def test_cos_stats_min_p1_median() -> None:
    s = gate.cos_stats(np.array([1.0, 0.9, 0.8, 1.0, 1.0]))
    assert s["n"] == 5 and s["min"] == 0.8 and s["median"] == 1.0 and s["p1"] < 0.81


def test_sample_positions_is_seeded_and_unique() -> None:
    a, b = gate.sample_positions(1000, 50), gate.sample_positions(1000, 50)
    assert (a == b).all() and len(set(a.tolist())) == 50
    assert len(gate.sample_positions(10, 50)) == 10  # never more than the population


def test_topk_overlap_identical_and_perturbed() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 16)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(16)
    index.add(base)
    ids = np.array([f"a{i}" for i in range(500)])
    q = base[:5]
    same = gate.overlap_stats(gate.topk_ids(index, ids, q), gate.topk_ids(index, ids, q))
    assert same["mean"] == 1.0 and same["n_below_1"] == 0 and same["n"] == 5
    far = gate.overlap_stats(gate.topk_ids(index, ids, q), gate.topk_ids(index, ids, base[100:105]))
    assert far["mean"] < 0.5 and far["n_below_1"] == 5


def test_load_queries_reads_text_field_in_order(tmp_path: Path) -> None:
    (tmp_path / "q02.json").write_text('{"text": "second"}', encoding="utf-8")
    (tmp_path / "q01.json").write_text('{"text": "first"}', encoding="utf-8")
    (tmp_path / "other.json").write_text('{"text": "ignored"}', encoding="utf-8")
    assert gate.load_queries(tmp_path) == ["first", "second"]


def test_bars_are_the_pre_registered_numbers() -> None:
    import re

    assert gate.MIN_COS == 0.9999 and gate.MIN_OVERLAP == 0.99
    rule = (Path(__file__).parents[1] / "evals" / "onnx_equivalence_rule.md").read_text(
        encoding="utf-8"
    )
    bars = {float(x) for x in re.findall(r"^- C\d.*?>=\s*(\d+\.\d+)", rule, flags=re.M | re.S)}
    assert bars == {gate.MIN_COS, gate.MIN_OVERLAP}


def test_pass_flag_flips_at_the_cosine_bar() -> None:
    stored = np.array([[1.0, 0.0]] * 4, dtype=np.float32)

    def with_cos(c: float) -> dict:
        v = np.array([[c, np.sqrt(1 - c * c)]] * 4, dtype=np.float32)
        return gate.controlled_stats(v, stored.copy(), stored)

    assert with_cos(0.99995)["pass"] is True
    assert with_cos(0.99985)["pass"] is False  # below 0.9999


def _unit(rows: np.ndarray) -> np.ndarray:
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def test_controlled_stats_excludes_only_items_torch_cannot_reproduce() -> None:
    rng = np.random.default_rng(1)
    stored = _unit(rng.normal(size=(10, 8)))
    torch_v = stored.copy()
    torch_v[:3] = _unit(rng.normal(size=(3, 8)))  # 3 items whose indexed text we cannot rebuild
    onnx_v = torch_v + 1e-7
    s = gate.controlled_stats(onnx_v, torch_v, stored)
    assert (s["n_sample"], s["n_reproduced_by_torch"], s["n_excluded"]) == (10, 7, 3)
    assert s["pass"] and s["valid"] and s["min"] > 0.99999
    # the exclusion must not hide a failure: full-sample numbers are reported and low
    assert s["onnx_vs_stored_all_items"]["min"] < 0.99


def test_controlled_stats_fails_when_onnx_is_wrong_on_a_reproduced_item() -> None:
    rng = np.random.default_rng(2)
    stored = _unit(rng.normal(size=(10, 8)))
    onnx_v = stored.copy()
    onnx_v[5] = _unit(rng.normal(size=(1, 8)))[0]
    s = gate.controlled_stats(onnx_v, stored.copy(), stored)
    assert s["n_excluded"] == 0 and not s["pass"]


def test_controlled_stats_invalid_when_torch_reproduces_under_half() -> None:
    rng = np.random.default_rng(3)
    stored = _unit(rng.normal(size=(10, 8)))
    torch_v = _unit(rng.normal(size=(10, 8)))
    torch_v[:2] = stored[:2]
    s = gate.controlled_stats(torch_v, torch_v, stored)
    assert s["n_reproduced_by_torch"] == 2 and s["valid"] is False


def test_brand_texts_reads_only_brands_with_a_clip_index(tmp_path: Path) -> None:
    import pandas as pd

    for brand, with_clip in (("a", True), ("b", False), ("unified", True)):
        (tmp_path / brand).mkdir()
        pd.DataFrame({"article_id": [1, 2], "search_text": [f"{brand} x", None]}).to_parquet(
            tmp_path / brand / "catalogue.parquet"
        )
        if with_clip:
            (tmp_path / "clip" / brand).mkdir(parents=True)
    got = gate.brand_texts(tmp_path)
    assert got == {"1": "a x", "2": ""}  # b has no clip dir; unified is excluded by name
