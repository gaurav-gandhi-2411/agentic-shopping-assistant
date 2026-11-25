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
