"""Parity of the REAL exported ONNX encoders against vectors STORED in the production index.

Fixture: tests/fixtures/onnx_parity/stored_vectors.npz, 20 items drawn with the gate's seed-42
sample (scripts/onnx_equivalence_gate.py --write-fixture): the catalogue ``search_text`` and the
MiniLM / CLIP-text vectors read out of ``dense.faiss`` / ``clip.faiss``. Bar: min cosine >= 0.9999
(evals/onnx_equivalence_rule.md). Needs the exported models (``python scripts/export_onnx.py``);
CI exports them first and sets REQUIRE_ONNX_MODELS=1 so a missing export fails instead of skipping.
No torch is imported here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.retrieval.onnx_encoders import (
    OnnxCLIPEncoder,
    OnnxTextEncoder,
    onnx_model_dir,
)

FIXTURE = Path(__file__).parent / "fixtures" / "onnx_parity" / "stored_vectors.npz"
MIN_COS = 0.9999
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CLIP = "clip-ViT-B-32"


def _models_present() -> bool:
    return (onnx_model_dir(MINILM) / "model.onnx").is_file() and (
        onnx_model_dir(CLIP) / "vision.onnx"
    ).is_file()


@pytest.fixture(scope="module")
def stored() -> dict[str, np.ndarray]:
    if not _models_present():
        if os.environ.get("REQUIRE_ONNX_MODELS") == "1":
            pytest.fail("ONNX models not exported but REQUIRE_ONNX_MODELS=1 (see scripts/export_onnx.py)")
        pytest.skip("ONNX models not exported; run python scripts/export_onnx.py")
    return dict(np.load(FIXTURE, allow_pickle=False))


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_fixture_is_the_expected_shape(stored: dict[str, np.ndarray]) -> None:
    assert stored["minilm_stored"].shape == (20, 384)
    assert stored["clip_stored"].shape == (20, 512)
    assert len(stored["texts"]) == len(stored["clip_texts"]) == 20


def test_minilm_matches_stored_dense_vectors(stored: dict[str, np.ndarray]) -> None:
    got = OnnxTextEncoder(MINILM).encode(
        [str(t) for t in stored["texts"]], normalize_embeddings=True
    )
    c = _cos(got, stored["minilm_stored"])
    assert c.min() >= MIN_COS, f"min cosine {c.min():.7f}"


def test_clip_text_matches_stored_clip_vectors(stored: dict[str, np.ndarray]) -> None:
    got = OnnxCLIPEncoder(CLIP).encode_texts([str(t) for t in stored["clip_texts"]])
    c = _cos(got, stored["clip_stored"])
    assert c.min() >= MIN_COS, f"min cosine {c.min():.7f}"
