"""Regenerate the tiny ONNX test fixtures (needs `onnx`, i.e. requirements-export.txt).

    python tests/fixtures/onnx_tiny/make_fixture.py

Writes ``text_model.onnx`` (embedding lookup -> last_hidden_state, MiniLM-shaped I/O),
``clip_text.onnx`` (lookup + mean over tokens), ``clip_vision.onnx`` (mean over H and W) and
``tables.npz`` (the lookup tables, so the tests can compute the expected vectors in numpy).
These are a few KB, not real models. They exist so the encoder classes (tokenizers + onnxruntime +
pooling + normalisation) are unit-tested in CI without torch and without the real weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).parent
VOCAB = 16
DIM = 4


def _model(graph: onnx.GraphProto, path: Path) -> None:
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def _int_in(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "seq"])


def main() -> None:
    rng = np.random.default_rng(42)
    table = rng.normal(size=(VOCAB, DIM)).astype(np.float32)
    clip_table = rng.normal(size=(VOCAB, DIM)).astype(np.float32)
    np.savez(OUT / "tables.npz", text=table, clip=clip_table)

    # MiniLM-shaped: three int inputs -> (batch, seq, DIM) hidden states (pooling is done outside)
    g = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"], axis=0)],
        "text",
        [_int_in("input_ids"), _int_in("attention_mask"), _int_in("token_type_ids")],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT,
                                       ["batch", "seq", DIM])],
        [numpy_helper.from_array(table, "table")],
    )
    _model(g, OUT / "text_model.onnx")

    # CLIP text: lookup + mean over the token axis -> (batch, DIM)
    g = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["emb"], axis=0),
         helper.make_node("ReduceMean", ["emb"], ["text_embeds"], axes=[1], keepdims=0)],
        "clip_text",
        [_int_in("input_ids"), _int_in("attention_mask")],
        [helper.make_tensor_value_info("text_embeds", TensorProto.FLOAT, ["batch", DIM])],
        [numpy_helper.from_array(clip_table, "table")],
    )
    _model(g, OUT / "clip_text.onnx")

    # CLIP vision: mean over H and W -> (batch, 3), i.e. the per-channel mean of the pixels
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["image_embeds"], axes=[2, 3],
                          keepdims=0)],
        "clip_vision",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT,
                                       ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["batch", 3])],
    )
    _model(g, OUT / "clip_vision.onnx")


if __name__ == "__main__":
    main()
