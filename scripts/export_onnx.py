"""Export the two retrieval encoders to fp32 ONNX + tokenizer.json (build-time, needs torch).

    pip install -r requirements-export.txt
    python scripts/export_onnx.py --out-dir onnx_models

Run by the Dockerfile's ``export`` stage (the only stage that has torch); the runtime stage copies
only the output directory. Also run by CI before the tests that load the real encoders.

Output layout (``ONNX_MODEL_DIR`` points at ``--out-dir`` at runtime)::

    all-MiniLM-L6-v2/model.onnx  tokenizer.json  manifest.json
    clip-ViT-B-32/text.onnx  vision.onnx  tokenizer.json  manifest.json

Models are fetched at the pinned Hugging Face commits below (the same repos the code used before
this change: config.yaml ``retrieval.dense_model`` and ``clip.model``). Bumping a revision is a
deliberate edit here and must be followed by ``scripts/onnx_equivalence_gate.py``.

fp32 only: no fp16, no quantisation (evals/onnx_equivalence_rule.md). Each export is self-checked
against the torch model on probe inputs and the script exits non-zero if cosine < 0.99999, so a
broken export fails the image build instead of shipping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Pinned Hugging Face commits (resolved 2026-11-26 from huggingface.co/api/models/<id>).
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CLIP_REPO = "sentence-transformers/clip-ViT-B-32"
CLIP_REVISION = "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"

OPSET = 17  # widely supported by onnxruntime 1.x; the legacy exporter handles BERT/CLIP at 17
SELF_CHECK_MIN_COSINE = 0.99999

PROBE_TEXTS = [
    "women red cotton kurta set festive",
    "men navy blue slim fit formal shirt for office " * 12,  # long: exercises truncation
    "a",
    "",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cos_min(a, b) -> float:  # noqa: ANN001
    import numpy as np

    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return float((a * b).sum(axis=-1).min())


def _write_manifest(out: Path, info: dict) -> None:
    import onnx
    import onnxruntime
    import tokenizers
    import torch
    import transformers

    info["export_versions"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "tokenizers": tokenizers.__version__,
        "opset": OPSET,
    }
    info["files"] = {p.name: _sha256(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def export_minilm(root: Path) -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    out = root / MINILM_REPO.rsplit("/", 1)[-1]
    out.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(MINILM_REPO, revision=MINILM_REVISION, device="cpu")
    st.eval()
    assert st.max_seq_length == 256, st.max_seq_length  # hard-coded in OnnxTextEncoder
    pooling = st[1].get_pooling_mode_str()
    assert pooling == "mean", pooling  # OnnxTextEncoder mean-pools

    bert = st[0].auto_model

    class _Wrap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = bert

        def forward(self, input_ids, attention_mask, token_type_ids):  # noqa: ANN001, ANN201
            return self.m(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )[0]

    enc = st.tokenizer(PROBE_TEXTS, padding=True, truncation=True, max_length=256,
                       return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    path = out / "model.onnx"
    torch.onnx.export(
        _Wrap().eval(), args, str(path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in
                      ("input_ids", "attention_mask", "token_type_ids", "last_hidden_state")},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    st.tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))

    # Self-check through the same code path production uses (tokenizers + ORT + numpy pooling).
    from src.retrieval.onnx_encoders import OnnxTextEncoder

    ref = st.encode(PROBE_TEXTS, normalize_embeddings=True, convert_to_numpy=True)
    got = OnnxTextEncoder(MINILM_REPO, root=root).encode(PROBE_TEXTS, normalize_embeddings=True)
    cos = _cos_min(np.asarray(ref), got)
    print(f"minilm self-check min cosine vs torch: {cos:.8f}")
    if cos < SELF_CHECK_MIN_COSINE:
        sys.exit(f"MiniLM export self-check failed: {cos} < {SELF_CHECK_MIN_COSINE}")
    _write_manifest(out, {"repo": MINILM_REPO, "revision": MINILM_REVISION,
                          "max_seq_length": 256, "pooling": pooling})


def export_clip(root: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    out = root / CLIP_REPO.rsplit("/", 1)[-1]
    out.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(CLIP_REPO, revision=CLIP_REVISION, device="cpu")
    st.eval()
    clip = st[0].model  # transformers CLIPModel (what sentence-transformers' CLIPModel wraps)
    proc = st[0].processor

    # Constants OnnxCLIPEncoder hard-codes must match the processor actually shipped.
    ip = proc.image_processor
    assert ip.crop_size["height"] == 224 and ip.size["shortest_edge"] == 224, (ip.size, ip.crop_size)
    assert np.allclose(ip.image_mean, [0.48145466, 0.4578275, 0.40821073]), ip.image_mean
    assert np.allclose(ip.image_std, [0.26862954, 0.26130258, 0.27577711]), ip.image_std

    class _Text(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = clip

        def forward(self, input_ids, attention_mask):  # noqa: ANN001, ANN201
            pooled = self.m.text_model(input_ids=input_ids, attention_mask=attention_mask)[1]
            return self.m.text_projection(pooled)

    class _Vision(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m = clip

        def forward(self, pixel_values):  # noqa: ANN001, ANN201
            return self.m.visual_projection(self.m.vision_model(pixel_values=pixel_values)[1])

    tok = proc.tokenizer
    enc = tok(PROBE_TEXTS, padding=True, truncation=True, max_length=77, return_tensors="pt")
    torch.onnx.export(
        _Text().eval(), (enc["input_ids"], enc["attention_mask"]), str(out / "text.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "text_embeds": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    pix = torch.randn(2, 3, 224, 224)
    torch.onnx.export(
        _Vision().eval(), (pix,), str(out / "vision.onnx"),
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))

    from src.retrieval.onnx_encoders import OnnxCLIPEncoder

    onnx_enc = OnnxCLIPEncoder(CLIP_REPO, root=root)
    cos_t = _cos_min(
        st.encode(PROBE_TEXTS, convert_to_numpy=True, normalize_embeddings=True),
        onnx_enc.encode_texts(PROBE_TEXTS),
    )
    rng = np.random.default_rng(42)
    imgs = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
            for h, w in ((300, 200), (200, 300), (224, 224))]
    ref_i = np.stack([st.encode(i, convert_to_numpy=True, normalize_embeddings=True) for i in imgs])
    got_i = np.stack([onnx_enc.encode_image(i) for i in imgs])
    cos_i = _cos_min(ref_i, got_i)
    print(f"clip self-check min cosine vs torch: text={cos_t:.8f} image={cos_i:.8f}")
    if min(cos_t, cos_i) < SELF_CHECK_MIN_COSINE:
        sys.exit(f"CLIP export self-check failed: text={cos_t} image={cos_i}")
    _write_manifest(out, {"repo": CLIP_REPO, "revision": CLIP_REVISION,
                          "max_tokens": 77, "image_size": 224})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=Path("onnx_models"))
    ap.add_argument("--only", choices=("minilm", "clip"), default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    out = args.out_dir.resolve()
    if args.only in (None, "minilm"):
        export_minilm(out)
    if args.only in (None, "clip"):
        export_clip(out)
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
