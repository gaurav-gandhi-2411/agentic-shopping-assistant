# ONNX encoder equivalence report

Verdict: **PASS**

Rule: `evals/onnx_equivalence_rule.md`. Seed 42. Note: linux container from repo Dockerfile, image asa-onnx-bench:onnx-6dc78c9 (d45888d3d5af); torch/ST 5.7.0 installed ephemerally via pip --target for the torch reference only; Docker Desktop 29.6.1 (WSL2), --cpus 4

| check | n | stat | stat | stat | result |
|---|---|---|---|---|---|
| C1_minilm_vs_stored | sample 2000, reproduced by torch 2000 (excluded 0) | n=2000 min 0.9999999 | p1 0.9999999 | median 1.0000000 | PASS |
| &nbsp;&nbsp;C1 control_torch_vs_stored_all | n=2000 | min 0.9999999 | p1 0.9999999 | median 1.0000000 | info |
| &nbsp;&nbsp;C1 onnx_vs_stored_all_items | n=2000 | min 0.9999999 | p1 0.9999999 | median 1.0000000 | info |
| &nbsp;&nbsp;C1 onnx_vs_torch_all_items | n=2000 | min 0.9999999 | p1 0.9999999 | median 1.0000000 | info |
| C2_clip_text_vs_stored | sample 2000, reproduced by torch 1761 (excluded 239) | n=1761 min 0.9999999 | p1 0.9999999 | median 1.0000000 | PASS |
| &nbsp;&nbsp;C2 control_torch_vs_stored_all | n=2000 | min 0.9272062 | p1 0.9895080 | median 1.0000000 | info |
| &nbsp;&nbsp;C2 onnx_vs_stored_all_items | n=2000 | min 0.9272061 | p1 0.9895080 | median 1.0000000 | info |
| &nbsp;&nbsp;C2 onnx_vs_torch_all_items | n=2000 | min 0.9999999 | p1 0.9999999 | median 1.0000000 | info |
| C4_dense_query_vectors_cosine | n=60 | min 0.9999999 | p1 0.9999999 | median 1.0000000 | info |
| C4_dense_top10_overlap | n=60 | mean 1.0000 | min 1.00 | below-1: 0 | PASS |
| C4_clip_text_top10_overlap | n=60 | mean 1.0000 | min 1.00 | below-1: 0 | PASS |
| C3_clip_image_vs_torch | n=200 | min 0.9999999 | p1 0.9999999 | median 1.0000000 | PASS |
| C4_clip_image_top10_overlap | n=200 | mean 1.0000 | min 1.00 | below-1: 0 | PASS |

```json
{
  "platform": "Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.41",
  "python": "3.11.18",
  "cpu_count": 16,
  "onnxruntime": "1.32.0",
  "tokenizers": "0.22.2",
  "faiss": "1.15.1",
  "torch": "2.15.0+cpu",
  "transformers": "5.10.4",
  "models": {
    "all-MiniLM-L6-v2": {
      "export_versions": {
        "onnx": "1.23.2",
        "onnxruntime": "1.32.0",
        "opset": 17,
        "tokenizers": "0.22.2",
        "torch": "2.15.0+cpu",
        "transformers": "5.10.4"
      },
      "files": {
        "model.onnx": "3daa0d7afabb354c88ddeb99e9097def5faf87e4eec135138c3d74c60989a049",
        "model.onnx.data": "f088b7984e821e25d4ce5b42190cb1825f08dc455c91f2ea7ebfb565b04223fc",
        "tokenizer.json": "851ca67100d372ca3ae031a6abd168f53489eebfd7d89523f35c5c9b4d372c3c"
      },
      "max_seq_length": 256,
      "pooling": "mean",
      "repo": "sentence-transformers/all-MiniLM-L6-v2",
      "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    },
    "clip-ViT-B-32": {
      "export_versions": {
        "onnx": "1.23.2",
        "onnxruntime": "1.32.0",
        "opset": 17,
        "tokenizers": "0.22.2",
        "torch": "2.15.0+cpu",
        "transformers": "5.10.4"
      },
      "files": {
        "text.onnx": "f76894e822cd13e9cce5cb1441f6fb052398cda46f8a249e71d1ac865bf79b79",
        "text.onnx.data": "e6c180f1ef47ea1e4c311b9f775d890ba4c46b8e28e2c4cc30577d227f9fda8a",
        "tokenizer.json": "d656a0f8bc9d30a93ea9f2858ebf4f076b1ff914b8e79290ed9687721e7ac74a",
        "vision.onnx": "67ce3d320db996e5eedc613cac35f82fc92c9c27cb0103ddb1a0d15ccb2d3c08",
        "vision.onnx.data": "4e5a6a036de433823b2e81a24fabba8fead32998686e82955691e6aa32aa99cb"
      },
      "image_size": 224,
      "max_tokens": 77,
      "repo": "sentence-transformers/clip-ViT-B-32",
      "revision": "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"
    }
  }
}
```
