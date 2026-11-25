# ONNX encoder equivalence report

Verdict: **PASS**

Rule: `evals/onnx_equivalence_rule.md`. Seed 42. Note: linux container from repo Dockerfile, image asa-onnx-bench:onnx-6b9914d (4a94d8a55342); torch/ST 5.7.0 installed ephemerally via pip --target for the torch reference only; Docker Desktop 29.6.1 (WSL2), --cpus 4

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
        "model.onnx": "9e026acac817c9a73d81767987429abafe1afafb8335eaddcc172c483eed716c",
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
        "text.onnx": "97af1b31dc3dcf702a814aa18e71893f293cb393f9cf69a44544ee214c2e80fe",
        "tokenizer.json": "d656a0f8bc9d30a93ea9f2858ebf4f076b1ff914b8e79290ed9687721e7ac74a",
        "vision.onnx": "bd49c07715cd56dc2eeab76a10a28b7b1526ecb1408f820aef17b5a0e9fb9225"
      },
      "image_size": 224,
      "max_tokens": 77,
      "repo": "sentence-transformers/clip-ViT-B-32",
      "revision": "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"
    }
  }
}
```
