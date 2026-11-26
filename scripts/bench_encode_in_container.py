"""Encode-latency probe, run INSIDE the API container (docker cp'd in by bench_container.py).

Backend-agnostic on purpose, so the same file measures the base image (sentence-transformers) and
the ONNX image: it uses ``src.retrieval.dense_search.load_text_encoder`` /
``get_clip_encoder`` when they exist (this branch) and falls back to the torch classes the base
commit has. Per-query latency of exactly what the request path calls for one query:
dense ``encode([q], normalize_embeddings=True)``, CLIP ``encode_texts([q])``, CLIP
``encode_image(img)``. Prints one JSON object.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.array(xs) * 1000.0, q))


def timed(fn, items, warm: int = 5) -> dict[str, float]:  # noqa: ANN001
    for it in items[:warm]:
        fn(it)
    ts = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        ts.append(time.perf_counter() - t0)
    return {"n": len(ts), "p50_ms": round(pct(ts, 50), 2), "p95_ms": round(pct(ts, 95), 2),
            "mean_ms": round(float(np.mean(ts)) * 1000, 2)}


def main() -> None:
    queries = json.load(open(sys.argv[1], encoding="utf-8"))
    enc: Any
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    out: dict = {"n_queries": len(queries)}
    t0 = time.perf_counter()
    try:
        from src.retrieval.dense_search import load_text_encoder
        enc = load_text_encoder(model_name)
        out["backend"] = type(enc).__name__
    except ImportError:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model_name, device="cpu")
        out["backend"] = "SentenceTransformer"
    out["dense_load_s"] = round(time.perf_counter() - t0, 2)
    out["dense_encode"] = timed(lambda q: enc.encode([q], normalize_embeddings=True,
                                                     show_progress_bar=False), queries)

    from src.retrieval.clip_encoder import get_clip_encoder
    t0 = time.perf_counter()
    clip = get_clip_encoder("clip-ViT-B-32")
    out["clip_load_s"] = round(time.perf_counter() - t0, 2)
    out["clip_text_encode"] = timed(lambda q: clip.encode_texts([q]), queries)
    rng = np.random.default_rng(42)
    imgs = [Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(30)]
    out["clip_image_encode_640x480"] = timed(clip.encode_image, imgs, warm=3)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
