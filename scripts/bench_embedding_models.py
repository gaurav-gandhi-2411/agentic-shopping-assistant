#!/usr/bin/env python
"""Embedding-model A/B benchmark against the strict hand-labeled gold set.

Builds a throwaway dense FAISS index for a candidate model over the FULL
production catalogue (fair comparison — precision@5 depends on the whole
competing pool, not just the labeled items), reuses the EXISTING sparse/BM25
index unchanged (isolates the dense-model variable), runs strict eval's raw
retrieval mode against the resulting HybridRetriever, and separately measures
single-query CPU encode latency (production Cloud Run has no GPU — index
BUILDING may use GPU if available for speed, but the reported latency number
is always CPU, matching what a live query actually costs).

Never touches the production index. Free/open-source only — every candidate
is self-hostable via sentence-transformers, $0 cost.

Usage:
    python scripts/bench_embedding_models.py --model BAAI/bge-base-en-v1.5
    python scripts/bench_embedding_models.py --model BAAI/bge-base-en-v1.5 --build-device cuda
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

_ROOT = Path(__file__).parent.parent
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_candidate_dense_index(
    model_name: str, catalogue_df: pd.DataFrame, out_dir: Path, build_device: str
) -> tuple[float, int]:
    """Encode the full catalogue with `model_name`, save a FAISS index to
    out_dir. Returns (build_seconds, embedding_dim)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=build_device)
    texts = catalogue_df["search_text"].tolist()

    t0 = time.time()
    embeddings = model.encode(
        texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)
    build_s = time.time() - t0

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "dense.faiss"))
    np.save(str(out_dir / "dense_article_ids.npy"), catalogue_df["article_id"].values.astype(str))
    return build_s, dim


def measure_query_latency_cpu(model_name: str, sample_queries: list[str], n_repeats: int = 3) -> dict:
    """Single-query CPU encode latency — matches production (Cloud Run, no GPU)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    # Warm-up (first call pays one-time graph/cache setup cost).
    model.encode([sample_queries[0]], normalize_embeddings=True, show_progress_bar=False)

    latencies_ms: list[float] = []
    for _ in range(n_repeats):
        for q in sample_queries:
            t0 = time.time()
            model.encode([q], normalize_embeddings=True, show_progress_bar=False)
            latencies_ms.append((time.time() - t0) * 1000)
    arr = np.array(latencies_ms)
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95)), "n": len(arr)}


def run_strict_raw_eval(retriever, top_k: int = 5) -> dict:
    """Mirror eval_strict.py --mode raw exactly (same labels, same rubric) —
    isolates PURE retrieval quality (no filters/gates) so the dense-model
    variable is measured cleanly, without the search-path fixes convolving
    the result."""
    import yaml

    queries = yaml.safe_load(
        (_ROOT / "eval" / "fixtures" / "strict_gold_queries.yaml").read_text(encoding="utf-8")
    )["queries"]
    labels_raw = yaml.safe_load(
        (_ROOT / "eval" / "fixtures" / "strict_gold_labels.yaml").read_text(encoding="utf-8")
    )["labels"]
    labels = {
        (e["query_id"], str(it["article_id"])): it
        for e in labels_raw for it in e["items"]
    }

    n_scored = n_relevant = n_unlabeled = 0
    by_category: dict[str, list[int]] = {}
    for q in queries:
        items = retriever.search(q["query"], top_k=50, filters={"gender": q["gender"]})[:top_k]
        cat = q.get("category", "uncategorized")
        by_category.setdefault(cat, [0, 0, 0])
        for it in items:
            key = (q["id"], str(it.get("article_id")))
            label = labels.get(key)
            if label is None:
                n_unlabeled += 1
                by_category[cat][2] += 1
                continue
            n_scored += 1
            if label["relevant"]:
                n_relevant += 1
                by_category[cat][0] += 1
            else:
                by_category[cat][1] += 1

    return {
        "precision_at_5": n_relevant / n_scored if n_scored else 0.0,
        "n_scored": n_scored,
        "n_unlabeled": n_unlabeled,
        "by_category": {
            c: (v[0] / (v[0] + v[1]) if (v[0] + v[1]) else 0.0, v[0] + v[1], v[2])
            for c, v in by_category.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="sentence-transformers model name/path")
    parser.add_argument("--build-device", default="cpu", choices=("cpu", "cuda"),
                        help="Device for the ONE-TIME index build (query latency is always CPU)")
    parser.add_argument("--data-dir", default=str(_ROOT / "data" / "processed" / "unified"))
    args = parser.parse_args()

    from eval_model import _build_components  # noqa: E402

    from src.retrieval.dense_search import DenseRetriever  # noqa: E402
    from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402

    comps = _build_components(need_agent=False, data_dir=Path(args.data_dir))
    baseline_retriever = comps["retriever"]
    catalogue_df = comps["catalogue_df"]
    config = baseline_retriever.dense.config

    print(f"\n=== Benchmarking {args.model} ===")
    safe_name = args.model.replace("/", "__")
    out_dir = Path(args.data_dir).parent / f"bench_{safe_name}"
    build_s, dim = build_candidate_dense_index(args.model, catalogue_df, out_dir, args.build_device)
    print(f"build: {build_s:.1f}s on {args.build_device}, dim={dim}, n={len(catalogue_df)}")

    candidate_dense = DenseRetriever.__new__(DenseRetriever)
    candidate_dense.config = config
    import faiss
    from sentence_transformers import SentenceTransformer
    candidate_dense.model = SentenceTransformer(args.model, device="cpu")  # query-time = CPU
    candidate_dense.index = faiss.read_index(str(out_dir / "dense.faiss"))
    candidate_dense.article_ids = np.load(str(out_dir / "dense_article_ids.npy"), allow_pickle=True)

    candidate_retriever = HybridRetriever(
        dense=candidate_dense, sparse=baseline_retriever.sparse,
        catalogue_df=catalogue_df, config=config,
    )

    result = run_strict_raw_eval(candidate_retriever)
    print(f"\nstrict RAW precision@5: {result['precision_at_5']:.3f}  "
          f"({result['n_scored']} scored, {result['n_unlabeled']} unlabeled)")
    print("by category:")
    for cat, (p5, n, unl) in sorted(result["by_category"].items()):
        print(f"  {cat:16s} {p5:.3f}  (n={n}, unlabeled={unl})")

    sample_queries = [
        "red saree for a wedding", "navy blue slim fit blazer for office under 5000",
        "bodycon dress for women", "sherwani for groom",
    ]
    lat = measure_query_latency_cpu(args.model, sample_queries)
    print(f"\nquery latency (CPU, n={lat['n']}): mean={lat['mean_ms']:.1f}ms  p95={lat['p95_ms']:.1f}ms")


if __name__ == "__main__":
    main()
