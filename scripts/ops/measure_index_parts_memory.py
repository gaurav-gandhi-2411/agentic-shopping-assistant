"""RSS cost of each component of one snapshot (diagnostic for the memory report).
usage: measure_index_parts_memory.py <data_root> <repo_root> <stub_dir>"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import psutil

data_root, repo_root, stub_dir = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(stub_dir))
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)
os.environ.pop("INCLUDE_NON_PUBLIC_STORES", None)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.retrieval.dense_search import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_search import HybridRetriever  # noqa: E402
from src.retrieval.snapshot import public_catalogue  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402

proc = psutil.Process()
MIB = 1024**2
cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
u, c = data_root / "unified", data_root / "clip_unified"


def rss() -> float:
    gc.collect()
    return proc.memory_info().rss / MIB


last = rss()
print(f"baseline {last:.0f} MiB")


def step(label: str) -> None:
    global last
    now = rss()
    print(f"{label:<52} +{now - last:7.0f} MiB  (total {now:.0f})")
    last = now


df_full = pd.read_parquet(u / "catalogue.parquet")
step("read_parquet catalogue (112,425 rows, full)")
df = public_catalogue(df_full)
step("public_catalogue() copy (89,407 rows)")
del df_full
step("drop the full frame")


class M:
    pass


dense = DenseRetriever.load(cfg, u, model=M())
step("dense.faiss + dense ids (float32 112k x 384)")
sparse = SparseRetriever.load(cfg, u)
step("bm25.pkl + ids + id->pos dict")
retr = HybridRetriever(dense, sparse, df, cfg)
step("HybridRetriever init (set_index copy, kids mask, id->pos dict)")
clip_i = faiss.read_index(str(c / "clip.faiss"))
clip_ids = np.load(str(c / "clip_article_ids.npy"), allow_pickle=True)
step("clip.faiss + clip ids (float32 112k x 512)")
