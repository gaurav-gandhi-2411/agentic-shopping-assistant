#!/usr/bin/env python
"""Derive a hygiene-applied COPY of a unified index without re-running the full build.

``scripts/build_unified_index.py`` applies :func:`src.catalogue.hygiene.apply_hygiene` per brand and
then re-embeds the merged catalogue. For before/after measurements on the SAME index we do the
equivalent offline: apply the same function to an existing unified catalogue, keep the dense
vectors of the surviving rows (the search_text is unchanged, so a re-embed would produce the same
vectors) and rebuild BM25 over the survivors (IDF is corpus-global, as in the real build).

Never touches ``--src``; refuses to write into an existing ``--dst`` (no deletions anywhere).
CLIP is not part of text retrieval and is not copied.

Usage:
    python scripts/make_hygiene_index_copy.py --src data/processed/unified \
        --dst data/processed/unified_hyg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.catalogue.hygiene import apply_hygiene  # noqa: E402
from src.catalogue.loader import load_config  # noqa: E402
from src.retrieval.sparse_search import SparseRetriever  # noqa: E402


def retype_quarantined(cat: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Stand-in for a rebuild with the fixed normalizer: re-type only the rows hygiene would
    quarantine, from their title, using src.catalogue.normalizer (the root-cause fix).

    A real build re-normalises every row from the raw store label; those labels are not in the
    built catalogue, so this touches only the rows whose stale type is the quarantine trigger
    (title-only, falls back to the existing type when the normalizer has no answer). Updates
    ``product_type_name`` and the ``product_type_name`` entry of ``facets``. Returns (frame, n
    rows whose type changed).
    """
    from src.catalogue.normalizer import normalize_garment_type

    _, report = apply_hygiene(cat)
    ids = set(report["quarantined"]["article_id"].astype(str))
    out = cat.copy()
    changed = 0
    for i in out.index[out["article_id"].astype(str).isin(ids)]:
        new = normalize_garment_type(str(out.at[i, "prod_name"]), None, None).garment_type
        if new and new != out.at[i, "product_type_name"]:
            out.at[i, "product_type_name"] = new
            facets = out.at[i, "facets"]
            if isinstance(facets, dict):
                out.at[i, "facets"] = {**facets, "product_type_name": new}
            changed += 1
    return out, changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--retype-quarantined", action="store_true",
                    help="re-type quarantined rows with the fixed normalizer first "
                         "(simulates a rebuild; see retype_quarantined)")
    args = ap.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if src == dst:
        raise SystemExit("--dst must differ from --src")
    if dst.exists():
        raise SystemExit(f"{dst} already exists; refusing to write into it")

    cat = pd.read_parquet(src / "catalogue.parquet")
    if args.retype_quarantined:
        cat, n_changed = retype_quarantined(cat)
        print(f"re-typed {n_changed} quarantined rows with the fixed normalizer")
    kept, report = apply_hygiene(cat)
    print(f"catalogue {len(cat)} -> {len(kept)}  dropped={len(report['dropped'])} "
          f"quarantined={len(report['quarantined'])}")

    index = faiss.read_index(str(src / "dense.faiss"))
    ids = np.load(str(src / "dense_article_ids.npy"), allow_pickle=True).astype(str)
    if index.ntotal != len(ids) or len(ids) != len(cat):
        raise SystemExit("src index/ids/catalogue lengths disagree")
    pos = {a: i for i, a in enumerate(ids)}
    keep_pos = np.array([pos[a] for a in kept["article_id"].astype(str)], dtype=np.int64)
    vecs = index.reconstruct_n(0, index.ntotal)[keep_pos]
    new_index = faiss.IndexFlatIP(index.d)
    new_index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    dst.mkdir(parents=True)
    faiss.write_index(new_index, str(dst / "dense.faiss"))
    np.save(str(dst / "dense_article_ids.npy"), kept["article_id"].values.astype(str))
    kept.to_parquet(str(dst / "catalogue.parquet"), index=False)
    SparseRetriever(load_config()).build_index(kept, dst)
    for key, name in (("dropped", "hygiene_dropped.csv"), ("quarantined", "hygiene_quarantine.csv")):
        report[key].to_csv(dst / name, index=False)
    print(f"wrote {dst}: dense={new_index.ntotal} bm25 docs={len(kept)}")


if __name__ == "__main__":
    main()
