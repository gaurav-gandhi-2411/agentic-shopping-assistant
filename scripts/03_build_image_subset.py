"""
Build the Space image subset:
  1. Select top-1800 items by purchase count (transactions_train.csv)
  2. Resize images to 300px wide, JPEG quality 75
  3. Save catalogue_space.parquet with image_url column
  4. Build dense_space.faiss + bm25_space.pkl indices
  5. Print sanity stats

Run from repo root:
    python scripts/03_build_image_subset.py
"""
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever

SUBSET_SIZE = 1800
IMAGE_WIDTH = 300
JPEG_QUALITY = 75


def select_subset(catalogue_df: pd.DataFrame, tx_path: Path, img_root: Path) -> pd.DataFrame:
    print("Loading transactions (this takes ~15s)...")
    t0 = time.time()
    tx = pd.read_csv(tx_path, usecols=["article_id"], dtype={"article_id": str})
    counts = tx["article_id"].value_counts().rename("purchase_count")
    print(f"  Loaded {len(tx):,} transactions for {len(counts):,} unique articles in {time.time()-t0:.0f}s")

    merged = catalogue_df.merge(counts, on="article_id", how="inner")
    print(f"  Catalogue items in transactions: {len(merged):,}")

    has_image = merged["article_id"].apply(
        lambda aid: (img_root / aid[:3] / f"{aid}.jpg").exists()
    )
    merged = merged[has_image].copy()
    print(f"  With image on disk: {len(merged):,}")

    subset = merged.sort_values("purchase_count", ascending=False).head(SUBSET_SIZE).copy()
    subset = subset.reset_index(drop=True)
    return subset


def resize_images(subset_df: pd.DataFrame, img_src: Path, img_dst: Path) -> None:
    img_dst.mkdir(parents=True, exist_ok=True)
    errors = 0
    for aid in subset_df["article_id"]:
        src = img_src / aid[:3] / f"{aid}.jpg"
        dst = img_dst / aid[:3] / f"{aid}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((IMAGE_WIDTH, 9999), Image.LANCZOS)
                im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
        except Exception as e:
            print(f"  WARNING: failed to resize {aid}: {e}")
            errors += 1
    if errors:
        print(f"  {errors} images failed — check source files")


def folder_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main():
    config = load_config()
    repo_root = Path(__file__).parent.parent
    processed = repo_root / config["catalogue"]["processed_dir"]
    img_src = repo_root / "data" / "hm" / "images"
    img_dst = processed / "images"
    tx_path = repo_root / "data" / "hm" / "transactions_train.csv"

    print("=== Step 1: select subset ===")
    full_df = pd.read_parquet(processed / "catalogue.parquet")
    print(f"Full catalogue: {len(full_df):,} items")

    subset = select_subset(full_df, tx_path, img_src)
    print(f"Subset size: {len(subset):,} items")

    print("\n=== Step 2: resize images ===")
    print(f"Resizing to {IMAGE_WIDTH}px wide, JPEG q{JPEG_QUALITY}...")
    t0 = time.time()
    resize_images(subset, img_src, img_dst)
    elapsed = time.time() - t0
    size_mb = folder_size_mb(img_dst)
    print(f"Done in {elapsed:.0f}s — images/ folder: {size_mb:.1f} MB")

    print("\n=== Step 3: save catalogue_space.parquet ===")
    subset["image_url"] = subset["article_id"].apply(
        lambda aid: f"images/{aid[:3]}/{aid}.jpg"
    )
    subset = subset.drop(columns=["purchase_count"])
    space_parquet = processed / "catalogue_space.parquet"
    subset.to_parquet(space_parquet, index=False)
    print(f"Saved {space_parquet.name}: {len(subset):,} rows, {len(subset.columns)} columns")

    print("\n=== Step 4: rebuild indices for Space ===")
    tmp = processed / "_space_tmp"
    tmp.mkdir(exist_ok=True)

    print("Building dense index...")
    t0 = time.time()
    dense = DenseRetriever(config)
    dense.build_index(subset, tmp)
    print(f"  Dense: {time.time()-t0:.1f}s")

    print("Building BM25 index...")
    t0 = time.time()
    sparse = SparseRetriever(config)
    sparse.build_index(subset, tmp)
    print(f"  BM25: {time.time()-t0:.1f}s")

    for src_name, dst_name in [
        ("dense.faiss",          "dense_space.faiss"),
        ("dense_article_ids.npy","dense_space_article_ids.npy"),
        ("bm25.pkl",             "bm25_space.pkl"),
        ("bm25_article_ids.npy", "bm25_space_article_ids.npy"),
    ]:
        (tmp / src_name).rename(processed / dst_name)
    shutil.rmtree(tmp)
    print("Space indices saved with _space suffix.")

    print("\n=== Sanity stats ===")
    merged_for_stats = full_df.merge(
        pd.read_csv(tx_path, usecols=["article_id"], dtype={"article_id": str})
        ["article_id"].value_counts().rename("purchase_count"),
        on="article_id", how="inner"
    )
    top_n = merged_for_stats.sort_values("purchase_count", ascending=False).head(SUBSET_SIZE)

    print(f"\nTotal subset size:       {len(subset):,} items")
    print(f"images/ folder size:     {size_mb:.1f} MB")

    print(f"\nPurchase count distribution in subset:")
    pc = top_n["purchase_count"]
    print(f"  min={pc.min():,}  max={pc.max():,}  mean={pc.mean():.0f}")

    print(f"\nTop 5 product_type_name in subset:")
    for pt, cnt in subset["product_type_name"].value_counts().head(5).items():
        print(f"  {pt:<35} {cnt:>4}")

    print("\nImage spot-check (3 random articles):")
    random.seed(99)
    for aid in random.sample(list(subset["article_id"]), 3):
        path = img_dst / aid[:3] / f"{aid}.jpg"
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                print(f"  {aid}: {path.name}  {im.size}  {path.stat().st_size//1024} KB  OK")
        except Exception as e:
            print(f"  {aid}: FAILED — {e}")

    print("\nDone. Run upload_artifacts.py --space to push to HF Space.")


if __name__ == "__main__":
    main()
