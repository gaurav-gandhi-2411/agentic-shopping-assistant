"""
Build script: loads catalogue, builds dense FAISS + sparse BM25 indices.
Run once before starting the API or smoke tests.

Usage
-----
    # H&M (default):
    python scripts/01_build_retrieval.py

    # Indian brand feed:
    python scripts/01_build_retrieval.py --brand sample_in

    # Override sample size and output directory:
    python scripts/01_build_retrieval.py --brand sample_in --sample 0 --out data/processed/sample_in
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.catalogue.adapter import adapt_feed
from src.catalogue.loader import build_searchable_text, load_articles, load_config
from src.config.brand import BrandConfig, get_brand_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dense + sparse retrieval indices for a product catalogue.",
    )
    parser.add_argument(
        "--brand",
        default=os.environ.get("BRAND", "hm"),
        help="Brand slug that maps to brands/<brand>.yaml  (default: hm)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output directory for indices.  "
            "Defaults to data/processed/ for hm, data/processed/<brand>/ for others."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Number of catalogue items to sample.  "
            "0 means use all rows.  "
            "Defaults to catalogue.sample_num_items from config.yaml."
        ),
    )
    return parser.parse_args()


def _load_brand_df(brand: str, brand_config: BrandConfig, config: dict, sample: int | None) -> pd.DataFrame:
    """Load and adapt the catalogue for *brand*.

    H&M uses the existing :func:`load_articles` path so behaviour is unchanged.
    All other brands read :attr:`BrandConfig.catalogue_path` as a CSV and run
    :func:`adapt_feed`.
    """
    if brand == "hm":
        # Override sample_num_items when caller passes --sample
        if sample is not None:
            effective_config = dict(config)
            effective_config["catalogue"] = dict(config["catalogue"])
            if sample == 0:
                # Load without sampling by reading entire CSV then not sampling
                from pathlib import Path as _Path
                from src.catalogue.loader import KEEP_COLUMNS
                csv_path = _Path(effective_config["catalogue"]["articles_csv"])
                df = pd.read_csv(csv_path, usecols=KEEP_COLUMNS, dtype={"article_id": str})
                return df.dropna(subset=["detail_desc"]).reset_index(drop=True)
            else:
                effective_config["catalogue"]["sample_num_items"] = sample
            return load_articles(effective_config)
        return load_articles(config)

    # Generic brand path
    csv_path = Path(brand_config.catalogue_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Catalogue file not found for brand '{brand}': {csv_path}\n"
            f"Set catalogue_path in brands/{brand}.yaml to a valid CSV path."
        )

    print(f"Reading feed from {csv_path} …")
    df_raw = pd.read_csv(csv_path)
    print(f"Raw feed rows: {len(df_raw):,}  columns: {list(df_raw.columns)}")

    df = adapt_feed(df_raw, brand_config)
    df = df.dropna(subset=["detail_desc"]).reset_index(drop=True)

    if sample is not None and sample > 0:
        seed = config["catalogue"].get("seed", 42)
        df = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)

    return df


def _resolve_save_dir(args: argparse.Namespace, config: dict) -> Path:
    if args.out:
        return Path(args.out)
    if args.brand == "hm":
        return Path(config["catalogue"]["processed_dir"])
    return Path("data/processed") / args.brand


def _build_searchable_text_to_dir(df: pd.DataFrame, config: dict, save_dir: Path) -> pd.DataFrame:
    """Thin wrapper: runs build_searchable_text but saves to an arbitrary directory.

    loader.build_searchable_text() writes to config["catalogue"]["processed_dir"].
    For non-hm brands we monkey-patch that key so the parquet lands in save_dir.
    """
    patched_config = dict(config)
    patched_config["catalogue"] = dict(config["catalogue"])
    patched_config["catalogue"]["processed_dir"] = str(save_dir)
    return build_searchable_text(df, patched_config)


def main() -> None:
    args = _parse_args()

    config = load_config()

    print(f"Brand: {args.brand}")

    # Temporarily set BRAND env var so get_brand_config() picks the right file
    os.environ["BRAND"] = args.brand
    # get_brand_config is lru_cache'd; bypass cache for CLI use
    brand_config = BrandConfig.model_validate(
        __import__("yaml").safe_load(
            (Path(__file__).parent.parent / "brands" / f"{args.brand}.yaml").read_text()
        )
    )

    save_dir = _resolve_save_dir(args, config)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {save_dir}")

    print("Loading catalogue…")
    df = _load_brand_df(args.brand, brand_config, config, args.sample)
    print(f"Loaded catalogue: {len(df):,} articles")
    print(f"Unique product types: {df['product_type_name'].nunique()}")
    print(f"Unique colour groups: {df['colour_group_name'].nunique()}")

    df = _build_searchable_text_to_dir(df, config, save_dir)

    print("\n--- Sanity stats ---")
    print(f"Total rows:               {len(df):,}")
    print(f"Non-null detail_desc:     {df['detail_desc'].notna().sum():,}")

    # price_inr stats only for brands that carry pricing
    if "price_inr" in df.columns and df["price_inr"].notna().any():
        print(f"price_inr range:          ₹{df['price_inr'].min():.0f} – ₹{df['price_inr'].max():.0f}")

    print("\nTop 10 product_type_name by frequency:")
    for name, count in df["product_type_name"].value_counts().head(10).items():
        print(f"  {name:<35} {count:>5}")

    print("\nTop 10 colour_group_name by frequency:")
    for name, count in df["colour_group_name"].value_counts().head(10).items():
        print(f"  {name:<35} {count:>5}")

    print("\n--- Building dense index ---")
    t0 = time.time()
    dense = DenseRetriever(config)
    dense.build_index(df, save_dir)
    print(f"Dense index built in {time.time() - t0:.1f}s")

    print("\n--- Building sparse index ---")
    t0 = time.time()
    sparse = SparseRetriever(config)
    sparse.build_index(df, save_dir)
    print(f"Sparse index built in {time.time() - t0:.1f}s")

    print(f"\nAll indices saved to {save_dir}")


if __name__ == "__main__":
    main()
