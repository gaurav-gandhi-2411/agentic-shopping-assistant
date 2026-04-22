"""
Build script: loads catalogue, builds retrieval indices.
Phase 1: catalogue loading and parquet save.
Phase 2: dense + sparse index building (stubs filled in later).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_articles, build_searchable_text, load_config


def main():
    config = load_config()

    print("Loading catalogue...")
    df = load_articles(config)
    print(f"Loaded catalogue: {len(df):,} articles")
    print(f"Unique product types: {df['product_type_name'].nunique()}")
    print(f"Unique colour groups: {df['colour_group_name'].nunique()}")

    df = build_searchable_text(df, config)

    # Extended sanity stats
    print(f"\n--- Sanity stats ---")
    print(f"Total rows:               {len(df):,}")
    print(f"Non-null detail_desc:     {df['detail_desc'].notna().sum():,}")

    print(f"\nTop 10 product_type_name by frequency:")
    for name, count in df["product_type_name"].value_counts().head(10).items():
        print(f"  {name:<35} {count:>5}")

    print(f"\nTop 10 colour_group_name by frequency:")
    for name, count in df["colour_group_name"].value_counts().head(10).items():
        print(f"  {name:<35} {count:>5}")


if __name__ == "__main__":
    main()
