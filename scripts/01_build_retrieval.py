"""
Build script: loads catalogue, builds dense FAISS + sparse BM25 indices.
Run once before starting the API or smoke tests.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalogue.loader import load_articles, build_searchable_text, load_config
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever


def main():
    config = load_config()
    save_dir = Path(config["catalogue"]["processed_dir"])

    print("Loading catalogue...")
    df = load_articles(config)
    print(f"Loaded catalogue: {len(df):,} articles")
    print(f"Unique product types: {df['product_type_name'].nunique()}")
    print(f"Unique colour groups: {df['colour_group_name'].nunique()}")

    df = build_searchable_text(df, config)

    print(f"\n--- Sanity stats ---")
    print(f"Total rows:               {len(df):,}")
    print(f"Non-null detail_desc:     {df['detail_desc'].notna().sum():,}")

    print(f"\nTop 10 product_type_name by frequency:")
    for name, count in df["product_type_name"].value_counts().head(10).items():
        print(f"  {name:<35} {count:>5}")

    print(f"\nTop 10 colour_group_name by frequency:")
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

    print("\nAll indices saved to", save_dir)


if __name__ == "__main__":
    main()
