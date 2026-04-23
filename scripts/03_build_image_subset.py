"""
Build the Space image subset with category quotas.

Selection: 1,500 items from the full H&M catalogue (105k articles) using
purchase-count priority within each category bucket:
  - Hard caps: lingerie/swimwear <=50, nightwear <=20, socks <=15
  - Minimums: dresses >=150, tops >=200, trousers/skirts >=200,
              outerwear >=150, knitwear >=150
  - Filler: remaining slots by purchase count

Quality filter: detail_desc length >= 50 chars (applied before selection)
Image filter: source JPEG must exist in data/hm/images/

Artifacts produced:
  data/processed/catalogue_space.parquet
  data/processed/dense_space.faiss + dense_space_article_ids.npy
  data/processed/bm25_space.pkl + bm25_space_article_ids.npy
  data/processed/images/ (fresh rebuild — old folder deleted)

Run from repo root:
    python scripts/03_build_image_subset.py
"""
import random
import re
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

SUBSET_SIZE = 1500
IMAGE_WIDTH = 300
JPEG_QUALITY = 75
MIN_DESC_LEN = 50

# Bucket targets — must sum to SUBSET_SIZE
TARGETS = {
    "cap_lingerie_swimwear": 50,
    "cap_nightwear":         20,
    "cap_socks":             15,
    "min_dresses":          175,
    "min_tops":             225,
    "min_trousers":         225,
    "min_outerwear":        150,
    "min_knitwear":         175,
    "filler":               465,
}
assert sum(TARGETS.values()) == SUBSET_SIZE

KEEP_COLS = [
    "article_id", "prod_name", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name", "department_name",
    "index_group_name", "garment_group_name", "detail_desc",
]


def classify_item(dept: str, ptype: str) -> str:
    """Exclusively assign item to one bucket. Capped categories take priority."""
    d = (dept or "").lower()
    p = (ptype or "").lower()
    # Capped categories checked first — these items cannot fill minimum buckets
    if re.search(r"lingerie|underwear|swimwear", d):
        return "cap_lingerie_swimwear"
    if "nightwear" in d:
        return "cap_nightwear"
    if re.search(r"sock|hosier", d):
        return "cap_socks"
    # Minimum-guarantee categories
    if re.search(r"\bdress\b", p):
        return "min_dresses"
    if re.search(r"\bt-shirt\b|\btop\b|\bvest top\b|\bblouse\b|\bshirt\b", p):
        return "min_tops"
    if re.search(r"\btrousers\b|\bjeans\b|\bshorts\b|\bskirt\b|\bleggings\b|\btights\b", p):
        return "min_trousers"
    if re.search(r"\bjacket\b|\bcoat\b|\bblazer\b", p):
        return "min_outerwear"
    if re.search(r"\bsweater\b|\bcardigan\b|\bhoodie\b|\bjumper\b", p):
        return "min_knitwear"
    return "filler"


def select_quota_subset(tx_path: Path, img_root: Path) -> pd.DataFrame:
    """Return a 1,500-row DataFrame selected from the full H&M catalogue with quotas."""
    print("  Loading full catalogue from CSV...")
    t0 = time.time()
    raw = pd.read_csv("data/hm/articles.csv", usecols=KEEP_COLS, dtype={"article_id": str})
    print(f"  {len(raw):,} items loaded in {time.time()-t0:.1f}s")

    raw = raw[raw["detail_desc"].fillna("").str.len() >= MIN_DESC_LEN].copy()
    print(f"  After quality filter (desc >={MIN_DESC_LEN} chars): {len(raw):,} items")

    print("  Checking image existence...")
    t0 = time.time()
    raw["_has_img"] = raw["article_id"].apply(
        lambda aid: (img_root / str(aid)[:3] / f"{aid}.jpg").exists()
    )
    raw = raw[raw["_has_img"]].drop(columns=["_has_img"]).copy()
    print(f"  With image on disk: {len(raw):,} items ({time.time()-t0:.1f}s)")

    print("  Loading purchase counts from transactions...")
    t0 = time.time()
    tx = pd.read_csv(tx_path, usecols=["article_id"], dtype={"article_id": str})
    counts = tx["article_id"].value_counts().rename("purchase_count")
    print(f"  {len(tx):,} transactions, {len(counts):,} unique articles ({time.time()-t0:.1f}s)")

    raw = raw.merge(counts, on="article_id", how="left")
    raw["purchase_count"] = raw["purchase_count"].fillna(0).astype(int)
    raw = raw.sort_values("purchase_count", ascending=False).reset_index(drop=True)

    raw["bucket"] = raw.apply(
        lambda r: classify_item(r["department_name"], r["product_type_name"]), axis=1
    )

    pools = {b: raw[raw["bucket"] == b] for b in TARGETS}

    print(f"\n  {'Bucket':<30} {'Pool':>7}  {'Target':>7}  Status")
    print(f"  {'-'*55}")
    shortage = False
    for bucket, target in TARGETS.items():
        n = len(pools[bucket])
        ok = n >= target
        if not ok:
            shortage = True
        print(f"  {bucket:<30} {n:>7,}  {target:>7,}  {'OK' if ok else 'SHORTAGE'}")

    if shortage:
        raise RuntimeError("Cannot build subset: one or more pools are too small.")

    # Select non-filler buckets first (straight top-N by purchase count)
    non_filler_parts = [pools[b].head(TARGETS[b]) for b in TARGETS if b != "filler"]
    non_filler_df = pd.concat(non_filler_parts)

    # Count Black items already committed, then cap filler additions
    BLACK_CEILING = int(SUBSET_SIZE * 0.35)  # 525 (35% of 1500)
    black_count = int((non_filler_df["colour_group_name"].str.lower() == "black").sum())
    print(f"\n  Black items in named buckets: {black_count}  (ceiling: {BLACK_CEILING})")

    filler_pool = pools["filler"].sort_values("purchase_count", ascending=False)
    filler_selected = []
    skipped_black = 0
    for _, row in filler_pool.iterrows():
        if len(filler_selected) >= TARGETS["filler"]:
            break
        is_black = str(row.get("colour_group_name", "")).lower() == "black"
        if is_black and black_count >= BLACK_CEILING:
            skipped_black += 1
            continue
        filler_selected.append(row)
        if is_black:
            black_count += 1

    if len(filler_selected) < TARGETS["filler"]:
        raise RuntimeError(
            f"Filler pool exhausted: got {len(filler_selected)}, need {TARGETS['filler']}. "
            f"Skipped {skipped_black} Black items due to ceiling."
        )
    print(f"  Black items skipped in filler: {skipped_black}  final Black total: {black_count}")

    filler_df = pd.DataFrame(filler_selected)
    subset = pd.concat([non_filler_df, filler_df]).drop_duplicates(subset="article_id").copy()

    if len(subset) != SUBSET_SIZE:
        raise RuntimeError(f"Expected {SUBSET_SIZE} items after dedup, got {len(subset)}")

    return subset.drop(columns=["bucket", "purchase_count"])


def add_processed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add search_text, display_name, facets — same logic as src/catalogue/loader.py."""
    df = df.copy()
    df["search_text"] = (
        df["prod_name"].fillna("") + ". "
        + df["product_type_name"].fillna("") + ". "
        + df["colour_group_name"].fillna("") + ". "
        + df["department_name"].fillna("") + ". "
        + df["detail_desc"].fillna("")
    )
    df["display_name"] = (
        df["prod_name"].fillna("").str.strip()
        + " ("
        + df["colour_group_name"].fillna("").str.strip()
        + " "
        + df["product_type_name"].fillna("").str.strip()
        + ")"
    )
    df["facets"] = df.apply(
        lambda r: {
            "colour_group_name": r["colour_group_name"],
            "product_type_name": r["product_type_name"],
            "department_name": r["department_name"],
            "index_group_name": r["index_group_name"],
            "garment_group_name": r["garment_group_name"],
        },
        axis=1,
    )
    return df


def print_composition(subset: pd.DataFrame) -> None:
    dept = subset["department_name"].str.lower().fillna("")
    ptype = subset["product_type_name"].str.lower().fillna("")

    constraints = [
        ("Lingerie/Underwear/Swimwear (dept, MAX 50)",
         dept.str.contains(r"lingerie|underwear|swimwear", regex=True).sum(), "<=", 50),
        ("Nightwear (dept, MAX 50)",
         dept.str.contains("nightwear").sum(), "<=", 50),
        ("Socks/Hosiery (dept, MAX 30)",
         dept.str.contains(r"sock|hosier", regex=True).sum(), "<=", 30),
        ("Dresses (ptype, MIN 150)",
         ptype.str.contains(r"\bdress\b", regex=True).sum(), ">=", 150),
        ("Tops (ptype, MIN 200)",
         ptype.str.contains(r"\bt-shirt\b|\btop\b|\bvest top\b|\bblouse\b|\bshirt\b", regex=True).sum(), ">=", 200),
        ("Trousers/Jeans/Shorts/Skirts/Leggings (ptype, MIN 200)",
         ptype.str.contains(r"\btrousers\b|\bjeans\b|\bshorts\b|\bskirt\b|\bleggings\b|\btights\b", regex=True).sum(), ">=", 200),
        ("Outerwear (ptype, MIN 150)",
         ptype.str.contains(r"\bjacket\b|\bcoat\b|\bblazer\b", regex=True).sum(), ">=", 150),
        ("Knitwear (ptype, MIN 150)",
         ptype.str.contains(r"\bsweater\b|\bcardigan\b|\bhoodie\b|\bjumper\b", regex=True).sum(), ">=", 150),
    ]

    print(f"\n{'='*68}")
    print(f"COMPOSITION AUDIT  (total: {len(subset):,} items)")
    print(f"{'='*68}")
    all_pass = True
    for name, count, op, limit in constraints:
        ok = (count <= limit) if op == "<=" else (count >= limit)
        if not ok:
            all_pass = False
        print(f"  {name:<48} {count:>4}  [{op}{limit}] {'PASS' if ok else 'FAIL'}")

    print(f"\n  Colour distribution (top 10):")
    for colour, cnt in subset["colour_group_name"].value_counts().head(10).items():
        print(f"    {colour:<28} {cnt:>4}  ({cnt/len(subset)*100:.1f}%)")

    print(f"\n  index_group_name:")
    for grp, cnt in subset["index_group_name"].value_counts().items():
        print(f"    {grp:<28} {cnt:>4}  ({cnt/len(subset)*100:.1f}%)")

    print(f"\n  product_type_name (top 20):")
    for pt, cnt in subset["product_type_name"].value_counts().head(20).items():
        print(f"    {pt:<35} {cnt:>4}")

    if not all_pass:
        raise RuntimeError("Composition check FAILED — one or more constraints violated.")
    print(f"\n  All constraints PASSED.")


def resize_images(subset_df: pd.DataFrame, img_src: Path, img_dst: Path) -> int:
    img_dst.mkdir(parents=True, exist_ok=True)
    errors = 0
    for aid in subset_df["article_id"]:
        src = img_src / aid[:3] / f"{aid}.jpg"
        dst = img_dst / aid[:3] / f"{aid}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((IMAGE_WIDTH, 9999), Image.LANCZOS)
                im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
        except Exception as e:
            print(f"  WARNING: {aid}: {e}")
            errors += 1
    return errors


def folder_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print composition only; skip image resize and index rebuild.")
    args = parser.parse_args()

    config = load_config()
    repo_root = Path(__file__).parent.parent
    processed = repo_root / config["catalogue"]["processed_dir"]
    img_src = repo_root / "data" / "hm" / "images"
    img_dst = processed / "images"
    tx_path = repo_root / "data" / "hm" / "transactions_train.csv"

    # ------------------------------------------------------------------
    print("=== Step 1: select quota subset ===")
    subset_raw = select_quota_subset(tx_path, img_src)

    # ------------------------------------------------------------------
    print("\n=== Step 2: add processed columns ===")
    subset = add_processed_columns(subset_raw)

    # ------------------------------------------------------------------
    print("\n=== Step 3: composition audit ===")
    print_composition(subset)

    if args.dry_run:
        print("\nDry run complete. Confirm composition above, then re-run without --dry-run.")
        return

    # ------------------------------------------------------------------
    print("\n=== Step 4: resize images (deleting old folder first) ===")
    if img_dst.exists():
        print(f"  Removing existing {img_dst} ...")
        shutil.rmtree(img_dst)
    print(f"  Resizing {len(subset):,} images -> {IMAGE_WIDTH}px wide, JPEG q{JPEG_QUALITY}...")
    t0 = time.time()
    errs = resize_images(subset, img_src, img_dst)
    elapsed = time.time() - t0
    img_mb = folder_size_mb(img_dst)
    print(f"  Done in {elapsed:.0f}s  ({errs} errors)  images/: {img_mb:.1f} MB")

    # ------------------------------------------------------------------
    print("\n=== Step 5: save catalogue_space.parquet ===")
    subset["image_url"] = subset["article_id"].apply(
        lambda aid: f"images/{aid[:3]}/{aid}.jpg"
    )
    space_parquet = processed / "catalogue_space.parquet"
    subset.to_parquet(space_parquet, index=False)
    parquet_mb = space_parquet.stat().st_size / 1024 / 1024
    print(f"  Saved {len(subset):,} rows -> {parquet_mb:.1f} MB")

    # ------------------------------------------------------------------
    print("\n=== Step 6: rebuild Space indices ===")
    tmp = processed / "_space_tmp"
    tmp.mkdir(exist_ok=True)

    print("  Dense index...")
    t0 = time.time()
    dense = DenseRetriever(config)
    dense.build_index(subset, tmp)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("  BM25 index...")
    t0 = time.time()
    sparse = SparseRetriever(config)
    sparse.build_index(subset, tmp)
    print(f"  Done in {time.time()-t0:.1f}s")

    for src_name, dst_name in [
        ("dense.faiss",           "dense_space.faiss"),
        ("dense_article_ids.npy", "dense_space_article_ids.npy"),
        ("bm25.pkl",              "bm25_space.pkl"),
        ("bm25_article_ids.npy",  "bm25_space_article_ids.npy"),
    ]:
        dst = processed / dst_name
        if dst.exists():
            dst.unlink()
        (tmp / src_name).rename(dst)
    shutil.rmtree(tmp)
    print("  Indices written with _space suffix.")

    # ------------------------------------------------------------------
    print("\n=== Artifact size report ===")
    artifacts = [
        ("catalogue_space.parquet",       space_parquet),
        ("dense_space.faiss",             processed / "dense_space.faiss"),
        ("dense_space_article_ids.npy",   processed / "dense_space_article_ids.npy"),
        ("bm25_space.pkl",                processed / "bm25_space.pkl"),
        ("bm25_space_article_ids.npy",    processed / "bm25_space_article_ids.npy"),
    ]
    total_mb = img_mb
    for label, path in artifacts:
        mb = path.stat().st_size / 1024 / 1024
        total_mb += mb
        print(f"  {label:<42} {mb:>6.1f} MB")
    print(f"  {'images/ folder':<42} {img_mb:>6.1f} MB")
    print(f"  {'─'*52}")
    print(f"  {'TOTAL':<42} {total_mb:>6.1f} MB")
    budget = "within budget" if total_mb <= 50 else "EXCEEDS 50 MB BUDGET"
    print(f"  ({budget})")

    # ------------------------------------------------------------------
    print("\n=== Image spot-check (3 random) ===")
    random.seed(99)
    for aid in random.sample(list(subset["article_id"]), 3):
        path = img_dst / aid[:3] / f"{aid}.jpg"
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                print(f"  {aid}: {im.size}  {path.stat().st_size//1024} KB  OK")
        except Exception as e:
            print(f"  {aid}: FAILED — {e}")

    print("\nDone. Run:  python spaces/upload_artifacts.py --repo <user>/<space> --space")


if __name__ == "__main__":
    main()
