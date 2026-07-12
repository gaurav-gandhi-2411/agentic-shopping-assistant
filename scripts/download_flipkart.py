"""
Convert the Flipkart fashion JSON to a normalized catalogue CSV.

The raw JSON is already at data/raw/flipkart/flipkart_fashion_products_dataset.json
(downloaded via: kaggle datasets download -d aaditshukla/flipkart-fasion-products-dataset).

Produces:
    data/raw/flipkart/fashion_products.csv
    ~15 000 rows, stratified by sub_category, in-stock items only.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAW_JSON = Path("data/raw/flipkart/flipkart_fashion_products_dataset.json")
OUT_CSV = Path("data/raw/flipkart/fashion_products.csv")
TARGET_ROWS = 15_000
SEED = 42

# Sub-categories to keep (excludes Fabrics, Raincoats, misc)
KEEP_CATS = {
    "Topwear",
    "Bottomwear",
    "Winter Wear",
    "Kurtas, Ethnic Sets and Bottoms",
    "Innerwear and Swimwear",
    "Clothing Accessories",
    "Men's Footwear",
    "Blazers, Waistcoats and Suits",
    "Sleepwear",
    "Tracksuits",
}


def _clean_price(val: object) -> float | None:
    if val is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _first_image(images_str: object) -> str | None:
    """Parse images list repr and return first URL, upgraded from 128px to 832px."""
    if not images_str or (isinstance(images_str, float)):
        return None
    try:
        urls = ast.literal_eval(str(images_str))
        if urls:
            url = str(urls[0]).strip()
            # Upgrade Flipkart thumbnail to full-size image
            url = re.sub(r"/image/\d+/\d+/", "/image/832/832/", url)
            return url
    except (ValueError, SyntaxError):
        pass
    return None


def main() -> None:
    if not RAW_JSON.exists():
        print(f"[ERROR] {RAW_JSON} not found.", file=sys.stderr)
        print(
            "Download with: kaggle datasets download "
            "-d aaditshukla/flipkart-fasion-products-dataset "
            "--unzip -p data/raw/flipkart",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {RAW_JSON} ...")
    with open(RAW_JSON, encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records):,} total records")

    rows = []
    for r in records:
        # Skip out-of-stock
        if str(r.get("out_of_stock", "False")).lower() == "true":
            continue
        # Filter to useful sub-categories
        sub_cat = str(r.get("sub_category", "")).strip()
        if sub_cat not in KEEP_CATS:
            continue

        price = _clean_price(r.get("selling_price"))
        if not price or price <= 0:
            continue

        rows.append(
            {
                "id": r.get("_id", ""),
                "title": str(r.get("title", "")).strip(),
                "type": sub_cat,
                "brand": str(r.get("brand", "")).strip(),
                "description": str(r.get("description", "")).strip(),
                "price_inr": price,
                "image_url": _first_image(r.get("images")),
                "url": str(r.get("url", "")).strip(),
            }
        )

    df = pd.DataFrame(rows)
    print(f"  {len(df):,} in-stock rows in kept categories")

    # Stratified sample: preserve category proportions, cap at TARGET_ROWS
    if len(df) > TARGET_ROWS:
        df = df.groupby("type", group_keys=False).apply(
            lambda g: g.sample(
                n=min(len(g), max(1, round(TARGET_ROWS * len(g) / len(df)))),
                random_state=SEED,
            )
        )
        # Top up or trim to exact target
        if len(df) > TARGET_ROWS:
            df = df.sample(n=TARGET_ROWS, random_state=SEED)
        print(f"  Sampled to {len(df):,} rows (stratified by type)")

    df = df.reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(df):,} rows to {OUT_CSV}")
    print("\nCategory breakdown:")
    for cat, cnt in df["type"].value_counts().items():
        print(f"  {cat:<45} {cnt:>5}")


if __name__ == "__main__":
    main()
