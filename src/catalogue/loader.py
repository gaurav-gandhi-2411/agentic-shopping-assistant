from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Synthetic INR prices for H&M demo — deterministic by article_id hash so
# results are stable across rebuilds. Real brand feeds supply actual prices.
_PRICE_RANGES: dict[str, tuple[int, int]] = {
    "Dress": (999, 3499),
    "Blouse": (799, 2499),
    "Blazer": (1499, 4999),
    "Trousers": (999, 2999),
    "Jeans": (1299, 3499),
    "T-shirt": (399, 1299),
    "Top": (499, 1599),
    "Sweater": (999, 2999),
    "Cardigan": (899, 2499),
    "Coat": (2499, 6999),
    "Jacket": (1499, 4999),
    "Skirt": (699, 1999),
    "Shorts": (599, 1799),
    "Leggings/Tights": (399, 999),
    "Swimsuit": (799, 2499),
}
_DEFAULT_PRICE_RANGE: tuple[int, int] = (599, 2999)


def _synth_price(article_id: str, product_type: str) -> float:
    """Return a deterministic synthetic INR price for a H&M article.

    Uses the first 8 hex digits of the article_id's MD5 hash to pick a price
    within the product-type range, rounded to the nearest ₹10.
    """
    lo, hi = _PRICE_RANGES.get(product_type, _DEFAULT_PRICE_RANGE)
    h = int(hashlib.md5(article_id.encode()).hexdigest()[:8], 16)  # noqa: S324
    return round(lo + (h % (hi - lo + 1)), -1)  # round to nearest 10


KEEP_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "department_name",
    "index_group_name",
    "garment_group_name",
    "detail_desc",
]


def load_articles(config: dict) -> pd.DataFrame:
    csv_path = Path(config["catalogue"]["articles_csv"])
    df = pd.read_csv(csv_path, usecols=KEEP_COLUMNS, dtype={"article_id": str})
    df = df.dropna(subset=["detail_desc"])
    df = df.sample(
        n=config["catalogue"]["sample_num_items"],
        random_state=config["catalogue"]["seed"],
    ).reset_index(drop=True)
    return df


def build_searchable_text(articles_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = articles_df.copy()

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

    if "price_inr" not in df.columns:
        df["price_inr"] = df.apply(
            lambda r: _synth_price(str(r["article_id"]), str(r.get("product_type_name", ""))),
            axis=1,
        )
    if "size_system" not in df.columns:
        df["size_system"] = None
    if "pdp_handle" not in df.columns:
        df["pdp_handle"] = None

    save_dir = Path(config["catalogue"]["processed_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "catalogue.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Catalogue saved to %s", out_path)

    return df


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
