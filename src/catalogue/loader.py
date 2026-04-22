import pandas as pd
import yaml
from pathlib import Path


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

    save_dir = Path(config["catalogue"]["processed_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "catalogue.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

    return df


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
