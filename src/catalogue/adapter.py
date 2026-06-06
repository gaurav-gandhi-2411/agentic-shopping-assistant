"""
Catalogue feed adapter: maps arbitrary product CSV columns to the internal catalogue schema.

Internal schema columns (all required, some nullable):
    article_id, prod_name, product_type_name, product_group_name,
    graphical_appearance_name, colour_group_name, department_name,
    index_group_name, garment_group_name, detail_desc,
    price_inr (nullable float), size_system (nullable str), pdp_handle (nullable str)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Internal schema ────────────────────────────────────────────────────────────

INTERNAL_COLUMNS: list[str] = [
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
    # New nullable columns (B2) — must not break existing H&M index readers
    "price_inr",       # float | None
    "size_system",     # str | None — "IN" | "EU" | "alpha"
    "pdp_handle",      # str | None
]

# Columns that existed before B2; used for back-compat fill logic
_LEGACY_COLUMNS: set[str] = set(INTERNAL_COLUMNS) - {"price_inr", "size_system", "pdp_handle"}

# Default fill values for legacy string columns when the source feed omits them
_COLUMN_DEFAULTS: dict[str, str] = {
    "product_group_name": "N/A",
    "graphical_appearance_name": "N/A",
    "department_name": "N/A",
    "index_group_name": "N/A",
    "garment_group_name": "N/A",
}

# ── Generic/Shopify column mapping ─────────────────────────────────────────────
# Each entry: internal_column -> ordered list of candidate source column names.
# The first matching source column wins.
_GENERIC_COLUMN_MAP: dict[str, list[str]] = {
    "article_id":               ["id"],
    "prod_name":                ["name"],
    "product_type_name":        ["type", "category"],
    "product_group_name":       ["group", "product_group"],
    "graphical_appearance_name": ["appearance"],
    "colour_group_name":        ["colour", "color"],
    "department_name":          ["department"],
    "index_group_name":         ["index_group"],
    "garment_group_name":       ["garment_group"],
    "detail_desc":              ["description", "desc"],
    "price_inr":                ["price_inr", "price"],
    "pdp_handle":               ["handle", "slug"],
}


# ── Public API ─────────────────────────────────────────────────────────────────

def adapt_feed(
    df: pd.DataFrame,
    brand_config: object,  # BrandConfig — typed as object to avoid circular import
    *,
    size_system: str | None = None,
) -> pd.DataFrame:
    """Map an arbitrary product feed DataFrame to the internal catalogue schema.

    Auto-detects the layout:
    - **H&M layout**: DataFrame already contains ``article_id`` → pass through,
      add nullable new columns with nulls.
    - **Generic/Shopify layout**: map columns using :data:`_GENERIC_COLUMN_MAP`.

    Parameters
    ----------
    df:
        Raw product feed, freshly read from CSV (or any source).
    brand_config:
        A ``BrandConfig`` instance.  Uses ``.sizing_system`` to populate the
        ``size_system`` column unless ``size_system`` is provided explicitly.
    size_system:
        Override the sizing system written to all rows.  ``None`` → read from
        ``brand_config.sizing_system``.

    Returns
    -------
    pd.DataFrame
        DataFrame with all columns in :data:`INTERNAL_COLUMNS`, in order.
    """
    effective_size_system: str | None = size_system or getattr(brand_config, "sizing_system", None)

    if _is_hm_layout(df):
        logger.debug("adapt_feed: detected H&M layout (%d rows)", len(df))
        out = _adapt_hm(df, effective_size_system)
    else:
        logger.debug("adapt_feed: detected generic layout (%d rows)", len(df))
        out = _adapt_generic(df, effective_size_system)

    # Guarantee column order and completeness
    out = _ensure_schema(out)
    return out


# ── Layout detection ───────────────────────────────────────────────────────────

def _is_hm_layout(df: pd.DataFrame) -> bool:
    """Return True when *df* already uses the H&M internal column names."""
    return "article_id" in df.columns


# ── H&M pass-through ───────────────────────────────────────────────────────────

def _adapt_hm(df: pd.DataFrame, size_system: str | None) -> pd.DataFrame:
    """Pass H&M df through; add the three new nullable columns."""
    out = df.copy()
    if "price_inr" not in out.columns:
        out["price_inr"] = None
    if "pdp_handle" not in out.columns:
        out["pdp_handle"] = None
    out["size_system"] = size_system  # broadcasts to all rows
    return out


# ── Generic/Shopify mapping ────────────────────────────────────────────────────

def _adapt_generic(df: pd.DataFrame, size_system: str | None) -> pd.DataFrame:
    """Map generic feed columns to the internal schema."""
    out = pd.DataFrame(index=df.index)

    for internal_col, candidates in _GENERIC_COLUMN_MAP.items():
        matched: str | None = _find_column(df, candidates)
        if matched is not None:
            out[internal_col] = df[matched]
        else:
            # article_id: fall back to the DataFrame's first column (stringified)
            if internal_col == "article_id":
                out[internal_col] = df.iloc[:, 0].astype(str)
                logger.debug("adapt_feed: article_id fallback → first column '%s'", df.columns[0])
            else:
                out[internal_col] = None

    # Coerce article_id to string
    out["article_id"] = out["article_id"].astype(str)

    # Coerce price_inr to float (errors → NaN)
    if out["price_inr"].notna().any():
        out["price_inr"] = pd.to_numeric(out["price_inr"], errors="coerce")

    # Size system — broadcast scalar
    out["size_system"] = size_system

    return out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in *df* (case-insensitive)."""
    lower_map: dict[str, str] = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee all internal columns exist with correct defaults; return ordered df."""
    out = df.copy()

    for col in INTERNAL_COLUMNS:
        if col not in out.columns:
            out[col] = None

    # Apply string defaults for legacy columns that are entirely null
    for col, default in _COLUMN_DEFAULTS.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)

    # Return only internal columns, in canonical order
    return out[INTERNAL_COLUMNS]
