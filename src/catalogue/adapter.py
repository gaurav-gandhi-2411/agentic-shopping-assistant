"""
Catalogue feed adapter: maps arbitrary product CSV columns to the internal catalogue schema.

Internal schema columns (all required, some nullable):
    article_id, prod_name, product_type_name, product_group_name,
    graphical_appearance_name, colour_group_name, department_name,
    index_group_name, garment_group_name, detail_desc,
    price_inr (nullable float), size_system (nullable str), pdp_handle (nullable str)
"""

from __future__ import annotations

import ast
import logging
import math
import re

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
    "image_url",       # str | None — first product image URL
    "pdp_live",        # bool | None — True=live, False=dead, None=not checked
    "gender",           # str: "men" | "women" | "unknown"
    "type_confidence",  # str: "high" | "medium" | "low" | "unknown"
]

# Columns that existed before B2; used for back-compat fill logic
_LEGACY_COLUMNS: set[str] = set(INTERNAL_COLUMNS) - {
    "price_inr", "size_system", "pdp_handle", "image_url", "pdp_live", "gender"
}

# Default fill values for legacy string columns when the source feed omits them
_COLUMN_DEFAULTS: dict[str, str] = {
    "product_group_name": "N/A",
    "graphical_appearance_name": "N/A",
    "department_name": "N/A",
    "index_group_name": "N/A",
    "garment_group_name": "N/A",
    "gender": "unknown",
    "type_confidence": "unknown",
}

# ── Colour extraction from product titles ─────────────────────────────────────
# Deterministic scan: check each phrase (longest first) against the title.
# Maps to canonical H&M colour_group_name vocabulary so facet filters work.
# Phrases are sorted longest-first so "dark blue" matches before "blue".
_COLOUR_PHRASES: list[tuple[str, str]] = sorted(
    [
        ("off white", "Off White"),
        ("off-white", "Off White"),
        ("dark blue", "Dark Blue"),
        ("navy blue", "Dark Blue"),
        ("dark red", "Dark Red"),
        ("dark green", "Dark Green"),
        ("dark grey", "Dark Grey"),
        ("dark gray", "Dark Grey"),
        ("light blue", "Light Blue"),
        ("sky blue", "Light Blue"),
        ("light pink", "Light Pink"),
        ("baby pink", "Light Pink"),
        ("powder pink", "Light Pink"),
        ("light beige", "Light Beige"),
        ("light grey", "Light Grey"),
        ("light gray", "Light Grey"),
        ("silver grey", "Light Grey"),
        ("navy", "Dark Blue"),
        ("maroon", "Dark Red"),
        ("wine", "Dark Red"),
        ("burgundy", "Dark Red"),
        ("magenta", "Pink"),
        ("coral", "Pink"),
        ("rose", "Pink"),
        ("peach", "Light Pink"),
        ("cream", "Off White"),
        ("ivory", "Off White"),
        ("beige", "Beige"),
        ("khaki", "Khaki"),
        ("olive", "Khaki"),
        ("mustard", "Yellow"),
        ("lemon", "Yellow"),
        ("yellow", "Yellow"),
        ("orange", "Orange"),
        ("turquoise", "Turquoise"),
        ("teal", "Turquoise"),
        ("mint", "Green"),
        ("green", "Green"),
        ("purple", "Purple"),
        ("lavender", "Purple"),
        ("violet", "Purple"),
        ("lilac", "Purple"),
        ("brown", "Brown"),
        ("tan", "Brown"),
        ("camel", "Brown"),
        ("chocolate", "Brown"),
        ("black", "Black"),
        ("white", "White"),
        ("grey", "Grey"),
        ("gray", "Grey"),
        ("silver", "Grey"),
        ("blue", "Blue"),
        ("red", "Red"),
        ("pink", "Pink"),
        ("multi", None),   # multicoloured → skip (no canonical match)
        ("print", None),   # "floral print" → skip
    ],
    key=lambda t: -len(t[0]),   # longest phrase first
)


def extract_colour_from_title(title: str) -> str | None:
    """Scan a product title for the first colour keyword; return canonical colour or None.

    Uses a longest-match priority list to avoid 'blue' matching before 'dark blue'.
    Returns None for multicoloured / printed / unrecognised items.
    """
    if not title:
        return None
    lower = title.lower()
    for phrase, canonical in _COLOUR_PHRASES:
        if phrase in lower:
            return canonical   # None for "multi"/"print" → caller treats as unknown
    return None


# ── Gender keyword sets ────────────────────────────────────────────────────────
# Women checked BEFORE men so "women's kurta" doesn't false-match "men" substring
_GENDER_WOMEN_KEYWORDS: frozenset[str] = frozenset({
    "saree", "lehenga", "dupatta", "kurti", "anarkali", "sharara",
    "salwar kameez", "palazzo", "ghagra", "choli",
    "women", "woman", "ladies", "female", "girl", "girls",
})
_GENDER_MEN_KEYWORDS: frozenset[str] = frozenset({
    "sherwani", "bandhgala",
    "men", "man", "male", "boys", "boy",
})

# ── Generic/Shopify column mapping ─────────────────────────────────────────────
# Each entry: internal_column -> ordered list of candidate source column names.
# The first matching source column wins.
_GENERIC_COLUMN_MAP: dict[str, list[str]] = {
    "article_id":               ["id"],
    "prod_name":                ["name", "title"],
    "product_type_name":        ["type", "category"],
    "product_group_name":       ["group", "product_group"],
    "graphical_appearance_name": ["appearance"],
    "colour_group_name":        ["colour", "color"],
    "department_name":          ["department"],
    "index_group_name":         ["index_group"],
    "garment_group_name":       ["garment_group"],
    "detail_desc":              ["description", "desc"],
    "price_inr":                ["price_inr", "price"],
    "pdp_handle":               ["handle", "slug", "url"],
    "image_url":                ["image_url", "image_urls", "image"],
}


# ── Gender derivation ──────────────────────────────────────────────────────────

def derive_item_gender(prod_name: str, product_type_name: str, brand_default: str) -> str:
    """Derive per-item gender via fallback chain.

    1. Name/type keyword scan (women-specific → women; men-specific → men).
       Check women BEFORE men so "women's kurta" doesn't match "men" inside "women".
    2. Brand default if not "mixed" or "unknown".
    3. Return "unknown".
    """
    combined = (prod_name + " " + product_type_name).lower()
    # Women keywords take priority (checked first to avoid "women" matching "men" sub-string)
    if any(kw in combined for kw in _GENDER_WOMEN_KEYWORDS):
        return "women"
    if any(kw in combined for kw in _GENDER_MEN_KEYWORDS):
        return "men"
    if brand_default and brand_default.lower() not in ("mixed", "unknown", ""):
        return brand_default.lower()
    return "unknown"


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
    elif _is_myntra_layout(df):
        logger.debug("adapt_feed: detected Myntra layout (%d rows)", len(df))
        out = _adapt_myntra(df, effective_size_system)
    else:
        logger.debug("adapt_feed: detected generic layout (%d rows)", len(df))
        brand_gender_default: str = getattr(brand_config, "gender_default", "unknown") or "unknown"
        out = _adapt_generic(df, effective_size_system, brand_gender_default)

    # Guarantee column order and completeness
    out = _ensure_schema(out)
    return out


# ── Layout detection ───────────────────────────────────────────────────────────

def _is_hm_layout(df: pd.DataFrame) -> bool:
    """Return True when *df* already uses the H&M internal column names."""
    return "article_id" in df.columns


def _is_myntra_layout(df: pd.DataFrame) -> bool:
    """Return True when df looks like the Myntra product feed (p_id + img + brand columns)."""
    return "p_id" in df.columns and "img" in df.columns and not _is_hm_layout(df)


# ── H&M pass-through ───────────────────────────────────────────────────────────

def _adapt_hm(df: pd.DataFrame, size_system: str | None) -> pd.DataFrame:
    """Pass H&M df through; add the three new nullable columns."""
    out = df.copy()
    if "price_inr" not in out.columns:
        out["price_inr"] = None
    if "pdp_handle" not in out.columns:
        out["pdp_handle"] = None
    out["size_system"] = size_system  # broadcasts to all rows

    # Derive gender from the already-populated index_group_name
    def _ig_to_gender(ig: object) -> str:
        s = str(ig or "").lower()
        if "menswear" in s or s == "men":
            return "men"
        if "ladieswear" in s or "women" in s or "ladies" in s:
            return "women"
        return "unknown"

    out["gender"] = out["index_group_name"].apply(_ig_to_gender)

    # Normalise garment_type: run deterministic normalizer and, when confidence
    # is high or medium, overwrite product_type_name with the canonical garment type.
    from src.catalogue.normalizer import normalize_garment_type as _normalize  # noqa: PLC0415

    _norm_results = [
        _normalize(str(n), str(pt))
        for n, pt in zip(
            out.get("prod_name", pd.Series(dtype=str)).fillna(""),
            out.get("product_type_name", pd.Series(dtype=str)).fillna(""),
        )
    ]
    _confident_mask = [r.type_confidence in ("high", "medium") for r in _norm_results]
    out["product_type_name"] = [
        r.garment_type if confident else orig
        for r, confident, orig in zip(_norm_results, _confident_mask, out["product_type_name"])
    ]
    out["type_confidence"] = [r.type_confidence for r in _norm_results]

    return out


# ── Generic/Shopify mapping ────────────────────────────────────────────────────

def _adapt_generic(
    df: pd.DataFrame,
    size_system: str | None,
    brand_gender_default: str = "unknown",
) -> pd.DataFrame:
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

    # Extract colour from product title when the feed has no colour column.
    # Shopify stores embed colour in the product name ("Black Solid Halter Neck...").
    if out["colour_group_name"].isna().all():
        out["colour_group_name"] = out["prod_name"].fillna("").apply(extract_colour_from_title)

    # Size system — broadcast scalar
    out["size_system"] = size_system

    # Derive per-item gender
    out["gender"] = [
        derive_item_gender(
            str(n or ""),
            str(pt or ""),
            brand_gender_default,
        )
        for n, pt in zip(
            out.get("prod_name", pd.Series(dtype=str)).fillna(""),
            out.get("product_type_name", pd.Series(dtype=str)).fillna(""),
        )
    ]

    # Normalise garment_type: run deterministic normalizer and, when confidence
    # is high or medium, overwrite product_type_name with the canonical garment type.
    from src.catalogue.normalizer import normalize_garment_type as _normalize  # noqa: PLC0415

    _norm_results = [
        _normalize(str(n), str(pt))
        for n, pt in zip(
            out.get("prod_name", pd.Series(dtype=str)).fillna(""),
            out.get("product_type_name", pd.Series(dtype=str)).fillna(""),
        )
    ]
    _confident_mask = [r.type_confidence in ("high", "medium") for r in _norm_results]
    out["product_type_name"] = [
        r.garment_type if confident else orig
        for r, confident, orig in zip(_norm_results, _confident_mask, out["product_type_name"])
    ]
    out["type_confidence"] = [r.type_confidence for r in _norm_results]

    return out


def _clean_price_column(series: pd.Series) -> pd.Series:
    """Strip currency symbols, commas, spaces from price strings; coerce to float.

    Handles inputs like '₹999', 'Rs. 1,299', '2499.0', 1299 (int), etc.
    Returns NaN for unparseable values.
    """
    def _one(val: object) -> float | None:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        cleaned = re.sub(r"[^\d.]", "", str(val))
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    return series.apply(_one)


def _extract_first_url(val: object) -> str | None:
    """Return the first URL from a possibly pipe- or semicolon-separated list."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    for sep in ("|", ";", "\n"):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return parts[0] if parts else None
    return s


# ── Myntra mapping ──────────────────────────────────────────────────────────────


def _adapt_myntra(df: pd.DataFrame, size_system: str | None) -> pd.DataFrame:
    """Map Myntra product feed to internal schema.

    Actual source columns (hiteshsuthar101/myntra-fashion-product-dataset, 'Fashion Dataset.csv'):
        p_id, name, price, colour, brand, img, description, p_attributes,
        ratingCount, avg_rating
    """
    out = pd.DataFrame(index=df.index)

    # article_id: numeric p_id as integer string (drop the .0 float suffix)
    out["article_id"] = (
        pd.to_numeric(df["p_id"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )

    # prod_name
    out["prod_name"] = df.get("name", pd.Series(dtype=str, index=df.index))

    # product_type_name: parse p_attributes, fall back to name keyword scan, then "Fashion"
    attrs_src = df.get("p_attributes")
    _names = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    if attrs_src is not None:
        _pt_from_attrs = attrs_src.apply(_extract_myntra_product_type)
        _pt_from_name = _names.apply(_product_type_from_name)
        out["product_type_name"] = _pt_from_attrs.where(_pt_from_attrs.notna(), _pt_from_name).fillna("Fashion")
    else:
        out["product_type_name"] = _names.apply(_product_type_from_name).fillna("Fashion")

    # colour_group_name: already title-cased in this dataset
    colour_src = df.get("colour", pd.Series(dtype=str, index=df.index))
    out["colour_group_name"] = colour_src.str.strip()

    # detail_desc: strip HTML tags from description; fall back to name if null
    desc_src = df.get("description")
    name_src = df.get("name", pd.Series("", index=df.index)).fillna("")
    if desc_src is not None and desc_src.notna().any():
        clean_desc = desc_src.apply(
            lambda v: re.sub(r"<[^>]+>", " ", str(v)).strip() if pd.notna(v) else None
        )
        out["detail_desc"] = clean_desc.where(clean_desc.notna() & clean_desc.str.len().gt(0), name_src)
    else:
        out["detail_desc"] = name_src

    # price_inr: already float in this dataset; coerce to be safe
    price_src = df.get("price")
    if price_src is not None:
        out["price_inr"] = pd.to_numeric(price_src, errors="coerce")
    else:
        out["price_inr"] = None

    # image_url: single URL per product in this dataset.
    # Upgrade http:// → https:// — Myntra CDN serves over HTTPS; plain HTTP
    # causes mixed-content blocks when the app is served over HTTPS.
    def _upgrade_https(v: object) -> str | None:
        if not pd.notna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        if s.startswith("http://"):
            s = "https://" + s[7:]
        return s

    out["image_url"] = df.get("img", pd.Series(dtype=str, index=df.index)).apply(_upgrade_https)

    # pdp_handle: type-slug/brand-slug/name-slug/id/buy — mirrors real Myntra PDP URLs
    # pdp_url_template in brands/myntra.yaml = "https://www.myntra.com/{handle}"
    brand_src = df.get("brand", pd.Series("", index=df.index)).fillna("").astype(str)
    name_src2 = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    pid_src = out["article_id"]
    pt_src = out["product_type_name"].fillna("Fashion")
    out["pdp_handle"] = [
        _myntra_pdp_path(b, n, pid, pt)
        for b, n, pid, pt in zip(brand_src, name_src2, pid_src, pt_src)
    ]

    # Normalise garment_type: run deterministic normalizer and, when confidence
    # is high or medium, overwrite product_type_name with the canonical garment type.
    # Adds type_confidence column for filtering/reporting.
    from src.catalogue.normalizer import normalize_garment_type as _normalize  # noqa: PLC0415

    _brand_col = df.get("brand", pd.Series("", index=df.index)).fillna("").astype(str)
    _norm_results = [
        _normalize(str(n), str(pt), str(b))
        for n, pt, b in zip(
            out["prod_name"].fillna(""),
            out["product_type_name"].fillna(""),
            _brand_col,
        )
    ]
    _confident_mask = [r.type_confidence in ("high", "medium") for r in _norm_results]
    # Overwrite product_type_name only for high/medium confidence items
    out["product_type_name"] = [
        r.garment_type if confident else orig
        for r, confident, orig in zip(_norm_results, _confident_mask, out["product_type_name"])
    ]
    out["type_confidence"] = [r.type_confidence for r in _norm_results]

    # Legacy string columns — fill with defaults
    out["size_system"] = size_system
    out["product_group_name"] = "N/A"
    out["graphical_appearance_name"] = "N/A"
    out["department_name"] = "N/A"
    out["garment_group_name"] = "N/A"

    # Derive per-item gender and map to index_group_name for filter compatibility
    out["gender"] = [
        derive_item_gender(n, pt, "women")
        for n, pt in zip(
            out["prod_name"].fillna("").astype(str),
            out["product_type_name"].fillna("").astype(str),
        )
    ]
    out["index_group_name"] = out["gender"].map(
        {"men": "Menswear", "women": "Ladieswear"}
    ).fillna("N/A")

    return out


def _extract_myntra_product_type(attrs_val: object) -> str | None:
    """Parse the p_attributes dict string and return the most specific product type."""
    if attrs_val is None or (isinstance(attrs_val, float) and math.isnan(attrs_val)):
        return None
    try:
        attrs = ast.literal_eval(str(attrs_val))
        for key in ("Top Type", "Bottom Type", "Dress Type", "Footwear Type", "Accessory Type"):
            val = attrs.get(key)
            if val and val != "NA":
                return val
        return None
    except (ValueError, SyntaxError):
        return None


_MYNTRA_TYPE_KEYWORDS: list[str] = [
    "Kurta", "Kurti", "Saree", "Dupatta", "Lehenga", "Palazzo", "Salwar",
    "Kameez", "Dress", "Gown", "Blouse", "Skirt", "Shirt", "T-shirt",
    "Tshirt", "Jeans", "Jacket", "Coat", "Sweater", "Sweatshirt", "Kaftan",
    "Anarkali", "Tunic", "Top",
]


def _product_type_from_name(name: str) -> str | None:
    """Extract product type from product title by scanning for known fashion keywords."""
    name_lower = name.lower()
    for kw in _MYNTRA_TYPE_KEYWORDS:
        if kw.lower() in name_lower:
            return kw
    return None


# Maps internal product_type_name → Myntra URL path segment (always lowercase, pluralised)
_MYNTRA_TYPE_SLUG: dict[str, str] = {
    "Kurta": "kurtas",
    "Kurti": "kurtis",
    "Saree": "sarees",
    "Dupatta": "dupattas",
    "Lehenga": "lehengas",
    "Palazzo": "palazzos",
    "Salwar": "salwars",
    "Kameez": "kameez",
    "Dress": "dresses",
    "Gown": "gowns",
    "Blouse": "blouses",
    "Skirt": "skirts",
    "Shirt": "shirts",
    "T-shirt": "t-shirts",
    "Tshirt": "t-shirts",
    "Jeans": "jeans",
    "Jacket": "jackets",
    "Coat": "coats",
    "Sweater": "sweaters",
    "Sweatshirt": "sweatshirts",
    "Kaftan": "kaftans",
    "Anarkali": "anarkalis",
    "Tunic": "tunics",
    "Top": "tops",
    "Fashion": "clothing",
}


def _myntra_pdp_path(brand: str, name: str, p_id: str, product_type: str) -> str:
    """Construct the URL path segment for a Myntra product page.

    Real Myntra PDP format: {type-slug}/{brand-slug}/{name-slug}/{id}/buy
    e.g. kurtas/khushal-k/khushal-k-women-black-kurta/17048614/buy

    pdp_url_template in brands/myntra.yaml is 'https://www.myntra.com/{handle}'.
    """
    type_slug = _MYNTRA_TYPE_SLUG.get(
        product_type,
        re.sub(r"[^a-z0-9]+", "-", product_type.lower()).strip("-") + "s",
    )
    brand_slug = re.sub(r"[^a-z0-9]+", "-", brand.lower()).strip("-")
    name_slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80]
    return f"{type_slug}/{brand_slug}/{name_slug}/{p_id}/buy"


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
