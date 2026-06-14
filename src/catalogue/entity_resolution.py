"""Precision-gated cross-store entity resolution for same-product price comparison.

Design rationale
----------------
The unified catalogue has 6 stores: snitch, myntra, flipkart, fashor, powerlook,
virgio.  Four of the six are single-brand D2C stores (snitch, fashor, powerlook,
virgio) — they physically cannot share products with other stores.  The remaining
two (myntra/flipkart) carry multi-brand merchandise, but real same-SKU overlap
in this dataset is ~zero: brand-name collisions between myntra and flipkart
(e.g. "Fusion", "Free Authority", "Next") carry *different products* (tops vs
football shoes, vests vs sweatshirts, jeans vs underwear).

Matching rule — precision over recall
--------------------------------------
A candidate cross-store match is accepted only when ALL of the following hold:

  1. SAME normalised brand (see extract_brand()).
  2. SAME product_type_name (exact, normalised lowercase).
  3. SAME gender where BOTH items have a known gender (men/women/unisex).
     If either item's gender is "unknown" / None, the gender check is skipped.
  4. Normalised-title fuzzy similarity >= TITLE_SIMILARITY_THRESHOLD (0.90).
     Titles are stripped of the brand prefix and normalised before comparison.

The threshold 0.90 is deliberately conservative.  A genuine cross-listing of the
same product across two stores would have a nearly identical title (same brand,
same product name, tiny wording variation).  The hard-negative pairs in this
catalogue score 0.20–0.55 on title similarity, well below the gate.

Performance guard
-----------------
To avoid an O(n^2) scan per request over ~44k items on the hot path, a
brand-to-stores map is precomputed at module load time.  Matching is skipped
entirely for single-store brands (which is ~all items) and only attempted for
the handful of brand tokens that appear in multiple stores.

Snapshot price labeling
------------------------
Prices in the returned match structs are labelled ``is_snapshot_price=True``.
They are NOT real-time.  The frontend must display a snapshot disclaimer.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from typing import Any

import pandas as pd

from src.config.stores import build_pdp_url, get_store_display_name

# ---------------------------------------------------------------------------
# Constants — threshold + rationale documented inline
# ---------------------------------------------------------------------------

# Minimum title-similarity score (SequenceMatcher ratio) to accept a match.
# Rationale: genuine same-product cross-listings share near-identical titles
# (same brand + same product name, possibly tiny suffix variation).  Real hard
# negatives in this catalogue score 0.20–0.55.  Setting the bar at 0.90 makes
# the gate practically unreachable for different products that happen to share
# a brand token (e.g. "Fusion" football shoes vs "Fusion Crest" shirt).
TITLE_SIMILARITY_THRESHOLD: float = 0.90

# Minimum overall confidence required to surface a match to callers.
# Currently tied to TITLE_SIMILARITY_THRESHOLD (the only numeric signal), but
# callers can pass a higher floor for extra conservatism.
DEFAULT_MIN_CONFIDENCE: float = TITLE_SIMILARITY_THRESHOLD

# Single-brand D2C stores where the brand is always the store brand.
# Virgio stores sub-brands before the pipe (e.g. "Brakin|100% Cotton ..."),
# handled specially in extract_brand().
_D2C_STORE_BRANDS: dict[str, str] = {
    "snitch": "snitch",
    "fashor": "fashor",
    "powerlook": "powerlook",
}

# Known gender values that carry real signal.  "unknown" / None are skipped.
_KNOWN_GENDERS: frozenset[str] = frozenset({"men", "women", "unisex"})


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation.

    Produces a clean normalised string for fuzzy comparison.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()
    # Remove punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_brand(prod_name: str, store: str) -> str:
    """Extract and normalise the brand token for one catalogue item.

    Parameters
    ----------
    prod_name:
        The raw ``prod_name`` string from the catalogue row.
    store:
        The store slug (e.g. ``"snitch"``, ``"myntra"``).

    Returns
    -------
    str
        Lowercase normalised brand string, e.g. ``"snitch"``, ``"free authority"``.

    Notes
    -----
    - Single-brand D2C stores (snitch, fashor, powerlook): the brand is always
      the store name itself, regardless of what the prod_name says (snitch
      product names often begin with a colour adjective, not a brand name).
    - Virgio: sub-brands appear as ``"BrandName|product description"``.  The
      brand is the text before the first ``|``, normalised.
    - Myntra / Flipkart: the brand is the leading token(s) of prod_name up to
      the first gender word (``Men``/``Women``) or the first product-descriptor
      keyword.  We take the first 1–2 space-separated tokens as a pragmatic
      heuristic; longer multi-word brands (e.g. "all about you",
      "Free Authority") are captured by taking everything before the first
      gender keyword.
    """
    store = (store or "").lower().strip()

    # D2C single-brand stores — brand is always the store
    if store in _D2C_STORE_BRANDS:
        return _D2C_STORE_BRANDS[store]

    # Virgio — sub-brand before pipe
    if store == "virgio":
        if "|" in prod_name:
            return _normalise_text(prod_name.split("|")[0].strip())
        # Fallback: first token
        return _normalise_text(prod_name.split()[0]) if prod_name.split() else "virgio"

    # Myntra / Flipkart — extract leading brand from prod_name.
    # Strategy: take all leading tokens up to (but not including) any gender or
    # descriptor keyword that signals the start of the product description.
    _GENDER_KEYWORDS = {"men", "women", "man", "woman", "boys", "girls", "unisex", "kids"}
    _DESC_KEYWORDS = {
        "round", "polo", "v-neck", "solid", "printed", "slim", "regular", "relaxed",
        "skinny", "fit", "loose", "embroidered", "casual", "formal", "crew", "cotton",
        "linen", "viscose",
    }
    tokens = prod_name.split()
    brand_tokens: list[str] = []
    for tok in tokens:
        tok_lower = tok.lower().rstrip("s")  # handles "Mens" -> "men"
        if tok_lower in _GENDER_KEYWORDS or tok_lower in _DESC_KEYWORDS:
            break
        brand_tokens.append(tok)

    # Safety: if we consumed all tokens, use at most first 2 as brand
    if not brand_tokens or len(brand_tokens) == len(tokens):
        brand_tokens = tokens[:2]

    return _normalise_text(" ".join(brand_tokens))


def _strip_brand_from_title(title: str, brand: str) -> str:
    """Remove leading brand occurrence from title for fairer title similarity.

    Comparing "Free Authority Women Hooded Sweatshirt" vs
    "Free Authority Men Vest" would score higher than warranted because
    the shared brand prefix inflates the ratio.  Stripping the brand first
    forces the comparison to focus on the product-description part.
    """
    normalised = _normalise_text(title)
    brand_norm = brand.lower().strip()
    if normalised.startswith(brand_norm):
        normalised = normalised[len(brand_norm):].strip()
    return normalised


# ---------------------------------------------------------------------------
# Brand-to-stores precomputed index
# ---------------------------------------------------------------------------


def build_brand_stores_map(catalogue_df: pd.DataFrame) -> dict[str, set[str]]:
    """Precompute a mapping from normalised brand token -> set of stores carrying it.

    This is the critical performance guard: for the ~99.9% of items whose
    brand only appears in ONE store, find_cross_store_matches returns [] without
    any fuzzy comparison.  Only items whose brand appears in 2+ stores enter
    the full matching loop.

    Should be called once at startup on the loaded catalogue.
    """
    brand_stores: dict[str, set[str]] = {}
    for _, row in catalogue_df.iterrows():
        store = str(row.get("store", "") or "")
        prod_name = str(row.get("prod_name", "") or "")
        if not store or not prod_name:
            continue
        brand = extract_brand(prod_name, store)
        if brand not in brand_stores:
            brand_stores[brand] = set()
        brand_stores[brand].add(store)
    return brand_stores


# ---------------------------------------------------------------------------
# Core matcher
# ---------------------------------------------------------------------------


def find_cross_store_matches(
    item: dict[str, Any],
    catalogue_df: pd.DataFrame,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    brand_stores_map: dict[str, set[str]] | None = None,
    brand_index: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Find same-product listings in OTHER stores for one catalogue item.

    Precision-over-recall: returns an empty list unless all four matching
    criteria (brand, product_type, gender, title similarity) are satisfied
    above the conservative thresholds.

    Parameters
    ----------
    item:
        Dict representing the query item (must contain at minimum
        ``prod_name``, ``store``, ``product_type_name``, optionally
        ``gender``, ``price_inr``, ``pdp_handle``, ``article_id``).
    catalogue_df:
        Full unified catalogue DataFrame.
    min_confidence:
        Minimum confidence threshold; default is TITLE_SIMILARITY_THRESHOLD.
        Pass a higher value for extra conservatism.
    brand_stores_map:
        Pre-built brand->stores map from build_brand_stores_map().  If None,
        computed on-the-fly (slow — pass the precomputed map in production).
    brand_index:
        Pre-built brand->list[row_position] map for fast candidate lookup.
        When provided alongside brand_stores_map, the scan only visits rows
        whose brand matches the query brand, reducing complexity from O(n) to
        O(|brand_rows|) — the key performance optimisation for the eval scan.

    Returns
    -------
    list of dicts, each with keys:
        store, store_display, price_inr, pdp_url, confidence,
        article_id, prod_name, is_snapshot_price.
    Sorted by price_inr ascending (lowest first).  Returns [] when no
    cross-store match meets the confidence gate.

    Notes
    -----
    - Never returns a match from the SAME store as the query item.
    - ``is_snapshot_price=True`` on every returned struct signals to callers
      (and the frontend) that these prices are catalogue snapshots, not
      real-time.  Never imply real-time freshness.
    - The brand-bucket guard means the full fuzzy loop runs only for the small
      fraction of items (here ~0%) whose brand token appears in 2+ stores.
    - When brand_index is provided, the candidate loop only visits rows sharing
      the query brand, making the hot-path scan O(|brand_rows|) instead of O(n).
    """
    item_store = str(item.get("store", "") or "").lower().strip()
    item_prod_name = str(item.get("prod_name", "") or "")
    item_type = str(item.get("product_type_name", "") or "").lower().strip()
    item_gender = str(item.get("gender", "") or "").lower().strip()
    item_article_id = str(item.get("article_id", "") or "")

    if not item_prod_name or not item_store:
        return []

    item_brand = extract_brand(item_prod_name, item_store)
    item_title_stripped = _strip_brand_from_title(item_prod_name, item_brand)

    # Performance guard 1: skip matching if the brand is single-store.
    if brand_stores_map is not None:
        stores_with_brand = brand_stores_map.get(item_brand, set())
        if len(stores_with_brand) <= 1:
            # Brand only in one store — cannot have cross-store same-product.
            return []

    # Performance guard 2: when a brand_index is available, only iterate the
    # rows belonging to this brand rather than the full catalogue.  For brands
    # that appear in 2+ stores (the only case that reaches here), the brand_index
    # narrows the candidate pool from ~44k to tens or hundreds of rows.
    if brand_index is not None and item_brand in brand_index:
        candidate_positions = brand_index[item_brand]
        candidate_rows = catalogue_df.iloc[candidate_positions]
    else:
        candidate_rows = catalogue_df

    matches: list[dict[str, Any]] = []

    for _, row in candidate_rows.iterrows():
        candidate_store = str(row.get("store", "") or "").lower().strip()
        if not candidate_store or candidate_store == item_store:
            continue

        candidate_article_id = str(row.get("article_id", "") or "")
        if candidate_article_id == item_article_id:
            continue

        candidate_prod_name = str(row.get("prod_name", "") or "")
        if not candidate_prod_name:
            continue

        # Gate 1: same normalised brand
        candidate_brand = extract_brand(candidate_prod_name, candidate_store)
        if candidate_brand != item_brand:
            continue

        # Gate 2: same product_type_name (normalised)
        candidate_type = str(row.get("product_type_name", "") or "").lower().strip()
        if candidate_type != item_type:
            continue

        # Gate 3: gender check (only when both sides have known genders)
        candidate_gender = str(row.get("gender", "") or "").lower().strip()
        if (
            item_gender in _KNOWN_GENDERS
            and candidate_gender in _KNOWN_GENDERS
            and item_gender != candidate_gender
        ):
            continue

        # Gate 4: high fuzzy title similarity (brand-stripped)
        candidate_title_stripped = _strip_brand_from_title(candidate_prod_name, candidate_brand)
        ratio = difflib.SequenceMatcher(
            None, item_title_stripped, candidate_title_stripped
        ).ratio()
        if ratio < min_confidence:
            continue

        # All gates passed — build match struct
        candidate_price = row.get("price_inr")
        candidate_pdp_handle = row.get("pdp_handle")
        candidate_row_dict: dict[str, Any] = (
            row.to_dict() if hasattr(row, "to_dict") else dict(row)
        )
        pdp_url = build_pdp_url(candidate_store, candidate_row_dict)

        # Normalise price: pandas stores None in float columns as NaN; convert to None.
        price_val: float | None
        if candidate_price is not None and not (
            isinstance(candidate_price, float) and math.isnan(candidate_price)
        ):
            price_val = float(candidate_price)
        else:
            price_val = None

        matches.append(
            {
                "store": candidate_store,
                "store_display": get_store_display_name(candidate_store) or candidate_store,
                "price_inr": price_val,
                "pdp_url": pdp_url,
                "pdp_handle": str(candidate_pdp_handle) if candidate_pdp_handle else None,
                "confidence": round(ratio, 4),
                "article_id": candidate_article_id,
                "prod_name": candidate_prod_name,
                "is_snapshot_price": True,  # prices are catalogue snapshots, not real-time
            }
        )

    # Sort by price ascending (lowest-first), then by confidence descending as tiebreaker.
    # Items without a price sort last.
    matches.sort(
        key=lambda m: (
            m["price_inr"] is None,
            m["price_inr"] if m["price_inr"] is not None else float("inf"),
            -m["confidence"],
        )
    )

    return matches


# ---------------------------------------------------------------------------
# Brand index (brand -> list of row positions for fast candidate lookup)
# ---------------------------------------------------------------------------


def build_brand_index(catalogue_df: pd.DataFrame) -> dict[str, list[int]]:
    """Precompute a brand -> [row_position, ...] index for O(|brand|) candidate lookup.

    When used alongside brand_stores_map in find_cross_store_matches(), this
    narrows the candidate scan from O(44k) to O(|brand_rows|) for multi-store
    brands.  The typical multi-store brand has tens to hundreds of rows, vs
    44k for a full scan.

    Should be called once at startup together with build_brand_stores_map().
    """
    brand_index: dict[str, list[int]] = {}
    for pos, (_, row) in enumerate(catalogue_df.iterrows()):
        store = str(row.get("store", "") or "")
        prod_name = str(row.get("prod_name", "") or "")
        if not store or not prod_name:
            continue
        brand = extract_brand(prod_name, store)
        if brand not in brand_index:
            brand_index[brand] = []
        brand_index[brand].append(pos)
    return brand_index


# ---------------------------------------------------------------------------
# Cached brand-stores map (module-level singleton for hot-path use)
# ---------------------------------------------------------------------------


# We can't lru_cache a DataFrame directly (unhashable), so we use module-level
# variables instead.
_brand_stores_cache: dict[str, set[str]] | None = None
_brand_index_cache: dict[str, list[int]] | None = None
_brand_stores_cache_size: int = -1


def get_cached_brand_stores_map(catalogue_df: pd.DataFrame) -> dict[str, set[str]]:
    """Return a cached brand-to-stores map, rebuilding only when catalogue changes.

    Uses catalogue row count as a lightweight cache key.  Good enough for the
    read-only hot path where the catalogue is loaded once at startup.
    """
    global _brand_stores_cache, _brand_stores_cache_size  # noqa: PLW0603
    current_len = len(catalogue_df)
    if _brand_stores_cache is None or _brand_stores_cache_size != current_len:
        _brand_stores_cache = build_brand_stores_map(catalogue_df)
        _brand_stores_cache_size = current_len
    return _brand_stores_cache


def get_cached_brand_index(catalogue_df: pd.DataFrame) -> dict[str, list[int]]:
    """Return a cached brand-to-row-positions index, rebuilding only when catalogue changes.

    Pair with get_cached_brand_stores_map() and pass both to find_cross_store_matches()
    for maximum performance on the hot path.
    """
    global _brand_index_cache, _brand_stores_cache_size  # noqa: PLW0603
    current_len = len(catalogue_df)
    if _brand_index_cache is None or _brand_stores_cache_size != current_len:
        _brand_index_cache = build_brand_index(catalogue_df)
        # Note: _brand_stores_cache_size is shared as the cache invalidation key.
        # get_cached_brand_stores_map() must be called first (or alongside) to
        # set the correct size.
    return _brand_index_cache
