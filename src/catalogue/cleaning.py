"""Phase A index-quality cleaning helpers — deterministic, rule-based, no LLM calls.

Used by ``scripts/build_unified_index.py`` (build-time catalogue cleanup) and by
``src/agents/graph.py`` / ``src/retrieval/hybrid_search.py`` (runtime fabric-bolt
exclusion), so the "what counts as a fabric bolt" rule is defined exactly once.

Covers:
    - Saree reclassification: rows tagged ``fabric_material`` that are actually
      finished sarees sold "with blouse piece" (a bundled accessory, not the
      product itself) get reclassified to ``product_type_name="saree"``.
    - Fabric-bolt text exclusion: the shared runtime predicate used to keep true
      fabric bolts (unstitched material, dress material, fabric piece) out of
      search results without also excluding finished sarees.
    - Colour backfill: extracts a canonical ``colour_group_name`` from
      ``prod_name`` (checked first) then ``detail_desc`` for rows with a
      null/empty colour, reusing the intent-parser colour vocabulary so query-side
      and catalogue-side colour matching share one source of truth.
    - Mojibake cleanup: strips non-breaking-space runs and unrecoverable
      replacement-character (U+FFFD) artifacts from text columns.
"""

from __future__ import annotations

import re

import pandas as pd

from src.agents.intent_parser import _COLOUR_SORTED

# ---------------------------------------------------------------------------
# Saree / fabric-bolt classification
# ---------------------------------------------------------------------------

# Saree word — appears both as a genuine garment noun ("Silk Saree...") and,
# confusingly, as part of brand names such as "Saree Mall" or "Pandadi Saree".
# It is therefore only used IN COMBINATION with the "blouse piece" phrase below
# (never on its own) to decide reclassification — see reclassify_finished_sarees.
SAREE_WORD_RE = re.compile(r"\bsarees?\b|\bsari\b", re.IGNORECASE)

# "Blouse piece" is the fabric swatch bundled with a saree purchase (stitched or
# unstitched) — its presence does NOT make the saree itself unwearable. Real
# catalogue examples: "Saree With Unstitched Blouse Piece", "Saree & Embellished
# Blouse Piece". These are finished, shoppable sarees.
BLOUSE_PIECE_RE = re.compile(r"blouse\s*piece", re.IGNORECASE)

# True fabric-bolt signals — always mean "not a wearable garment", regardless of
# whether the word "saree" also appears (e.g. "Saree Mall ... Unstitched Dress
# Material" is a fabric bolt sold by a brand whose name happens to contain
# "Saree"; "Unstitched Half Saree" is genuinely unstitched attire fabric).
TRUE_FABRIC_RE = re.compile(r"\bunstitched\b|dress material|fabric piece", re.IGNORECASE)


def is_fabric_bolt_text(text: str | None) -> bool:
    """Return True when *text* describes a fabric bolt, not a wearable garment.

    Single source of truth for the runtime exclusion previously duplicated (and
    inconsistently applied) in ``src/agents/graph.py`` and
    ``src/retrieval/hybrid_search.py``. A "blouse piece" mention alone is only a
    fabric-bolt signal when the text is NOT also a finished saree (i.e. does not
    also contain the word "saree"/"sari") — see module docstring.
    """
    if not text:
        return False
    if TRUE_FABRIC_RE.search(text):
        return True
    if BLOUSE_PIECE_RE.search(text):
        return not SAREE_WORD_RE.search(text)
    return False


def reclassify_finished_sarees(
    df: pd.DataFrame,
    *,
    type_col: str = "product_type_name",
    name_col: str = "prod_name",
) -> tuple[pd.DataFrame, int]:
    """Reclassify rows tagged ``fabric_material`` that are finished sarees.

    A row is reclassified to ``saree`` when it is currently tagged
    ``fabric_material`` AND its name contains BOTH a saree word and the phrase
    "blouse piece" — the signature of a finished saree bundled with a blouse
    fabric swatch, as opposed to a genuine fabric bolt (which may contain the
    word "saree" only as part of a brand name, e.g. "Saree Mall").

    Returns
    -------
    (df, n_reclassified) — a copy of *df* with the fix applied, and the count of
    rows that were reclassified.
    """
    out = df.copy()
    names = out[name_col].fillna("").astype(str)
    is_fabric = out[type_col].fillna("").str.lower() == "fabric_material"
    mask = is_fabric & names.str.contains(SAREE_WORD_RE) & names.str.contains(BLOUSE_PIECE_RE)
    n = int(mask.sum())
    if n:
        out.loc[mask, type_col] = "saree"
    return out, n


def drop_true_fabric_material(
    df: pd.DataFrame,
    *,
    type_col: str = "product_type_name",
) -> tuple[pd.DataFrame, int]:
    """Drop rows still tagged ``fabric_material`` after saree reclassification.

    These are genuine fabric bolts / unstitched dress material — not shoppable
    garments — and are removed entirely from the index rather than merely
    filtered at query time. Call AFTER :func:`reclassify_finished_sarees`.

    Returns (df, n_dropped).
    """
    is_fabric = df[type_col].fillna("").str.lower() == "fabric_material"
    n = int(is_fabric.sum())
    out = df.loc[~is_fabric].reset_index(drop=True)
    return out, n


# ---------------------------------------------------------------------------
# Colour backfill
# ---------------------------------------------------------------------------

# Trailing parenthetical colour-list pattern used by Flipkart titles, e.g.
# "Men Solid Cotton Satin Blend Straight Kurta  (Maroon, Dark Blue, Black, Pink)".
# Only the FIRST colour in the list is taken (arbitrary tie-break — the feed does
# not indicate which colour swatch the row's image/price corresponds to).
_TRAILING_PAREN_RE = re.compile(r"\(([A-Za-z][A-Za-z\s]*(?:,\s*[A-Za-z][A-Za-z\s]*)*)\)\s*$")


def _scan_colour(text: str) -> str | None:
    """Longest-match, word-boundary scan of *text* against the shared colour vocabulary."""
    if not text:
        return None
    lower = text.lower()
    for phrase, canonical in _COLOUR_SORTED:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        if re.search(pattern, lower):
            return canonical
    return None


def extract_colour(prod_name: str | None, detail_desc: str | None = None) -> str | None:
    """Extract a canonical colour_group_name from *prod_name*, falling back to *detail_desc*.

    Order of precedence:
        1. Trailing parenthetical colour list in *prod_name* (Flipkart convention) —
           first colour in the list wins.
        2. Longest-match word-boundary scan over the whole of *prod_name*.
        3. Same scan over *detail_desc*.

    Returns None when no known colour phrase is found anywhere.
    """
    name = prod_name or ""
    paren = _TRAILING_PAREN_RE.search(name)
    if paren:
        first_token = paren.group(1).split(",")[0].strip()
        hit = _scan_colour(first_token)
        if hit:
            return hit

    hit = _scan_colour(name)
    if hit:
        return hit

    return _scan_colour(detail_desc or "")


def backfill_colours(
    df: pd.DataFrame,
    *,
    colour_col: str = "colour_group_name",
    name_col: str = "prod_name",
    desc_col: str = "detail_desc",
) -> tuple[pd.DataFrame, int]:
    """Fill null/empty *colour_col* values by extracting colour from name/description.

    Returns (df, n_filled).
    """
    out = df.copy()
    current = out[colour_col] if colour_col in out.columns else pd.Series("", index=out.index)
    is_null = current.isna() | (current.astype(str).str.strip() == "")

    if not is_null.any():
        return out, 0

    names = out.loc[is_null, name_col].fillna("").astype(str)
    descs = (
        out.loc[is_null, desc_col].fillna("").astype(str)
        if desc_col in out.columns
        else pd.Series("", index=out.loc[is_null].index)
    )
    filled = [extract_colour(n, d) for n, d in zip(names, descs)]

    if colour_col not in out.columns:
        out[colour_col] = None
    out.loc[is_null, colour_col] = filled
    n_filled = sum(1 for v in filled if v is not None)
    return out, n_filled


# ---------------------------------------------------------------------------
# Mojibake cleanup
# ---------------------------------------------------------------------------

# Non-breaking space (U+00A0) — Flipkart titles use runs of these before a
# trailing parenthetical, e.g. "Sneakers For Men\xa0\xa0(Grey)".
_NBSP_RE = re.compile(" +")

# Unicode replacement character (U+FFFD) — marks an unrecoverable decode failure
# (the original byte is gone by the time it reaches this stage). A single
# occurrence between two letters is almost always a lost apostrophe
# ("men�s" -> "men's"); any other occurrence is stripped to a space.
_FFFD_APOSTROPHE_RE = re.compile(r"(?<=[a-zA-Z])�(?=[a-zA-Z])")
_FFFD_ANY_RE = re.compile("�+")

_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def fix_mojibake(text: str | None) -> str | None:
    """Deterministically clean nbsp runs and U+FFFD replacement-character artifacts.

    U+FFFD marks bytes that were already lost before this stage — there is no way
    to recover the original character — so this is a best-effort cleanup (collapse
    to an apostrophe when it sits between two letters, else drop it), not a true
    mojibake "fix". Also normalises non-breaking spaces and collapses whitespace.
    Returns *text* unchanged if it is None/empty.
    """
    if not text:
        return text
    cleaned = _NBSP_RE.sub(" ", text)
    cleaned = _FFFD_APOSTROPHE_RE.sub("'", cleaned)
    cleaned = _FFFD_ANY_RE.sub(" ", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def clean_mojibake_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("prod_name", "detail_desc"),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply :func:`fix_mojibake` to *columns*; returns (df, {column: n_changed})."""
    out = df.copy()
    stats: dict[str, int] = {}
    for col in columns:
        if col not in out.columns:
            continue
        original = out[col]
        cleaned = original.apply(fix_mojibake)
        n_changed = int((cleaned.fillna("") != original.fillna("")).sum())
        out[col] = cleaned
        stats[col] = n_changed
    return out, stats


# ---------------------------------------------------------------------------
# Derived-column recomputation (search_text / display_name / facets)
# ---------------------------------------------------------------------------
# Mirrors src/catalogue/loader.py::build_searchable_text exactly. Must be
# reapplied after any of the cleaning steps above (saree reclass, colour
# backfill, mojibake fix) change prod_name / product_type_name / colour_group_name,
# so BM25, the colour facet filter, and display_name all reflect the fixes.


def recompute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute search_text, display_name, and facets from current column values."""
    out = df.copy()

    out["search_text"] = (
        out["prod_name"].fillna("") + ". "
        + out["product_type_name"].fillna("") + ". "
        + out["colour_group_name"].fillna("") + ". "
        + out["department_name"].fillna("") + ". "
        + out["detail_desc"].fillna("")
    )

    out["display_name"] = (
        out["prod_name"].fillna("").str.strip()
        + " ("
        + out["colour_group_name"].fillna("").str.strip()
        + " "
        + out["product_type_name"].fillna("").str.strip()
        + ")"
    )

    out["facets"] = out.apply(
        lambda r: {
            "colour_group_name": r["colour_group_name"],
            "product_type_name": r["product_type_name"],
            "department_name": r["department_name"],
            "index_group_name": r["index_group_name"],
            "garment_group_name": r["garment_group_name"],
        },
        axis=1,
    )

    return out
