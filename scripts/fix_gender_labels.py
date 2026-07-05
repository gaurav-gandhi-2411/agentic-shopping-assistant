"""One-off idempotent migration: backfill 'unknown' gender labels from title/search_text.

Context
-------
Live bug: refining "white shirt men" -> "in blue now" returned items whose catalogue
gender is "unknown" (mostly store=globalrepublic).  hybrid_search.py now hard-excludes
unknown-gender rows whenever an explicit gender filter is set (see src/retrieval/
hybrid_search.py), which is the correctness fix for retrieval.  This script is the
complementary data fix: where the title/description clearly states a gender, backfill
the 'unknown' label so those rows aren't needlessly dropped from gender-filtered
searches going forward.

Per-brand gender value_counts (inspected 2026-07, all 9 brands)::

    hm              — no gender column (not applicable)
    myntra          — women=14207, men=123                         (0.0% unknown)
    flipkart        — men=8762, unknown=293, women=26               (3.2% unknown)
    snitch          — men=15000                                     (0.0% unknown)
    fashor          — women=3615, men=3                             (0.0% unknown)
    powerlook       — men=922                                       (0.0% unknown)
    virgio          — women=1765, men=45                            (0.0% unknown)
    berrylush       — women=5388, men=14                            (0.0% unknown)
    globalrepublic  — unknown=3302, women=290, men=148              (88.3% unknown)  <- target
    libas           — women=14862, men=5                            (0.0% unknown)

Only globalrepublic clears an "unknown-heavy" bar (>10% unknown); every other brand's
unknown rows are a handful of genuinely ambiguous SKUs, not a systemic labelling gap,
so they are left untouched.  Use ``--brand <slug>`` to inspect (dry-run only) any other
brand's profile before adding it to TARGET_BRANDS.

Usage
-----
    python scripts/fix_gender_labels.py            # dry run (default): prints
                                                     # before/after value_counts, writes nothing
    python scripts/fix_gender_labels.py --apply     # writes catalogue.parquet back in place

IMPORTANT: this script must NEVER touch data/processed/unified/catalogue.parquet.
The next unified-index rebuild (scripts/build_unified_index.py) picks up the corrected
per-brand parquet files automatically.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("fix_gender_labels")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).parent.parent
_CATALOGUE_DIR = _REPO_ROOT / "data" / "processed"

# Brands verified (via value_counts inspection, 2026-07) to have an unknown-heavy gender
# column worth backfilling.  See the module docstring table above.
TARGET_BRANDS: tuple[str, ...] = ("globalrepublic",)

# The unified catalogue is a REBUILD ARTIFACT, never a per-brand source file — this
# script must refuse to touch it even if someone passes --brand unified by mistake.
_FORBIDDEN_BRAND = "unified"

# Word-boundary keyword sets.  IMPORTANT: "women" contains "men" as a literal substring
# ("wo|men"), so a naive `"men" in text` check would misclassify every women's item as
# men's.  Using \b-anchored regex avoids this: there is no word boundary between "wo"
# and "men" inside "women" (both are word characters), so _MEN_RE structurally cannot
# match inside a "women" token.  The WOMEN pattern is still checked first as defence in
# depth.  "unisex" is deliberately NOT matched by either pattern (no "men" or "women"
# token appears in "unisex") so unisex-labelled items are left as 'unknown'.
_WOMEN_RE = re.compile(
    r"\b(women'?s?|woman'?s?|ladies|lady'?s?|female|girls?|dress(?:es)?|saree(?:s)?|"
    r"kurti(?:s)?|lehenga(?:s)?|anarkali(?:s)?|blouse(?:s)?|skirt(?:s)?|gown(?:s)?)\b",
    re.IGNORECASE,
)
_MEN_RE = re.compile(
    r"\b(men'?s?|man'?s?|male|boys?|gents?|sherwani(?:s)?)\b",
    re.IGNORECASE,
)


def infer_gender(text: str | None) -> str | None:
    """Infer 'women' or 'men' from free text using conservative keyword matching.

    Returns ``None`` when neither pattern matches — callers should leave the row's
    gender as 'unknown' in that case rather than guessing.

    The WOMEN pattern is checked before MEN so any (already structurally excluded by
    \\b boundaries) overlap favours women, per the "women contains men" gotcha this
    function is designed to avoid.
    """
    if not text:
        return None
    if _WOMEN_RE.search(text):
        return "women"
    if _MEN_RE.search(text):
        return "men"
    return None


# Inline conservative self-checks, run at import time so a regex regression fails fast.
assert infer_gender("Women's Top") == "women"
assert infer_gender("Men's Shirt") == "men"
assert infer_gender("Unisex Tee") is None
assert infer_gender("Womens Formal Shirt") == "women"
assert infer_gender("Mens Formal Shirt") == "men"
assert infer_gender("Black Cotton Blend Casual Shirt") is None  # no gender token — stays unknown
assert infer_gender("") is None
assert infer_gender(None) is None


def fix_brand_gender(brand: str, catalogue_dir: Path, *, apply: bool) -> pd.DataFrame:
    """Backfill 'unknown' gender rows for *brand* from title/search_text keywords.

    Loads ``catalogue_dir/<brand>/catalogue.parquet``, infers gender for rows currently
    labelled 'unknown', and — only when ``apply=True`` — writes the corrected frame back
    to the same path.  Always prints a before/after ``value_counts()`` table.

    Args:
        brand:         Store slug (must not be "unified" — see module docstring).
        catalogue_dir: Base directory containing ``<brand>/catalogue.parquet``.
        apply:         When False (default), dry-run only — no file is written.

    Returns:
        The (possibly modified in-memory) DataFrame, for callers/tests that want to
        inspect the result without touching disk.

    Raises:
        ValueError: If ``brand == "unified"`` — this script must never touch the
            unified rebuild artifact.
        FileNotFoundError: If no catalogue parquet exists for *brand*.
    """
    if brand.lower() == _FORBIDDEN_BRAND:
        raise ValueError(
            "fix_gender_labels.py must never touch data/processed/unified/catalogue.parquet "
            "— it is a rebuild artifact, not a per-brand source file."
        )

    path = catalogue_dir / brand / "catalogue.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No catalogue found for brand={brand!r} at {path}")

    df = pd.read_parquet(path)
    if "gender" not in df.columns:
        logger.warning("brand=%s has no gender column — skipping", brand)
        return df

    before_counts = df["gender"].value_counts(dropna=False)

    unknown_mask = df["gender"].astype(str).str.lower() == "unknown"
    combined_text = (
        df.loc[unknown_mask, "prod_name"].fillna("")
        + " "
        + df.loc[unknown_mask, "search_text"].fillna("")
        if "search_text" in df.columns
        else df.loc[unknown_mask, "prod_name"].fillna("")
    )
    inferred = combined_text.map(infer_gender)
    n_backfilled = int(inferred.notna().sum())

    # Only overwrite rows where a confident inference was made; everything else stays
    # 'unknown' exactly as before (idempotent — a second run backfills the same rows to
    # the same values, then n_backfilled_new below would be 0).
    df.loc[unknown_mask, "gender"] = df.loc[unknown_mask, "gender"].where(
        inferred.isna(), inferred
    )

    after_counts = df["gender"].value_counts(dropna=False)

    print(f"\n=== {brand} ===")
    print(f"path: {path}")
    print(f"rows backfilled: {n_backfilled} / {int(unknown_mask.sum())} unknown rows")
    print("--- before ---")
    print(before_counts.to_string())
    print("--- after ---")
    print(after_counts.to_string())

    if apply:
        df.to_parquet(path, index=False)
        logger.info("WROTE %s (%d rows backfilled)", path, n_backfilled)
    else:
        logger.info("DRY RUN — no file written for brand=%s (pass --apply to write)", brand)

    return df


def main() -> None:
    """CLI entry point: dry-run by default, ``--apply`` to write changes in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the corrected catalogue.parquet back in place. Default is dry-run (print only).",
    )
    parser.add_argument(
        "--brand",
        action="append",
        dest="brands",
        help=(
            "Override TARGET_BRANDS; may be passed multiple times. Useful to inspect a "
            "brand's value_counts profile (dry-run only recommended) before deciding "
            "whether to add it to TARGET_BRANDS."
        ),
    )
    args = parser.parse_args()

    brands = args.brands or list(TARGET_BRANDS)
    for brand in brands:
        fix_brand_gender(brand, _CATALOGUE_DIR, apply=args.apply)


if __name__ == "__main__":
    main()
