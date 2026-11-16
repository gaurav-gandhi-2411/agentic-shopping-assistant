"""Build-time catalogue hygiene (Wave 1b V3(b2)(b)): category, non-fashion drop, quarantine.

Deterministic, rule-based, no LLM and no network. Used by ``scripts/build_unified_index.py``
(per-brand, after Phase A cleaning) and by ``src/retrieval/hybrid_search.py`` (to decide whether
a query's product type is itself an accessory/jewellery/footwear request).

Three outputs per row, all derived from ``product_type_name`` and the title (``prod_name``):

``category``
    One of ``garment`` / ``accessory`` / ``jewellery`` / ``footwear`` / ``non_fashion`` /
    ``unknown``. ``unknown`` means no rule fired (it is never excluded by the retrieval guard,
    so a rule gap fails open rather than hiding items).
``garment_type``
    A finer canonical type (``kurta``, ``lehenga``, ``earrings``, ``bag``, ...) or None.
``quarantine_reason``
    Non-empty when the title and the merchant product type disagree on whether the item is a
    garment at all (title "Lehenga", type "Bangles"). Which of the two is right is not decidable
    without a human, so the row is excluded from the index instead of guessed at.

Why the title head noun: titles carry modifiers and accessories ("Maroon Kundan Choker for
Lehenga", "Kurta with Palazzo and Dupatta"). The head noun is the LAST lexicon word before the
first preposition/conjunction, which resolves both of those correctly and also resolves
"Lehenga Box" to a box (non-garment).

Catalogue columns used: ``product_type_name``, ``prod_name``. The catalogue and the raw Shopify
CSVs carry no ``tags`` column, so tags cannot be used; ``detail_desc`` is deliberately not used
(descriptions mention coordinating garments and jewellery constantly, which would add noise).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.catalogue.hygiene_lexicon import GENERIC_TYPES, LEXICON

CATEGORIES: tuple[str, ...] = (
    "garment",
    "accessory",
    "jewellery",
    "footwear",
    "non_fashion",
    "unknown",
)

# Categories a garment query must not return unless it asks for them (retrieval guard).
GARMENT_QUERY_EXCLUDED_CATEGORIES: frozenset[str] = frozenset(
    {"accessory", "jewellery", "non_fashion"}
)


@dataclass(frozen=True)
class CategoryResult:
    """Outcome of :func:`derive_category` for one item."""

    category: str
    garment_type: str | None
    source: str  # "type" | "title" | "none"
    quarantine_reason: str = ""


_NF = "non_fashion"
_J = "jewellery"
_F = "footwear"
_A = "accessory"
_G = "garment"


def _norm(text: object) -> str:
    """Lower-case, collapse every non-alphanumeric run to one space."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


_COMPILED: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (gtype, cat, re.compile(rf"\b(?:{pat})\b")) for gtype, cat, pat in LEXICON
)


# Cut the title where the head noun phrase ends: "<head> for/with/and/& <extras>".
_HEAD_CUT = re.compile(r"\b(?:for|with|to|and|in|by|from|matching|featuring|paired|combo|pack)\b")


_TAIL_OK_WORDS: frozenset[str] = frozenset(
    {"size", "lid", "set", "sets", "combo", "pack", "pair", "ml", "gm", "gms", "pcs", "of"}
)


def _is_tail_ok(tail: str) -> bool:
    """True when only SKU-like words follow a non-fashion noun (size, count, 1-2 char codes).

    A non-fashion noun is the head only if it ends the phrase: "Bangle Box T 69" yes,
    "Box Lakshmi Necklace" no.
    """
    return all(len(w) <= 2 or w.isdigit() or w in _TAIL_OK_WORDS for w in tail.split())


def _last_match(text: str, skip: frozenset[str] = frozenset()) -> tuple[str, str, int] | None:
    """Last lexicon hit in *text* as (garment_type, category, end position), or None.

    Last wins (head noun of an English noun phrase). Among hits that END at the same position
    (overlapping alternatives such as "sets" inside "kurta sets"), the longest match wins.
    Lexicon groups named in *skip* are ignored.
    """
    best: tuple[int, int, str, str] | None = None  # (end, length, gtype, category)
    for gtype, cat, rx in _COMPILED:
        if gtype in skip:
            continue
        for m in rx.finditer(text):
            if cat == _NF and not _is_tail_ok(text[m.end() :]):
                continue  # "Box Lakshmi necklace", "Bottle Green kundan": a modifier, not the head
            cand = (m.end(), m.end() - m.start(), gtype, cat)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return None
    return best[2], best[3], best[0]


def classify_type(product_type: object) -> tuple[str, str] | None:
    """(garment_type, category) from the merchant product type alone, None when generic/no hit."""
    t = _norm(product_type)
    if t in GENERIC_TYPES:
        return None
    hit = _last_match(t)
    return (hit[0], hit[1]) if hit else None


def title_head(title: object) -> tuple[str, str] | None:
    """(garment_type, category) of the title's head noun, None when no lexicon word is found."""
    t = _norm(title)
    if not t:
        return None
    cut = _HEAD_CUT.search(t)
    prefix = t[: cut.start()].strip() if cut and cut.start() > 0 else t
    hit = _last_match(prefix)
    if hit is None and prefix != t:
        hit = _last_match(t)
    if hit is not None and hit[0] == "hamper":
        # "Kurta Rakhi Gift Box": a garment sold inside a gift box is still the garment.
        alt = _last_match(prefix, skip=frozenset({"hamper", "box", "rakhi"}))
        if alt is not None and alt[1] == _G:
            hit = alt
    if hit is not None and hit[0] == "dupatta" and re.search(r"\bsets?\b", prefix):
        # "Kurta Dupatta Set": the dupatta is a part of a garment set, not the product itself.
        alt = _last_match(prefix, skip=frozenset({"dupatta"}))
        if alt is not None and alt[1] == _G:
            hit = alt
    return (hit[0], hit[1]) if hit else None


def derive_category(product_type: object, title: object) -> CategoryResult:
    """Derive ``category``, a finer ``garment_type`` and a quarantine reason for one item.

    Precedence:
      1. The title's head noun is non-fashion (box, trunk, hamper, gift card, ...): non_fashion,
         whatever the product type says ("Lehenga Box" typed lehenga).
      2. A specific product type decides category and garment_type (the merchant's own taxonomy).
         If the title head disagrees on garment-vs-not-garment the item is quarantined.
      3. A generic/unknown product type: the title head decides.
      4. Otherwise ``unknown``.
    """
    head = title_head(title)
    typ = classify_type(product_type)

    if head is not None and head[1] == _NF:
        return CategoryResult(_NF, head[0], "title")
    if typ is not None:
        gtype, cat = typ
        if head is not None and head[1] in (_G, _J, _F, _A) and cat in (_G, _J, _F, _A):
            if gtype == "dupatta" and head[1] == _G:
                # The feed normaliser typed "<Garment> Set With Dupatta" listings as dupatta
                # (libas: ~3.5k "Suit Set With Dupatta"); the bundle is the garment, so the
                # title is decisive here and nothing is quarantined.
                return CategoryResult(_G, head[0], "title")
            # A dupatta/accessory named last in a garment set title is not a contradiction.
            if (head[1] == _G) != (cat == _G) and not (cat == _G and head[1] == _A):
                reason = f"title head '{head[0]}' ({head[1]}) vs product type '{gtype}' ({cat})"
                return CategoryResult(cat, gtype, "type", reason)
        return CategoryResult(cat, gtype, "type")
    if head is not None:
        return CategoryResult(head[1], head[0], "title")
    return CategoryResult("unknown", None, "none")


def is_garment_type_name(product_type_name: object) -> bool:
    """True when a catalogue/intent type name (``kurta``, ``lehenga``) is a garment type.

    Used by the retrieval guard: only a query whose type resolves to a garment is shielded from
    accessories/jewellery/non-fashion. Unknown names are NOT garments, so the guard stays off.
    """
    hit = classify_type(product_type_name)
    return hit is not None and hit[1] == _G


def apply_hygiene(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Add ``category``/``garment_type`` and drop non-fashion + quarantined rows.

    Returns ``(kept, report)`` where ``report`` holds two frames with the columns
    ``article_id, store, prod_name, product_type_name, price_inr, category, reason``:
    ``dropped`` (non-fashion rows) and ``quarantined`` (title-vs-type contradictions).
    Row order of ``kept`` follows the input; ``reset_index`` is applied.
    """
    pts = (
        df["product_type_name"]
        if "product_type_name" in df.columns
        else pd.Series("", index=df.index)
    )
    names = df["prod_name"] if "prod_name" in df.columns else pd.Series("", index=df.index)
    results = [derive_category(p, n) for p, n in zip(pts, names, strict=True)]
    out = df.copy()
    out["category"] = [r.category for r in results]
    out["garment_type"] = [r.garment_type for r in results]
    reasons = pd.Series([r.quarantine_reason for r in results], index=out.index)

    is_drop = out["category"] == _NF
    is_quar = (reasons != "") & ~is_drop

    cols = [
        c
        for c in ("article_id", "store", "prod_name", "product_type_name", "price_inr", "category")
        if c in out.columns
    ]
    dropped = out.loc[is_drop, cols].assign(reason="non_fashion")
    quarantined = out.loc[is_quar, cols].assign(reason=reasons[is_quar])
    kept = out.loc[~is_drop & ~is_quar].reset_index(drop=True)
    return kept, {"dropped": dropped, "quarantined": quarantined}
