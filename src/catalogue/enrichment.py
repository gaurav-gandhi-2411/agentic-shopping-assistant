"""Phase-C attribute enrichment — season / occasion / style / fabric facet extraction.

Deterministic, rule-based regex/keyword extraction (no LLM, no network calls) for
four new facet keys not covered by ``cleaning.py``'s ``recompute_derived_columns``:
``season``, ``occasion_tag``, ``style_tag``, ``fabric``.

Used by ``scripts/enrich_attributes.py`` (the build-time driver, which adds an
Ollama fallback for rows these rules leave incomplete) and unit-tested standalone
here — mirrors the split between ``cleaning.py`` (pure rules) and
``build_unified_index.py`` (the I/O driver that calls it).

Design
------
Each facet has one or more ORDERED tiers of ``(compiled regex, canonical label)``
pairs — first match wins within a tier, and higher-confidence tiers are checked
before lower-confidence ones (e.g. an explicit "summer wear" mention beats an
inferred "linen + sleeveless" cue). Ambiguous or unsupported text returns
``None`` — never an invented label. ``merge_enrichment`` applies the same
never-invent policy to the LLM fallback's output: any value outside the fixed
vocab (typos, hallucinated categories) is discarded to ``None`` rather than kept.

Canonical values use lowercase words with spaces (matching ``detail_desc``
convention) so they tokenize identically whether they end up in a facet filter
or appended to ``search_text`` for BM25 (``SparseRetriever._tokenize`` splits on
non-alnum, so "ethnic fusion" and "ethnic_fusion" would tokenize the same way —
spaces are simply more readable in a free-text blob).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical vocab — the ONLY values these functions (and the Ollama fallback's
# validated output) may produce. Anything else collapses to None.
# ---------------------------------------------------------------------------

SEASON_VOCAB: frozenset[str] = frozenset({"summer", "winter", "monsoon", "all season"})
OCCASION_VOCAB: frozenset[str] = frozenset({"wedding", "festive", "party", "office", "casual"})
STYLE_VOCAB: frozenset[str] = frozenset(
    {"boho", "minimalist", "classic", "streetwear", "ethnic fusion", "athleisure"}
)
FABRIC_VOCAB: frozenset[str] = frozenset(
    {
        "chanderi", "khadi", "georgette", "chiffon", "denim", "muslin", "velvet",
        "satin", "crepe", "net", "corduroy", "tweed", "leather", "spandex",
        "viscose", "rayon", "modal", "cashmere", "wool", "linen", "silk", "cotton",
        "polyester", "nylon", "acrylic", "jersey",
    }
)

FACET_VOCAB: dict[str, frozenset[str]] = {
    "season": SEASON_VOCAB,
    "occasion_tag": OCCASION_VOCAB,
    "style_tag": STYLE_VOCAB,
    "fabric": FABRIC_VOCAB,
}

# ---------------------------------------------------------------------------
# Season — direct mentions checked before weaker fabric/sleeve inference.
# ---------------------------------------------------------------------------

_SEASON_DIRECT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmonsoon\b|\brain\s*coat\b|\brainwear\b|\bwaterproof\b", re.I), "monsoon"),
    (re.compile(r"\bsummer\b", re.I), "summer"),
    (re.compile(r"\bwinter\b", re.I), "winter"),
    (re.compile(r"\ball[- ]season\b", re.I), "all season"),
]

# Weaker inference from fabric weight / sleeve length — only consulted when no
# direct season word matched. Bare "cotton" or "full sleeve" alone is NOT
# included: both are worn across seasons and are too weak a signal on their own
# (see module docstring — honest None beats a fabricated tag).
_SEASON_FABRIC_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\blinen\b|\bsleeveless\b|\blightweight\b|\bbreathable\b", re.I), "summer"),
    (
        re.compile(
            r"\bwool(?:en|len)?\b|\bfleece\b|\bthermal\b|\bquilted\b|\bpadded\b|\bcorduroy\b",
            re.I,
        ),
        "winter",
    ),
]

# ---------------------------------------------------------------------------
# Occasion — "formal" and "sangeet" fold into "office" and "wedding"
# respectively (see docstring); order = most-specific occasion first so a
# wedding-lehenga description that also says "party wear" still tags "wedding".
# ---------------------------------------------------------------------------

_OCCASION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwedding\b|\bbridal\b|\bsangeet\b|\bmehendi\b|\bmehndi\b", re.I), "wedding"),
    (re.compile(r"\bfestive\b|\bfestival\b|\bdiwali\b|\bnavratri\b|\bpuja\b", re.I), "festive"),
    (re.compile(r"\bparty\s*wear\b|\bcocktail\b|\bparty\b", re.I), "party"),
    (
        re.compile(r"\boffice\s*wear\b|\bwork\s*wear\b|\bformal\s*wear\b|\bformal\b", re.I),
        "office",
    ),
    (
        re.compile(r"\bcasual\s*wear\b|\beveryday\s*wear\b|\bloungewear\b|\bcasual\b", re.I),
        "casual",
    ),
]

# ---------------------------------------------------------------------------
# Style — bare "classic" is too generic (frequently describes a collar/fit, not
# the garment's overall style register), so "classic" only fires on stronger
# phrases ("classic style/fit/look") or "timeless".
# ---------------------------------------------------------------------------

_STYLE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bboho\b|\bbohemian\b", re.I), "boho"),
    (re.compile(r"\bminimalist\b|\bminimal(?:istic)?\b", re.I), "minimalist"),
    (re.compile(r"\bstreetwear\b|\bstreet[- ]style\b", re.I), "streetwear"),
    (
        re.compile(r"\bindo[- ]western\b|\bethnic\s*fusion\b|\bfusion\s*wear\b", re.I),
        "ethnic fusion",
    ),
    (re.compile(r"\bathleisure\b", re.I), "athleisure"),
    (re.compile(r"\bclassic\s*(?:style|fit|look)\b|\btimeless\b", re.I), "classic"),
]

# ---------------------------------------------------------------------------
# Fabric — most-specific/distinctive fabric first (e.g. "chanderi" over the
# generic "cotton" it's often blended with) so a saree described as "70%
# cotton, 30% chanderi" is tagged with the more informative fabric. Within
# that ordering, distinctive weave/texture names (chanderi, khadi, georgette,
# muslin, corduroy, tweed, ...) are checked before the raw fibers they're
# commonly woven from (cotton, wool, ...) for the same reason. "spandex" and
# "nylon" absorb their regional-label synonyms — "elastane" (EU/UK garment-tag
# name) and "polyamide" (EU/UK name for nylon fiber content) are the same
# fiber under a different label, not a distinct fabric — same convention as
# the existing "lycra" -> "spandex" merge. "jersey" (a knit style, not a raw
# fiber) is ordered last so an explicit fiber mention (e.g. "cotton jersey")
# still wins the more informative tag.
# ---------------------------------------------------------------------------

_FABRIC_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bchanderi\b", re.I), "chanderi"),
    (re.compile(r"\bkhadi\b", re.I), "khadi"),
    (re.compile(r"\bgeorgette\b", re.I), "georgette"),
    (re.compile(r"\bchiffon\b", re.I), "chiffon"),
    (re.compile(r"\bdenim\b", re.I), "denim"),
    (re.compile(r"\bmuslin\b", re.I), "muslin"),
    (re.compile(r"\bvelvet\b", re.I), "velvet"),
    (re.compile(r"\bsatin\b", re.I), "satin"),
    (re.compile(r"\bcrepe\b", re.I), "crepe"),
    (re.compile(r"\bnet\b", re.I), "net"),
    (re.compile(r"\bcorduroy\b", re.I), "corduroy"),
    (re.compile(r"\btweed\b", re.I), "tweed"),
    (re.compile(r"\bleather\b", re.I), "leather"),
    (re.compile(r"\bspandex\b|\blycra\b|\belastane\b", re.I), "spandex"),
    (re.compile(r"\bviscose\b", re.I), "viscose"),
    (re.compile(r"\brayon\b", re.I), "rayon"),
    (re.compile(r"\bmodal\b", re.I), "modal"),
    (re.compile(r"\bcashmere\b", re.I), "cashmere"),
    (re.compile(r"\bwool(?:en|len)?\b", re.I), "wool"),
    (re.compile(r"\blinen\b", re.I), "linen"),
    (re.compile(r"\bsilk\b", re.I), "silk"),
    (re.compile(r"\bcotton\b", re.I), "cotton"),
    (re.compile(r"\bpolyester\b", re.I), "polyester"),
    (re.compile(r"\bnylon\b|\bpolyamide\b", re.I), "nylon"),
    (re.compile(r"\bacrylic\b", re.I), "acrylic"),
    (re.compile(r"\bjersey\b", re.I), "jersey"),
]


def _first_match(text: str, rules: list[tuple[re.Pattern[str], str]]) -> str | None:
    """Return the label of the first rule in *rules* (in order) that matches *text*."""
    for pattern, label in rules:
        if pattern.search(text):
            return label
    return None


def extract_season(text: str | None) -> str | None:
    """Extract a canonical season tag from *text*, or None if unsupported.

    Direct season words ("summer", "winter", "monsoon", "all-season") take
    precedence over weaker fabric/sleeve-length inference (linen/sleeveless ->
    summer; wool/fleece/thermal -> winter).
    """
    if not text:
        return None
    lower = text.lower()
    return _first_match(lower, _SEASON_DIRECT_RULES) or _first_match(lower, _SEASON_FABRIC_RULES)


def extract_occasion(text: str | None) -> str | None:
    """Extract a canonical occasion tag from *text*, or None if unsupported."""
    if not text:
        return None
    return _first_match(text.lower(), _OCCASION_RULES)


def extract_style(text: str | None) -> str | None:
    """Extract a canonical style tag from *text*, or None if unsupported."""
    if not text:
        return None
    return _first_match(text.lower(), _STYLE_RULES)


def extract_fabric(text: str | None) -> str | None:
    """Extract a canonical fabric tag from *text*, or None if unsupported."""
    if not text:
        return None
    return _first_match(text.lower(), _FABRIC_RULES)


def rules_pass(
    prod_name: str | None,
    product_type_name: str | None,
    detail_desc: str | None,
) -> dict[str, str | None]:
    """Run all four rule-based extractors over the combined product text.

    Returns a dict with keys ``season``, ``occasion_tag``, ``style_tag``,
    ``fabric`` — each either a canonical label or None.
    """
    combined = " ".join(t for t in (prod_name, product_type_name, detail_desc) if t)
    return {
        "season": extract_season(combined),
        "occasion_tag": extract_occasion(combined),
        "style_tag": extract_style(combined),
        "fabric": extract_fabric(combined),
    }


def merge_enrichment(
    rules: dict[str, str | None],
    llm: dict[str, str | None] | None,
) -> tuple[dict[str, str | None], dict[str, str]]:
    """Merge the (higher-trust) rules pass with the LLM fallback pass.

    Rules values always win. For any facet where the rules pass is None, the
    LLM's value is used ONLY if it is a member of that facet's fixed vocab
    (:data:`FACET_VOCAB`) — this is the hallucination guard: a model reply of
    "boho-chic" or "spring" (not in the vocab) is discarded to None rather than
    kept, since it's either an invented category or a paraphrase we can't trust
    to be exact.

    Returns ``(merged, source)`` where ``source[facet]`` is one of
    ``"rule"``, ``"llm"``, or ``"none"`` — used for the enrichment-coverage report.
    """
    merged: dict[str, str | None] = {}
    source: dict[str, str] = {}
    llm = llm or {}
    for facet, vocab in FACET_VOCAB.items():
        rule_value = rules.get(facet)
        if rule_value is not None:
            merged[facet] = rule_value
            source[facet] = "rule"
            continue
        llm_value = llm.get(facet)
        if llm_value is not None and llm_value in vocab:
            merged[facet] = llm_value
            source[facet] = "llm"
        else:
            merged[facet] = None
            source[facet] = "none"
    return merged, source


def append_enrichment_to_search_text(search_text: str, enrichment: dict[str, str | None]) -> str:
    """Append non-null enrichment tags to *search_text* so BM25/dense can match on them.

    Appends rather than replaces — existing search_text content (prod_name,
    product_type, colour, department, detail_desc — see
    ``cleaning.py::recompute_derived_columns``) is left untouched.
    """
    tags = [
        enrichment.get(facet)
        for facet in ("season", "occasion_tag", "style_tag", "fabric")
        if enrichment.get(facet)
    ]
    if not tags:
        return search_text
    base = (search_text or "").rstrip()
    suffix = ". ".join(tags)
    return f"{base}. {suffix}." if base else f"{suffix}."
