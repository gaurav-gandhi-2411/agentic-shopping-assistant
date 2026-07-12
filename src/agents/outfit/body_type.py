"""P3: body-type-aware styling registry for Indian occasion/wedding wear.

Data source: scratchpad research doc "P3 Research: Body-Type-Aware Styling
Ruleset for Indian Wedding/Occasion Wear" (2026-07-09), sections 5-7. This
module is intentionally a pure data + pure-function module (stdlib only, zero
project imports) so it can be imported by intent_parser.py-adjacent code and
tested in isolation without loading the catalogue/retriever/LLM layers —
mirrors the src/agents/outfit/occasions.py pattern.

Everything here is OPT-IN BIAS, never a filter (§7.7): callers add/subtract a
small score delta or append query tokens; the candidate pool itself is always
identical with or without a known body type. See body_type_score_delta's
docstring for the "never filter" contract.

Scope note (honest): only the 5 base shapes + 3 modifiers explicitly in scope
for this wave are encoded (pear, apple, hourglass, rectangle,
inverted_triangle; petite, tall, plus_size). The research doc's §5 menswear
rows (men_slim/men_broad/men_short/men_tall) are NOT encoded here — out of
scope for this task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Registry data model
# ---------------------------------------------------------------------------

# Garment classes used as the per-shape rule keys. "neckline" is a cross-
# garment attribute overlay (a blouse/choli neckline can appear on a saree OR
# an anarkali/kurta) — see garment_class_for_item's docstring for why item
# classification does NOT gate on it exclusively.
GARMENT_CLASSES: tuple[str, ...] = ("saree", "lehenga", "anarkali_kurta", "neckline")

BASE_SHAPE_SLUGS: tuple[str, ...] = (
    "pear", "apple", "hourglass", "rectangle", "inverted_triangle",
)
MODIFIER_SLUGS: tuple[str, ...] = ("petite", "tall", "plus_size")


@dataclass(frozen=True)
class GarmentRule:
    """recommend/deprioritize keyword lists + a positive WHY for one garment class."""

    recommend: tuple[str, ...]
    deprioritize: tuple[str, ...]
    why: str


@dataclass(frozen=True)
class BodyTypeProfile:
    """A body shape or modifier: per-garment-class rules + a query-augmentation hint."""

    slug: str
    garments: dict[str, GarmentRule]
    query_tokens: str  # short retrieval-query augmentation string (§5 encoding note)


# ---------------------------------------------------------------------------
# Base shapes (§5 table rows, transcribed verbatim from the research doc)
# ---------------------------------------------------------------------------

# Vocabulary note (post-launch defect fix, 2026-07-10): the recommend/
# deprioritize keyword lists below were reworked from the research doc's
# verbatim multi-word phrases ("A-line anarkali", "body-hugging straight
# kurta", ...) to catalogue-realistic single tokens / short bigrams that
# actually occur in this catalogue's prod_name/detail_desc text. The original
# phrasing was semantically correct but essentially never matched real
# product text — body_type_score_delta was measured to be a no-op against
# the live catalogue (see the hit-rate verification in this module's PR).
# Genuinely catalogue-absent style words from the research doc (e.g.
# "mermaid", "fishtail", "bodycon", "peplum", "empire waist", "high-waisted")
# are kept where the research is unambiguous, even though their real hit rate
# is near-zero — the WHY prose and query_tokens stay true to the research;
# only the SCORING keyword lists were tightened to real vocabulary. Words
# that are near-universal boilerplate across ALL ethnic wear regardless of
# silhouette (e.g. "printed", "solid", "silk", "cotton", "embroidered",
# "stitched"/"unstitched", "waistband", "border") are deliberately EXCLUDED
# even where semantically plausible — they would fire on the vast majority of
# rows and stop being a body-type signal at all (precision guard).
BASE_SHAPES: dict[str, BodyTypeProfile] = {
    "pear": BodyTypeProfile(
        slug="pear",
        query_tokens="a-line flared embellished embroidered yoke palazzo",
        garments={
            "saree": GarmentRule(
                recommend=("georgette", "chiffon", "crepe", "embellished", "embroidered yoke"),
                deprioritize=("straight", "banarasi"),
                why="fluid fabric glides over the hip while a statement blouse draws the eye upward",
            ),
            "lehenga": GarmentRule(
                recommend=("a-line", "flared", "flare", "embellished"),
                deprioritize=("straight", "mermaid", "fishtail"),
                why="flare falls from the waist so the skirt line leads, and top detail "
                "balances the look upward",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("a-line", "flared", "flare", "embroidered yoke", "palazzo"),
                deprioritize=("straight kurta", "straight fit", "bodycon"),
                why="fitted bodice with flare from the waist celebrates the waist and moves "
                "gracefully over the hip",
            ),
            "neckline": GarmentRule(
                recommend=("boat", "embellished"),
                deprioritize=(),
                why="width and detail up top create beautiful shoulder-hip balance",
            ),
        },
    ),
    "apple": BodyTypeProfile(
        slug="apple",
        query_tokens="empire straight cut v-neck side slit",
        garments={
            "saree": GarmentRule(
                recommend=("chiffon", "georgette", "crepe", "v-neck", "side slit"),
                deprioritize=("mandarin collar",),
                why="one long vertical fall keeps the eye moving; nothing anchors attention "
                "at the waistline",
            ),
            "lehenga": GarmentRule(
                recommend=("a-line", "flared", "flare", "v-neck"),
                deprioritize=("crop", "cropped"),
                why="a high waistline and flowing skirt create one graceful line from under the bust",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("empire", "straight kurta", "straight cut", "v-neck", "side slit"),
                deprioritize=("mandarin collar", "high neck"),
                why="flare starting below the bust flows freely; V-necklines lengthen the frame",
            ),
            "neckline": GarmentRule(
                recommend=("v-neck", "scoop neck", "deep v"),
                deprioritize=("mandarin collar", "high neck"),
                why="an open neckline lengthens the neck-to-waist line",
            ),
        },
    ),
    "hourglass": BodyTypeProfile(
        slug="hourglass",
        query_tokens="banarasi brocade fitted belted",
        garments={
            "saree": GarmentRule(
                recommend=("banarasi", "brocade", "zari"),
                deprioritize=("oversized", "loose"),
                why="the classic drape naturally showcases a defined waist",
            ),
            "lehenga": GarmentRule(
                recommend=("a-line", "flared", "flare", "fitted"),
                deprioritize=("boxy", "longline"),
                why="balanced curves carry nearly every cut; the waistline is the star",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("fitted", "belted", "wrap"),
                deprioritize=("loose", "oversized", "boxy"),
                why="a clear waist seam honors the natural waistline",
            ),
            "neckline": GarmentRule(
                recommend=("v-neck", "sweetheart neck"),
                deprioritize=(),
                why="proportions are balanced, so neckline choice is pure style preference",
            ),
        },
    ),
    "rectangle": BodyTypeProfile(
        slug="rectangle",
        query_tokens="flared belted organza layered sweetheart",
        garments={
            "saree": GarmentRule(
                recommend=("belt", "belted", "organza", "halter"),
                deprioritize=("straight",),
                why="a belt and bold border create a lovely waist cue and curve",
            ),
            "lehenga": GarmentRule(
                recommend=("flared", "flare", "sweetheart neck", "peplum"),
                deprioritize=("straight",),
                why="structured volume below and a curved neckline above build an "
                "hourglass impression",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("layered", "tiered", "angrakha", "belted"),
                deprioritize=("straight kurta", "straight fit"),
                why="layers and knee-flare add dimension and movement to a sleek frame",
            ),
            "neckline": GarmentRule(
                recommend=("sweetheart neck", "plunge", "embellished"),
                deprioritize=(),
                why="curve-shaped necklines suggest curves",
            ),
        },
    ),
    "inverted_triangle": BodyTypeProfile(
        slug="inverted_triangle",
        query_tokens="flared gathered pleated v-neck",
        garments={
            "saree": GarmentRule(
                recommend=("pleated", "gathered", "v-neck"),
                deprioritize=("boat", "puff sleeve", "yoke"),
                why="fullness at the hip and a clean shoulder line create elegant balance",
            ),
            "lehenga": GarmentRule(
                recommend=("flared", "flare", "gathered", "pleated"),
                deprioritize=("boat", "puff sleeve", "yoke"),
                why="skirt volume grounds the silhouette; quiet top half lets strong "
                "shoulders stay graceful",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("flared", "flare", "gathered"),
                deprioritize=("yoke", "boat", "puff sleeve"),
                why="volume below the waist balances a strong shoulder line",
            ),
            "neckline": GarmentRule(
                recommend=("v-neck", "scoop neck", "halter"),
                deprioritize=("boat", "square neck"),
                why="narrow vertical necklines refine shoulder width",
            ),
        },
    ),
}

# ---------------------------------------------------------------------------
# Modifiers (§5 table rows) — orthogonal to base shape; no "neckline" row
# ---------------------------------------------------------------------------

MODIFIERS: dict[str, BodyTypeProfile] = {
    "petite": BodyTypeProfile(
        slug="petite",
        query_tokens="ankle length short kurti v-neck",
        garments={
            "saree": GarmentRule(
                recommend=("georgette", "chiffon"),
                deprioritize=("banarasi", "brocade"),
                why="an unbroken vertical line and fine-scale detail read tall and elegant",
            ),
            "lehenga": GarmentRule(
                recommend=("a-line", "flared"),
                deprioritize=("tiered", "layered"),
                why="a high waist and contained flare keep proportions long",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("ankle length", "short kurti", "v-neck"),
                deprioritize=("floor length", "wide leg"),
                why="vertical lines and a visible leg line create height",
            ),
        },
    ),
    "tall": BodyTypeProfile(
        slug="tall",
        query_tokens="floor length banarasi layered tiered",
        garments={
            "saree": GarmentRule(
                recommend=("banarasi", "brocade", "zari"),
                deprioritize=(),
                why="height carries rich fabric and bold scale beautifully",
            ),
            "lehenga": GarmentRule(
                recommend=("tiered", "layered", "embellished"),
                deprioritize=(),
                why="drama and layering are a tall frame's superpower",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("floor length", "layered"),
                deprioritize=(),
                why="horizontal breaks and layers add pleasing balance to height",
            ),
        },
    ),
    "plus_size": BodyTypeProfile(
        slug="plus_size",
        query_tokens="a-line flared v-neck georgette",
        garments={
            "saree": GarmentRule(
                recommend=("georgette", "chiffon", "crepe", "v-neck", "sweetheart neck"),
                deprioritize=("banarasi",),
                why="fluid fabrics drape gracefully over curves; a well-fitted blouse gives "
                "the look architecture",
            ),
            "lehenga": GarmentRule(
                recommend=("a-line", "flared", "v-neck"),
                deprioritize=("straight",),
                why="a high waist marks the narrowest point and vertical lines flow into "
                "one graceful sweep",
            ),
            "anarkali_kurta": GarmentRule(
                recommend=("a-line", "flared", "panelled", "georgette", "crepe"),
                deprioritize=("fitted", "straight"),
                why="vertical panels and skimming drape show shape beautifully without cling",
            ),
        },
    ),
}

# ---------------------------------------------------------------------------
# Parsing vocabulary (§2) — aliases → canonical slug.
# ---------------------------------------------------------------------------
# NOTE: intent_parser.py deliberately does NOT import this dict (it keeps a
# zero-project-import invariant — see its module docstring) and instead
# carries its own copy of the same phrase → slug vocabulary. Keep both in
# sync when this list changes (same pattern already used for occasion slugs
# between occasions.py and intent_parser.py's _OCCASION_MAP).
SYNONYMS: dict[str, str] = {
    # pear
    "pear shaped": "pear",
    "pear-shaped": "pear",
    "pear shape": "pear",
    "pear": "pear",
    "triangle body": "pear",
    "curvy hips": "pear",
    "hip-forward": "pear",
    "hip forward": "pear",
    # apple
    "apple shaped": "apple",
    "apple-shaped": "apple",
    "apple shape": "apple",
    "apple": "apple",
    "round shape": "apple",
    "rounder middle": "apple",
    "midsection-forward": "apple",
    # hourglass
    "hourglass shaped": "hourglass",
    "hourglass-shaped": "hourglass",
    "hourglass shape": "hourglass",
    "hourglass": "hourglass",
    "balanced curves": "hourglass",
    # rectangle
    "rectangle shaped": "rectangle",
    "rectangle-shaped": "rectangle",
    "rectangle shape": "rectangle",
    "rectangle": "rectangle",
    "straight frame": "rectangle",
    "athletic frame": "rectangle",
    "athletic build": "rectangle",
    # inverted triangle
    "inverted triangle": "inverted_triangle",
    "inverted-triangle": "inverted_triangle",
    "broad-shouldered": "inverted_triangle",
    "broad shouldered": "inverted_triangle",
    # modifiers
    "plus size": "plus_size",
    "plus-size": "plus_size",
    "curvy": "plus_size",
    "petite": "petite",
    "tall": "tall",
}

_SYNONYMS_SORTED: list[tuple[str, str]] = sorted(
    SYNONYMS.items(), key=lambda kv: len(kv[0]), reverse=True
)

_PLUS_SIZE_NAME_RE = re.compile(r"\bplus[\s-]?size\b", re.IGNORECASE)


def demote_size_mismatched_items(items: list[dict], query: str) -> list[dict]:
    """Body SHAPE is not a SIZE (sweep 2026-07-10, relevance-adjacent): a
    "pear shaped" request surfaced an explicitly "Plus Size"-branded kurta in
    the top results — reading as a size assumption the user never made.

    When the query names a base shape (pear/apple/hourglass/rectangle/
    inverted_triangle) and does NOT name a plus-size modifier, stable-demote
    items whose product name markets "plus size" to the end of the list.
    Never a filter: the items stay available (they are valid products), they
    just cannot be the headline recommendation on shape grounds alone. A query
    that DOES say plus-size/curvy is returned untouched.
    """
    q = (query or "").lower()
    matched = {slug for phrase, slug in _SYNONYMS_SORTED if phrase in q}
    has_base_shape = any(s in BASE_SHAPES for s in matched)
    if not has_base_shape or "plus_size" in matched:
        return items
    keep = [it for it in items if not _PLUS_SIZE_NAME_RE.search(it.get("prod_name") or "")]
    demoted = [it for it in items if _PLUS_SIZE_NAME_RE.search(it.get("prod_name") or "")]
    return keep + demoted


def parse_body_type(text: str) -> tuple[str | None, list[str]]:
    """Extract a base body_type + modifier list from free text.

    Longest-phrase-first, non-overlapping-span matching (same algorithm shape
    as intent_parser._extract_colour/_extract_occasion) so overlapping aliases
    ("curvy hips" vs standalone "curvy") never double-fire — the LONGER phrase
    claims its span first and shorter phrases whose span overlaps it are
    skipped. Multiple non-overlapping matches are allowed (unlike colour/
    occasion, which return a single winner) since a user may state a base
    shape AND a modifier in the same message ("petite pear").

    Returns:
        (base_type, modifiers) — base_type is one of BASE_SHAPE_SLUGS or None;
        modifiers is a de-duplicated list (input order) of MODIFIER_SLUGS.
    """
    text_lower = text.lower()
    consumed: list[tuple[int, int]] = []
    ordered: list[tuple[int, str]] = []

    for phrase, slug in _SYNONYMS_SORTED:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        for m in re.finditer(pattern, text_lower):
            span = (m.start(), m.end())
            if any(not (span[1] <= c[0] or span[0] >= c[1]) for c in consumed):
                continue
            consumed.append(span)
            ordered.append((span[0], slug))

    ordered.sort(key=lambda t: t[0])
    slugs_in_order = [slug for _, slug in ordered]

    base = next((s for s in slugs_in_order if s in BASE_SHAPES), None)
    modifiers: list[str] = []
    for s in slugs_in_order:
        if s in MODIFIERS and s not in modifiers:
            modifiers.append(s)
    return base, modifiers


# ---------------------------------------------------------------------------
# Framing rules (§6) — ban-list + allow-list-derived positive templates.
# ---------------------------------------------------------------------------

BANNED_FRAMING_WORDS: frozenset[str] = frozenset({
    "hide", "hides", "hiding",
    "conceal", "conceals", "concealing",
    "camouflage", "camouflages", "camouflaging",
    "disguise", "disguises", "disguising",
    "mask", "masks", "masking",
    "flaw", "flaws",
    "problem area", "problem areas",
    "trouble spot", "trouble spots",
    "figure fault", "figure faults",
    "fix", "fixes", "fixing",
    "correct", "corrects", "correcting",
    "flabby",
    "bulky",
    "bulge", "bulges",
    "minimise", "minimize", "minimises", "minimizes",
    "slimming",
    "unflattering",
    "not for your body type",
    "you should avoid",
    "makes you look thinner",
    "makes you look slimmer",
    "suits your imperfections",
})

_BANNED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", re.IGNORECASE)
    for phrase in BANNED_FRAMING_WORDS
]


def contains_banned_framing(text: str) -> bool:
    """Case-insensitive, word-boundary scan for any BANNED_FRAMING_WORDS phrase.

    Used as the hard guardrail after LLM rationale generation (see
    src/agents/outfit/rationale.py::generate_rationales) — any hit discards
    the LLM text in favour of a deterministic body-positive template.
    """
    return any(p.search(text) for p in _BANNED_PATTERNS)


# One celebratory sentence per BASE shape, built entirely from §6's allow-list
# vocabulary (balances/flows/celebrates/...). Used as the template-fallback
# rationale for body-type turns (never references a body part directly —
# always the garment's line/movement, per §6 rule 4).
POSITIVE_TEMPLATES: dict[str, str] = {
    "pear": (
        "An A-line lehenga or anarkali balances a pear silhouette beautifully — "
        "the flare falls from the waist and moves with you."
    ),
    "apple": (
        "An empire-waist anarkali or seedha-pallu saree flows from just under the bust — "
        "one graceful, uninterrupted line."
    ),
    "hourglass": (
        "A fitted-waist saree or lehenga lets your hourglass silhouette shine — "
        "the waist stays the star of the look."
    ),
    "rectangle": (
        "A peplum or belted choli adds lovely movement and dimension to a rectangle frame."
    ),
    "inverted_triangle": (
        "A flared lehenga or sharara grounds the look beautifully, balancing a broader "
        "shoulder line with graceful ease."
    ),
}


def body_type_ack_message(body_type: str | None, modifiers: list[str] | None = None) -> str:
    """Deterministic acknowledgement for a bare body-type STATEMENT with nothing
    else to act on (no occasion, no garment, no product query, no prior look) —
    e.g. "I have an inverted triangle silhouette" said as the first and only
    message in a session (this is exactly what the photo body-shape confirm
    button sends — see frontend/lib/poseShape.ts's bodyShapeMessage()).

    Never an LLM call — mirrors body_type_clarify_message's determinism.
    graph.py's router short-circuits to this template BEFORE the
    conversational/product-search branching so the turn always completes with
    a real reply instead of silently degrading into an irrelevant product
    search (root cause: a shape statement alone is not a product query, so the
    deterministic router correctly picked action="respond", but a downstream
    guard in route_decision — designed to catch the LLM router hallucinating
    "respond" — force-converted it to "search" on any fresh-session turn with
    no retrieved_items yet, regardless of why "respond" was chosen).
    """
    labels = [m.replace("_", " ") for m in (modifiers or [])]
    if body_type:
        labels.append(body_type.replace("_", " "))
    shape_desc = " ".join(labels) if labels else "shape"
    why = POSITIVE_TEMPLATES.get(body_type or "", "")
    ack = f"Got it — I'll keep your {shape_desc} silhouette in mind!"
    if why:
        ack += f" {why}"
    ack += " What are you shopping for — a sangeet look, office wear, or something else?"
    return ack


def body_type_clarify_message() -> str:
    """Warm, judgment-free, opt-in prompt listing shape options.

    Deterministic template (never an LLM call) — used by graph.py's router
    short-circuit for a body-type QUESTION with no stated body type (§6
    interaction rules: never gate product results on this; always framed as
    optional).
    """
    return (
        "Happy to tailor styling to your shape — totally optional, and I'll style "
        "beautifully either way. A few shapes people mention:\n\n"
        "- Hip-forward / pear (fuller through the hip)\n"
        "- Midsection-forward / apple (curves centred at the waist)\n"
        "- Balanced curves / hourglass (bust and hip close in width, defined waist)\n"
        "- Straight or athletic frame / rectangle (little waist definition)\n"
        "- Broad-shouldered / inverted triangle (shoulders wider than the hip)\n\n"
        "You can also mention petite, tall, or plus-size, alone or combined "
        "(e.g. \"petite pear\"). Just say the word whenever you'd like — or skip it "
        "entirely and I'll style from your occasion and budget instead."
    )


# ---------------------------------------------------------------------------
# Retrieval-query augmentation (mirrors slots._occasion_register_tokens)
# ---------------------------------------------------------------------------


def query_tokens(body_type: str | None, modifiers: list[str] | None = None) -> str:
    """Return extra retrieval-query tokens for a known body type + modifiers.

    Mirrors slots.py's _occasion_register_tokens: a short, curated string
    appended to anchor/slot search queries so retrieval favours garments this
    body type's rules recommend. Empty string when nothing is known (no-op
    for callers that always append via f"{query} {query_tokens(...)}".strip()).
    """
    parts: list[str] = []
    if body_type and body_type in BASE_SHAPES:
        parts.append(BASE_SHAPES[body_type].query_tokens)
    for mod in modifiers or []:
        profile = MODIFIERS.get(mod)
        if profile:
            parts.append(profile.query_tokens)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Retrieval score bias (mirrors slots.fabric_score_delta)
# ---------------------------------------------------------------------------

_SAREE_MARKERS: tuple[str, ...] = ("saree", "sari")
_LEHENGA_MARKERS: tuple[str, ...] = ("lehenga", "lehnga", "ghagra")
_ANARKALI_KURTA_MARKERS: tuple[str, ...] = (
    "anarkali", "kurta", "kurti", "sharara", "salwar", "palazzo", "suit", "tunic",
)


def garment_class_for_item(product_type: str, prod_name: str) -> str | None:
    """Classify a catalogue item into one of the §5 body-type garment classes.

    Deliberately self-contained (no import of slots.classify_anchor) so this
    module stays a pure-data/pure-function module with zero project imports —
    the same isolation invariant occasions.py already relies on. Returns None
    for garments the research doc doesn't cover (western wear, footwear,
    outerwear, accessories) — body_type_score_delta returns 0.0 for those.
    """
    combined = f"{product_type} {prod_name}".lower()
    if any(m in combined for m in _SAREE_MARKERS):
        return "saree"
    if any(m in combined for m in _LEHENGA_MARKERS):
        return "lehenga"
    if any(m in combined for m in _ANARKALI_KURTA_MARKERS):
        return "anarkali_kurta"
    return None


def _profile_keywords(
    profile: BodyTypeProfile, garment_class: str | None
) -> tuple[set[str], set[str]]:
    """Return (recommend, deprioritize) keyword sets for one profile.

    Always includes the "neckline" overlay (a blouse/choli neckline can be
    mentioned in a saree/anarkali/kurta's own description text) in addition to
    the item's primary garment_class rule, when both exist.
    """
    recommend: set[str] = set()
    deprioritize: set[str] = set()
    classes = {"neckline"}
    if garment_class:
        classes.add(garment_class)
    for cls in classes:
        rule = profile.garments.get(cls)
        if rule:
            recommend.update(kw.lower() for kw in rule.recommend)
            deprioritize.update(kw.lower() for kw in rule.deprioritize)
    return recommend, deprioritize


def body_type_score_delta(
    item: dict,
    body_type: str | None,
    modifiers: list[str] | None = None,
) -> float:
    """Return a score adjustment based on body-type recommend/deprioritize keywords.

    BIAS ONLY — never a filter (§7.7): this function only ever returns a small
    additive delta on an already-retrieved candidate; it never removes a
    candidate from the pool. Callers (composer._score_candidates) must add
    this delta to the base score exactly like fabric_score_delta, and the
    candidate list passed in/out must be identical regardless of body_type.

    Recommend keyword present anywhere in the item's prod_name/detail_desc →
    +0.1. Else a deprioritize keyword present → -0.1. Else 0.0. Mirrors
    fabric_score_delta's exclusive if/elif magnitude and text-scan shape.

    modifiers compose with body_type via UNION of recommend keywords and UNION
    of deprioritize keywords (§5 encoding note) — a "petite pear" candidate
    gets +0.1 if it matches EITHER the pear OR the petite recommend list.
    """
    modifiers = modifiers or []
    if not body_type and not modifiers:
        return 0.0

    text = ((item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")).lower()
    garment_class = garment_class_for_item(
        item.get("product_type") or "", item.get("prod_name") or ""
    )

    recommend: set[str] = set()
    deprioritize: set[str] = set()
    if body_type and body_type in BASE_SHAPES:
        r, d = _profile_keywords(BASE_SHAPES[body_type], garment_class)
        recommend |= r
        deprioritize |= d
    for mod in modifiers:
        profile = MODIFIERS.get(mod)
        if profile:
            r, d = _profile_keywords(profile, garment_class)
            recommend |= r
            deprioritize |= d

    if any(kw in text for kw in recommend):
        return 0.1
    if any(kw in text for kw in deprioritize):
        return -0.1
    return 0.0
