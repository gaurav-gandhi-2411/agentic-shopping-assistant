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


# ---------------------------------------------------------------------------
# Kids/juniors item exclusion
# ---------------------------------------------------------------------------
# S5 fix (promoted from src/agents/outfit/slots.py, 2026-07-12): juniors/kids
# garments mislabeled as adult inventory. src.catalogue.adapter.derive_item_gender
# previously treated "girl"/"girls"/"boy"/"boys" as ADULT gender keywords, so
# juniors/girls/boys/kids SKUs were carrying gender="women"/"men" alongside
# genuinely adult items (verified: "M&H Juniors Girls Blue Straight Knee Length
# Denim Skirts" and "Juniors by Lifestyle Kids-Girls White Pure Cotton Print
# Top" both carried gender="women" in data/processed/unified/catalogue.parquet)
# — so gender_allowed()/a gender filter alone let them through into ADULT
# results. Live-proven: an office look's bottom slot filled with the Juniors
# denim-skirt item above, and "red lehenga bridal"/"gold jewellery to go with
# red lehenga" (non-occasion-keyword searches) surfaced girls' lehengas ranked
# above adult bridal options. Deliberately narrow (four markers, not a broader
# age/size heuristic) to avoid rejecting real adult inventory whose name
# happens to share a word.
#
# Promoted here (rather than left in src/agents/outfit/slots.py) so the
# retrieval layer (src/retrieval/hybrid_search.py) and the plain-search node
# (src/agents/graph.py) can both apply it as an UNCONDITIONAL hard exclusion,
# mirroring is_fabric_bolt_text above, without importing from the agents layer.
_KIDS_MARKER_RE = re.compile(r"\b(junior|juniors|girl|girls|boy|boys|kid|kids)\b", re.IGNORECASE)


def is_kids_item(prod_name: str | None) -> bool:
    """Return True if `prod_name` carries a juniors/girls/boys/kids marker.

    Applied as a hard exclusion at retrieval time (hybrid_search.py, graph.py)
    AND as an additional per-slot gate in the outfit composer — see module
    docstring above _KIDS_MARKER_RE for why the gender column alone isn't enough.
    """
    return bool(_KIDS_MARKER_RE.search(prod_name or ""))


# ---------------------------------------------------------------------------
# Occasion-merchandise leak exclusion
# ---------------------------------------------------------------------------
# Live-proven bug (2026-07-23): "what should I wear for raksha bandhan" (an
# apparel-intent occasion query, no garment noun) returned 3 of 5 items as
# literal Rakhi threads/gift objects (product_type_name="Rakhi", e.g. "Ram
# Mandir Blessings Rakhi") because BM25/dense retrieval on "raksha bandhan"
# text naturally ranks the 635 catalogue rows literally named "Rakhi" ahead of
# apparel — there was no exclusion mechanism for occasion-keyword text also
# matching non-apparel occasion merchandise (unlike kids items/fabric bolts
# above).
#
# Grounded in a full catalogue audit (data/processed/unified/catalogue.parquet,
# rows whose prod_name/detail_desc contain each occasion's keyword):
#   raksha_bandhan (1096 matches): Rakhi=635, Silver Rakhi=19, Rakhi Hamper=4,
#     Rakhi Gift Hamper=5, Gift Hamper=6 -- 669 non-apparel merchandise rows.
#   diwali (1757 matches): Gift Hamper=6, Rakhi/Rakhi Gift Hamper=11 (Diwali
#     hampers are catalogued generically), Idols=1, Others=19 (all idols/
#     showpieces/tealight-holders on inspection, e.g. "Gift of Grace Lord
#     Ganesha Idol", "Festive Decorative Tealight Holder").
#   navratri (545 matches): Idols=1.
#   karva_chauth (24 matches): Gift Hamper=3.
#   eid (658 matches): Rakhi=1.
#
# Deliberately narrow to these product_type_name values (not a broader
# "anything non-apparel" heuristic):
#   - "Potli"/"Potlis" bags DO appear in the diwali-matching set (11 rows) but
#     are legitimately styled as accessories in real looks (see the composer's
#     accessory slot) -- kept, not excluded.
#   - Jewellery (Earrings/Necklace/Bangles/Rings/jhumka/...) is apparel-
#     adjacent and must NEVER be excluded here -- it dominates every
#     occasion's keyword-matching set by volume and is exactly what the
#     2026-07-23 multi-family accessory-retrieval fix (commit 1717265) exists
#     to surface correctly for bridal/festive looks.
# 2026-07-30 addition: bare "hampers"/"hamper" product_type_name values were
# missing -- only compound phrases ("gift hamper", "rakhi hamper") were
# covered. Confirmed via catalogue audit: product_type_name=="Hampers" is
# exactly 3 rows, all "Mortantra X Zivame Bridal Hamper Box A/B/C" -- genuine
# gift hampers, zero false-positive risk. This surfaced via "bridal look for
# women" returning a literal hamper box in top-5 (see intent_parser's
# "bridal" occasion-map addition, the compounding root cause for that leak).
# 2026-07-30 SAME-DAY follow-up ("gift card"): live-proven via the new
# "anniversary" occasion mapping (intent_parser.py) -- "anniversary party
# outfit for women" ranked "Anniversary Day E-Gift Card" #1 of 5 results.
# Confirmed via catalogue audit (\bgift.card\b over prod_name): 21 rows, 100%
# genuine gift cards (e.g. "Rakhi E-Gift Card...", "Father's Day E-Gift
# Card", "Happy Mother's Day : Gift Card"), zero false-positive risk. Facet
# values found: "Gift-Card", "Gift Cards", "Gift Card".
_OCCASION_MERCHANDISE_TYPES: frozenset[str] = frozenset({
    "rakhi", "rakhi hamper", "rakhi gift hamper", "silver rakhi",
    "gift hamper", "idols", "hampers", "hamper",
    "gift-card", "gift cards", "gift card",
})


def is_occasion_merchandise_type(product_type_name: str | None) -> bool:
    """Return True if `product_type_name` is occasion merchandise (Rakhi
    threads, gift hampers, religious idols/showpieces), not a wearable
    garment or apparel-adjacent accessory.

    Callers must gate this on apparel-intent occasion context (see
    src.agents.graph._apply_occasion_merchandise_gate) -- an explicit request
    FOR the merchandise itself ("rakhi for my brother", "gift for raksha
    bandhan") must still surface it; this predicate only identifies the
    product-type class, it does not decide when to apply it.

    See is_occasion_merchandise_name below for the NAME-level complement --
    catches merchandise a store tagged with a GENERIC catalog bucket
    ("Fashion", "Others", "Article", ...) instead of a dedicated
    "Rakhi"/"Gift Hamper"/"Idols" type this function alone would miss.
    """
    return (product_type_name or "").strip().lower() in _OCCASION_MERCHANDISE_TYPES


# Generic/non-apparel product_type_name buckets — grounded in the real
# catalogue (data/processed/unified/catalogue.parquet). These types carry no
# apparel-vs-merchandise signal of their own (unlike "kurta"/"Bracelets"/
# "Tie Set", which ARE genuine apparel/accessory types even when their name
# also happens to mention rakhi/gift/idol), so a merchandise-suggestive NAME
# under one of these types is trustworthy signal -- the same name words under
# a real apparel/accessory type are NOT excluded, because the type IS the
# product there (see is_occasion_merchandise_name's docstring for the exact
# live-proven example this protects).
_GENERIC_PRODUCT_TYPES: frozenset[str] = frozenset({
    "fashion", "others", "article", "all products", "all product",
    "clothing accessories", "giftables", "",
})

# Name/description-level occasion-merchandise markers, scoped to the generic
# types above. Live-proven residual leak (2026-07-23 live-proof, revision
# asa-stylist-api-00084-7t4): "White And Pink Beautiful Floral Designer
# Bhaiya Bhabhi Rakhi Set" (store=ishhaara, product_type_name="Fashion")
# ranked #1 of only 2 results for "what should I wear for raksha bandhan" --
# is_occasion_merchandise_type's type-only exclusion missed it because this
# store tagged the SKU "Fashion" instead of "Rakhi".
#
# Grounded in a full audit of every raksha_bandhan/diwali/navratri/
# karva_chauth/eid-keyword-matching row currently under a generic type:
#   raksha_bandhan: 57 Fashion-typed rows -- 56 contain the word "rakhi"
#     (E-Gift Cards, kids rakhis, "Bhaiya Bhabhi" rakhi sets/combos/hampers,
#     "Evil Eye Rakhi Gift Combo With Mug"), and the 1 remaining ("Raksha
#     Bandhan Gift For Brother") carries the bare occasion phrase instead of
#     the word "rakhi" itself.
#   diwali/navratri: 21 Others/Article-typed rows, ALL idols/showpieces/
#     tealight holders (e.g. "Gift of Grace Lord Ganesha Idol", "Festive
#     Decorative Tealight Holder", "Voylla ... Kamdhenu Sacred Cow Idol").
#   karva_chauth/eid: 0 generic-typed rows carry any of these markers.
# "Bhaiya bhabhi" without the literal word "rakhi" does not occur anywhere in
# the catalogue (checked) -- no separate bhaiya-bhabhi pattern is needed.
#
# Genuine apparel is explicitly NOT caught by this: 19 "kurta"-typed and 2
# "nightwear"-typed rows also say "Rakhi Gift Box for Brother" in their name
# (a real kurta bundled with a rakhi) — "kurta"/"nightwear" are real apparel
# types, not in _GENERIC_PRODUCT_TYPES, so is_occasion_merchandise_name
# returns False for them regardless of the name match. Likewise 20
# "Bracelets"-typed rows literally named "... Rakhi Bracelet" are genuine
# jewellery (a real accessory type) and stay included, consistent with
# is_occasion_merchandise_type's jewellery carve-out above.
#
# 2026-07-24 CONCEPT-BROADENING addition ("favour"/"favor"): live-proven bug
# -- "bright haldi look for women" surfaced "Ellaichi Brooch" (store=ishhaara,
# product_type_name="Fashion"), whose OWN detail_desc frames it as a "Haldi &
# Mehendi Favours" return-gift for wedding guests, not a wearable item (see
# eval/fixtures/strict_gold_labels.yaml occ_adv_002). A full catalogue audit
# of every occasion this project recognises (haldi/mehendi/sangeet/diwali/
# navratri/karva_chauth/raksha_bandhan/eid/wedding_guest/engagement/
# reception/festive_puja) for decorative/gift/favour/party-supply vocabulary
# found genuine, non-redundant catalogue support for exactly ONE new concept
# term family -- "favour(s)"/"favor(s)" -- not a longer flat list:
#   ishhaara's "Haldi & Mehendi Favours" collection: 40 rows (36 Fashion-typed
#     + 4 bag-typed), ALL genuinely non-wearable guest tokens/return-gifts
#     (brooches, "malas", bracelets, keychains, a mouth-freshener-and-scrunchy
#     combo, a kumkum-stick applicator) sharing one boilerplate description
#     ("Are you looking for the perfect way to thank your guests ... Haldi &
#     Mehendi Favours ..."). Checked catalogue-wide (all product_type_name
#     buckets, not just generic ones): every OTHER "favour"/"favor" hit is a
#     real dual-use accessory explicitly marketed as wearable (tjori's silk
#     Potli bags: "pair with a lehenga for a regal coordinated look ... use as
#     a bridal favour"; sukkhi's clip-on "Party Favor" earrings: genuine
#     wearable costume jewellery) -- those sit under real accessory types
#     (Potlis/Earring), not a generic bucket, so the AND-gate below already
#     protects them without any special-case needed.
#   candidate terms researched but found ZERO genuine catalogue support (or
#   ONLY false-positive matches) and deliberately NOT added: "rangoli" (15
#     matches, the 1 generic-typed hit is "THE KILIM RANGOLI POCKET SQUARE" --
#     a men's pocket-square print name, not a rangoli decoration), "decor" (7
#     matches, the generic-typed hits are "Star Decor Metal Tassel Earrings"/
#     "Flower Decor Drop Earrings" -- real wearable earrings using "decor" as
#     a style descriptor), "diya"/"toran"/"festoon"/"streamer"/"candle"/
#     "return gift"/"memento"/"party supply" -- 0 matches catalogue-wide.
# is_occasion_merchandise_name below also now checks detail_desc (not just
# prod_name): "Ellaichi Brooch" and 33 of its 35 collection-siblings carry NO
# gift/hamper/favour word in the NAME itself (e.g. "Ellaichi Bracelet",
# "Shell And Jhumki Earchain", "Elaichi Swagat Mala") -- only the shared
# description names the collection as guest favours. Full re-audit of the
# combined (name OR desc) x (existing + favour) pattern under generic types,
# catalogue-wide: 124 matching rows, zero false positives (98 ishhaara + 26
# voylla, all previously-verified rakhi/idol/hamper/favour merchandise).
# 2026-07-30 addition ("gift card"): the actual catch for the live-proven
# "Anniversary Day E-Gift Card" leak (product_type_name=="Fashion", so the
# type-only exclusion above never saw it). Catalogue audit (21 gift-card
# rows, 100% genuine, zero false positives -- see _OCCASION_MERCHANDISE_TYPES
# comment above) confirms "gift card"/"giftcard" text is a safe marker; the
# pattern also matches "e-gift card"/"digital gift card" via the substring.
_OCCASION_MERCHANDISE_NAME_RE = re.compile(
    r"\brakhi\b|\braksha\s*bandhan\b|\bhamper\b|\bidol\b|\bidols\b"
    r"|\bshowpiece\b|\btealight\b"
    r"|\bfavour\b|\bfavours\b|\bfavor\b|\bfavors\b"
    r"|\bgift\s*card\b",
    re.IGNORECASE,
)


def is_occasion_merchandise_name(
    prod_name: str | None, product_type_name: str | None, detail_desc: str | None = None
) -> bool:
    """Return True if `prod_name` OR `detail_desc` names occasion merchandise
    AND `product_type_name` is a GENERIC catalog bucket carrying no apparel
    signal of its own (see _GENERIC_PRODUCT_TYPES).

    Complements is_occasion_merchandise_type (type-only exclusion) for rows a
    store tagged generically instead of "Rakhi"/"Gift Hamper"/"Idols"
    directly. A genuine apparel item whose name also mentions rakhi/gift
    ("Men's Yellow Lehariya Cotton Kurta Rakhi Gift Box for Brother", typed
    "kurta") is NEVER excluded here -- its type IS a real apparel type, so
    the AND-gate on _GENERIC_PRODUCT_TYPES protects it regardless of the
    name/desc match. Callers must gate this on the same apparel-intent
    occasion context as is_occasion_merchandise_type (see
    src.agents.graph._apply_occasion_merchandise_gate).

    `detail_desc` is optional (defaults to None, matched against nothing) so
    existing name-only callers keep working -- pass it whenever available: a
    whole ishhaara "Haldi & Mehendi Favours" collection (35 rows) carries the
    merchandise signal ONLY in the shared description, not the product name
    itself (see _OCCASION_MERCHANDISE_NAME_RE's 2026-07-24 comment block).
    """
    if (product_type_name or "").strip().lower() not in _GENERIC_PRODUCT_TYPES:
        return False
    return bool(
        _OCCASION_MERCHANDISE_NAME_RE.search(prod_name or "")
        or _OCCASION_MERCHANDISE_NAME_RE.search(detail_desc or "")
    )


# ---------------------------------------------------------------------------
# Loungewear-in-dress-bucket exclusion
# ---------------------------------------------------------------------------
# Live-proven bug (2026-07-13): "minimalist wedding guest dress" returned
# "Green Geometric Printed Cotton Kaftan Night Dress" (genuine libas sleepwear)
# with the assistant's own text calling it "perfect for a wedding celebration".
# product_type_name is correctly "dress" for these rows — the catalogue simply
# has sleepwear sitting in the same bucket as formal/wedding dresses, and there
# was no exclusion mechanism for it (unlike fabric-bolts/kids items above).
#
# Verified against data/processed/unified/catalogue.parquet: 15 rows in the
# "dress" bucket (all brand=libas) contain the literal phrase "Night Dress" /
# "Nightdress" ("Green Geometric Printed Cotton Kaftan Night Dress", "Blue
# Printed Cotton Night Dress", etc.) — catalogue-wide this phrase matches
# exactly 19 rows total, all brand=libas, all genuinely sleepwear (15 in
# "dress", 3 in "All Products", 1 already correctly tagged product_type_name
# ="kaftan"): zero false positives anywhere in the catalogue.
#
# Deliberately does NOT match on a bare "kaftan" (an initial hypothesis from
# the bug report, since 2 fashor + 1 virgio "dress"-bucket rows also contain
# "kaftan" without "night"). Checked those 3 rows' own detail_desc: fashor's
# "Kaftan Style Midi/Maxi Dress" is described as "Perfect for brunches,
# getaways, or breezy evenings"; virgio's "Kaftan Maxi Dress With Lace" as
# transitioning "from intimate gatherings to festive evenings" — genuine
# day/eveningwear, not sleepwear. A bare "kaftan" is also a legitimate
# standalone garment noun used on 71 other catalogue rows (kurtas, tops,
# tunics, co-ords) unrelated to loungewear, so matching on it alone would
# badly over-exclude. "night dress"/"nightgown" is the precise, calibrated
# signal for THIS pattern family.
#
# 2026-07-31 CLASS-BROADENING (live-proven bug): "baraat outfit for men"
# surfaced "Men Solid Multicolor Top & Pyjama Set" (product_type_name=
# "nightwear") in the bottom slot — root cause is classify_anchor's
# ETHNIC_BOTTOM_KEYWORDS containing a bare "pyjama" (added for genuine
# "kurta pajama" ethnic sets), which also misclassifies loungewear "Top &
# Pyjama"/"Night Suit" rows as ethnic_bottom, feeding them straight into
# compose_outfit's candidate pool with no loungewear check at all (that
# check only existed in graph.py's search_node path — see
# src.agents.outfit.coherence.is_coherent_candidate's gate 6 for the fix on
# the compose_outfit side). "night dress"/"nightgown" alone never catches
# this item (its name has neither word), so the underlying predicate needed
# broadening too, not just a new call site.
#
# Full catalogue audit (data/processed/unified/catalogue.parquet) of every
# candidate pattern, same false-positive discipline as "kaftan" above:
#   - "night suit"/"nightsuit": 209 name-level matches, zero false positives
#     (every hit, including 3 rows typed "top"/"Tops & Blouson"/"shorts" and
#     11 typed "All Products", is genuinely sleepwear on inspection — e.g.
#     "Bewakoof Women White & Blue Printed Cotton Night Suit" is literally a
#     "T-shirt and Pyjamas" set; "Black Floral Print Night Suit Set"'s own
#     desc says "sleepwear collection"). The OLD comment above claiming
#     "night suit" rows were "already correctly tagged" and therefore safe
#     to leave unmatched was the latent bug: nothing downstream actually
#     excluded them once classify_anchor's ETHNIC_BOTTOM_KEYWORDS pulled a
#     "pyjama"-bearing one into an ethnic look's candidate pool.
#   - "night set(s)": 7 matches, zero false positives (coord/Nightwear-typed
#     rows, all genuine loungewear per their own desc, e.g. "Maroon Solid
#     Round Neck Night Sets" -> "Tshirt and Lower ... comfort and style").
#   - "loungewear"/"lounge wear"/"loungwear" (catalogue typo variant): 20
#     matches (16 product_type_name=="Lounge Wear", 4 =="Nightwear"), zero
#     false positives.
#   - "sleepwear": 0 name-level matches catalogue-wide (only 2 rows carry it
#     as a product_type_name FACET value, never in the free-text name) — no
#     catalogue support for a text pattern, so none added, same "zero
#     genuine support -> don't add" discipline as
#     is_occasion_merchandise_name's rangoli/decor rejections.
# "night suit"/"night shorts" no longer need the old carve-out — "night
# suit" is now matched outright (above); "night shorts" is UNCHANGED and
# still deliberately unmatched (a "Basic Shorts, Night Shorts, Gym Shorts"
# multi-use item has no dedicated sleep-only signal the way "night suit"
# does — see TestIsLoungewearText).
_LOUNGEWEAR_MARKER_RE = re.compile(
    r"\bnight\s*dress\b|\bnightgown\b|\bnight\s*gown\b"
    r"|\bnight\s*suit\b|\bnightsuit\b|\bnight\s*sets?\b"
    r"|\bloungewear\b|\blounge\s*wear\b|\bloungwear\b",
    re.IGNORECASE,
)

# "Top"/"T-shirt" + "pyjama(s)" co-occurring in the SAME name is the other
# unambiguous loungewear signal this class-broadening found — the exact
# live-bug item "Men Solid Multicolor Top & Pyjama Set" carries neither a
# "night"/"lounge" word NOR a useful detail_desc (most of its 8 catalogue
# duplicates have detail_desc "Good quality product"/"SOFT FABRIC"), so only
# a name-level top+pyjama combo check catches it. Catalogue audit: 26
# name-level matches. 25/26 genuine loungewear (T-shirt/top + pyjama
# co-ord sets, e.g. "iki chic Women Multicoloured Printed T-shirt with
# Pyjamas", "Sheomy Men Top - Pyjama Set Thermal"). The 1 exception —
# "Libas Women Navy Blue Sequinned Top with Pyjamas & Longline Jacket" — is
# a genuine ethnic 3-piece set (its own desc: "top, salwar and ethnic
# jacket... sequinned top... longline ethnic jacket"; the freeform name
# just mislabels the salwar as "pyjamas"), excluded via the "jacket" escape
# below (zero collateral: it is the only jacket-bearing row among the 26).
# Deliberately does NOT gate on a bare "pyjama" alone (no "top"/"t-shirt"
# co-occurrence) — the wider audit for this task found the catalogue is
# full of legitimate ethnic/festive bare-pyjama garments mistyped
# product_type_name="nightwear" (e.g. "Men's Black Cotton Blend Patiala
# Pyjama": desc "appropriate for a range of occasions... Style this for
# wedding ceremonies or pujas and festivals"; "Navy Blue Solid Cotton Silk
# Blend Aligarhi Pajama": desc "Perfect for traditional days, festivals,
# weddings... Complete the look with Kolhapuri footwear and ethnic
# kurtas"; VASTRAMAY-brand churidars/Patiala pyjamas throughout) — a bare
# "pyjama" match would have badly over-excluded exactly the groom/baraat
# formalwear this fix exists to protect. Same discipline applies to
# "sherwani"/"indo western"/"jodhpuri" + pyjama combos (27 catalogue rows,
# e.g. "Men Grey Kenzo Jacquard Silk Blend Sherwani with Cream Pyjama Set",
# explicitly the churidar-pyjama worn under a groom's sherwani) — none of
# those contain "top"/"t-shirt", so this pattern already leaves them
# untouched without needing a separate escape.
_TOP_PYJAMA_COMBO_RE = re.compile(
    r"\b(top|t-shirt|tshirt)\b.*\bpyjamas?\b|\bpyjamas?\b.*\b(top|t-shirt|tshirt)\b",
    re.IGNORECASE,
)
_TOP_PYJAMA_JACKET_ESCAPE_RE = re.compile(r"\bjacket\b", re.IGNORECASE)


def is_loungewear_text(text: str | None) -> bool:
    """Return True when *text* describes sleepwear/loungewear — a "night
    dress"/"nightgown", a "night suit"/"loungewear" item, or a "top"+"pyjama"
    co-ord loungewear set (see _LOUNGEWEAR_MARKER_RE / _TOP_PYJAMA_COMBO_RE
    docstrings above for the full catalogue audit each pattern is grounded
    in).

    NOT wired in as an unconditional hard exclusion (unlike is_fabric_bolt_text
    and is_kids_item above) — a bare "night dress" query with no wedding/formal
    occasion signal has a legitimate reason to want these items. Callers must
    gate this predicate on occasion context before applying it.
    """
    if _LOUNGEWEAR_MARKER_RE.search(text or ""):
        return True
    if _TOP_PYJAMA_COMBO_RE.search(text or ""):
        return not _TOP_PYJAMA_JACKET_ESCAPE_RE.search(text or "")
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
# Religious decor exclusion (idols / statues / frames / puja items)
# ---------------------------------------------------------------------------
# Jewellery-inventory-gap wave (2026-07-19): theamethyststore.com was added
# specifically to close the fashion-jewellery gap in the catalogue, but its
# feed also carries a "Silver Idols" product line (25 rows, e.g. "Balaji
# Temple Statue" @ Rs 19,79,480 — the single highest-priced item in the
# ENTIRE theamethyststore catalogue — and "Venkateswara 3D Idol" @ Rs
# 4,85,520), a "Kum Kum Box" line (9 rows, vermilion-powder puja
# containers), and 4 rows of religious photo frames ("Lakshmi Frame",
# "Ganesha Frame", "Perumal Frame", "Lord Murugan Frame With Peacock 3D
# SPL") mixed into the store's generic "Fashion" type bucket. These are
# temple/home decor, not wearable fashion jewellery — the entire premise
# this brand was onboarded for — and their outlier prices (idols alone:
# mean Rs 1,63,077 vs Rs 36,155 catalogue-wide) would skew price-based
# search/ranking and surface non-jewellery results if left in.
#
# Matched on title text, not the store's own "type" label, because "Fashion"
# also holds genuine jewellery (brooch pins, saree pins, waist charms) that
# must NOT be excluded — a type-label filter would either miss the frames or
# over-exclude real jewellery sharing the same label.
#
# "idol"/"idols" must NOT match when preceded by "non " / "non-": daivik uses
# "Non Idol"/"Non-Idol" as a genuine jewellery style descriptor (temple
# jewellery that does NOT feature a deity-idol motif, as opposed to "idol"-
# style pieces that do) — e.g. "Antique Non Idol Purple Long Necklace with
# Earrings", "Non-Idol Gold Polish Earrings", "AD Non Idol Jada/Hair
# accessory" (11 real daivik rows, all product_type_name Necklace/EARRINGS/
# BANGLES/OTHER ESSENTIALS — genuine jewellery, not decor). The negative
# lookbehind is fixed-width (4 chars: "non " / "non-") so it works with
# Python's `re` module without a variable-width-lookbehind library.
_RELIGIOUS_DECOR_RE = re.compile(
    r"(?:(?<!non )(?<!non-)\bidols?\b)|\bstatues?\b|\bkum\s*kum\s*box\b|\bframes?\b",
    re.IGNORECASE,
)


def is_religious_decor_item(prod_name: str | None) -> bool:
    """Return True when *prod_name* is a religious statue/idol/frame/puja item.

    Not fashion jewellery or apparel — see module comment above
    _RELIGIOUS_DECOR_RE for the theamethyststore rows that motivated this.
    """
    return bool(_RELIGIOUS_DECOR_RE.search(prod_name or ""))


def drop_religious_decor_items(
    df: pd.DataFrame,
    *,
    name_col: str = "prod_name",
) -> tuple[pd.DataFrame, int]:
    """Drop rows that are religious decor (idols/statues/frames/puja items).

    These are removed entirely from the index (not merely filtered at query
    time) since a jewellery/fashion shopping assistant has no legitimate use
    for temple statues or puja containers regardless of query context.

    Returns (df, n_dropped).
    """
    is_decor = df[name_col].fillna("").astype(str).apply(is_religious_decor_item)
    n = int(is_decor.sum())
    out = df.loc[~is_decor].reset_index(drop=True)
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
