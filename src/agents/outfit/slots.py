from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.agents.outfit import body_type as body_type_module
from src.agents.outfit.occasions import EITHER, ETHNIC_HEAVY, ETHNIC_ONLY, get_occasion

# ── Anchor type detection — keyword sets keyed on product_type_name (lowercase) ──

ETHNIC_TOP_KEYWORDS: frozenset[str] = frozenset({
    "kurta", "kurti", "kameez", "tunic", "kaftan",
    # 2026-07-30 (unknown-class keyword-coverage audit): "phiran" (31 rows,
    # e.g. "Kashifa Black Pure Woollen Phiran") is a Kashmiri traditional
    # woollen robe/cloak — grouped with "kaftan" as a full-length ethnic
    # robe-style garment.
    "phiran",
})
ETHNIC_ONE_PIECE_KEYWORDS: frozenset[str] = frozenset({
    "lehenga", "saree", "anarkali", "suit-set", "suit set", "sharara set",
    "salwar kameez", "palazzo set", "ethnic dress", "gown",
})
ETHNIC_BOTTOM_KEYWORDS: frozenset[str] = frozenset({
    "palazzo", "palazzos", "churidar", "salwar", "sharara", "pyjama", "dhoti",
    # 2026-07-25 (set-not-single follow-up): "pajama" (American spelling) is a
    # real, prevalent alternate to "pyjama" in this catalogue -- 1,386 rows
    # use it, including "kurta pajama" bare-juxtaposition listings (1,348
    # rows) that were silently invisible to every noun-counting check below.
    "pajama",
    # 2026-07-30 (unknown-class keyword-coverage audit): "plazzo"/"plazzos"
    # is a real catalogue typo/variant of "palazzo" that doesn't match via
    # substring ("palazzo" is already covered above).
    "plazzo", "plazzos",
})
WESTERN_TOP_KEYWORDS: frozenset[str] = frozenset({
    "shirt", "t-shirt", "tshirt", "top", "blouse", "sweater", "sweatshirt",
    "tank top", "crop top", "polo",
    # Wave 9 (2026-07-23, gym occasion): the new activewear catalogue's own
    # product_type_name facet is the literal string "sports_bra" (underscore,
    # no space) — "sports bra" (space) also catches the freeform-name variant.
    # Deliberately NOT a bare "bra" (2,509 catalogue rows contain that
    # substring, including unrelated jewellery like "Brass Necklace").
    "sports_bra", "sports bra",
    # 2026-07-30 (unknown-class keyword-coverage audit): "knitwear" (1,744
    # rows, e.g. "High Neck Textured Zipper Sweater", "URBANIC Women Pink
    # Solid Distressed Cardigan") is a genuine sweater/cardigan/sweatshirt
    # facet value. "skivvy" (54 rows, e.g. "Wine Viscose Skivvy Pullover") is
    # a lightweight pullover/turtleneck style, same category.
    "knitwear", "pullover", "pullovers", "skivvy",
})
WESTERN_BOTTOM_KEYWORDS: frozenset[str] = frozenset({
    "trousers", "jeans", "shorts", "skirt", "jeggings",
    # 2026-07-11: "pant"/"pants" is a distinct catalogue naming convention from
    # "trousers" ("Kurta with Pant & Dupatta") — same garment class, was
    # previously unrecognised by this keyword set entirely.
    "pant", "pants",
    # Wave 9 (2026-07-23, gym occasion): "leggings"/"joggers"/"skort" are new
    # product_type_name facet values from the activewear catalogue merge with
    # no prior keyword coverage here — without these, classify_anchor/
    # classify_item resolve them to "unknown", which SLOT_ALLOWED_CLASSES
    # then hard-rejects from ever filling a "bottom" slot at all. (track_pants/
    # cargo_pants already resolve via the existing "pants" substring match.)
    "leggings", "joggers", "skort",
    # 2026-07-30 (unknown-class keyword-coverage audit): "bottom"/"bottoms"/
    # "bottomwear" sampled as "Cotton Linen Lower", "Cargo Pant",
    # "Trackpants", "Men Cargos", "Three Fourths" — genuine Western
    # trousers/joggers/cargo/capri-length pants, no ethnic markers found in
    # samples. "cargo" (pt=="Cargo", 19 rows) means cargo pants
    # specifically as a facet value in this catalogue. "capri"/"capris",
    # "culottes", "breeches", "wide leg" are self-evidently Western trouser
    # styles. "trackpant"/"trackpants"/"track pant" and "three fourth"/
    # "three fourths" are additional catalogue naming variants.
    #
    # Deliberately EXCLUDES bare "lower"/"lowers" (spec'd but dropped as a
    # verified regression): as a plain substring these live-match inside
    # "flower"/"sunflower"/"flowers" — 943 catalogue rows, catalogue-
    # verified to flip garments (tops, shirts, skirts, kurtas, lehengas —
    # anything with a floral name) to "western_bottom". This isn't just an
    # unknown-vs-classified nuance: classify_anchor() checks WESTERN_BOTTOM
    # (this set) BEFORE WESTERN_TOP, so a plain "Top" with "Flower" in its
    # name would be misclassified as a BOTTOM. Worse, composer.py calls
    # classify_anchor() directly on the look's own ANCHOR/seed item to
    # decide get_fill_slots(anchor_class, ...) — the entire slot
    # composition for the look — so this collision would have silently
    # composed top-anchored looks as if they were bottom-anchored (filling
    # a bogus "top" slot, skipping the real "bottom" slot). The ~42 catalog
    # rows with product_type_name literally "Lower"/"Lowers" (the audited
    # facet values this was meant to cover) stay "unknown" as a result —
    # an accepted, deliberate coverage gap given the regression's severity
    # (943 rows corrupting anchor classification vs. 42 rows of unresolved
    # coverage). A facet-EQUALITY check (matching _WESTERN_FORMAL_CAPABLE_
    # TYPES' pattern above) would resolve both safely, but that's a new
    # matching mechanism, out of this task's declared scope (keyword-set
    # additions to the existing substring-scan control flow only).
    "bottom", "bottoms", "bottomwear", "capri", "capris",
    "culottes", "breeches", "cargo", "wide leg", "trackpant", "trackpants",
    "track pant", "three fourth", "three fourths",
})
WESTERN_ONE_PIECE_KEYWORDS: frozenset[str] = frozenset({
    "dress", "jumpsuit", "playsuit", "dungarees", "co-ord",
    # 2026-07-30 (unknown-class keyword-coverage audit): "coord" (no hyphen,
    # a distinct string from the already-covered "co-ord") — 1,138 rows
    # pt-alone, 32 remaining after the real-name pass. Tracksuits are sold
    # as a single top+bottom listing, given the same "one_piece for
    # classification purposes" treatment as co-ord.
    "coord", "tracksuit", "tracksuits", "track suit",
})
OUTERWEAR_KEYWORDS: frozenset[str] = frozenset({
    "jacket", "coat", "blazer", "cardigan", "nehru jacket", "waistcoat",
    "parka", "anorak", "sherwani", "bandhgala",
    # 2026-07-30 (unknown-class keyword-coverage audit): "outerwear" (3,326
    # rows pt-alone), "shrug"/"shrugs" (~93 rows, e.g. "Winter Shrug",
    # cardigan-like outer layer), "capes" (60 rows combined with "poncho"
    # below, e.g. "Ponchu & Capes"/"Capes & Overlays" — the facet VALUE
    # itself is always plural "Capes", so "capes" alone catches both real
    # facet values with zero substring risk), "poncho" (catches the
    # "Ponchu" catalogue typo via the combined-text fallback where the
    # freeform name itself says poncho), "shacket" (2 rows, shirt-jacket
    # hybrid), "tuxedo"/"tuxedos" (~81 rows, western formal), "business
    # suit" (25 rows, e.g. "MLS BUSINESS PLAIN SUIT 3PCS" — genuine western
    # business suit, distinct from the separately-audited "Suits" facet
    # value which is 100% ethnic salwar-suit-sets already handled via
    # "suit set" in ETHNIC_ONE_PIECE_KEYWORDS; deliberately NOT bare
    # "suit"/"suits").
    #
    # Deliberately EXCLUDES bare singular "cape" (spec'd but dropped as a
    # verified regression — see slots.py module-level deviation note near
    # WESTERN_BOTTOM_KEYWORDS's dropped "lower"/"lowers" for the mechanism;
    # "cape" as a plain substring live-matches inside "Escape"/"Seascape"/
    # "Dreamscape" brand/copy text, catalogue-verified to flip a plain
    # T-shirt (pt="top", "Endless Escape Oversized Stretch T-Shirt") and a
    # pair of Mule Heels (pt="footwear", "Email Escape : Mule Heels") to
    # "outerwear" — the footwear case is a direct regression against this
    # same task's own footwear-exclusion-gate goal). "capes" (plural) has
    # zero such collisions and is sufficient: both real facet values
    # ("Ponchu & Capes", "Capes & Overlays") are already plural.
    "outerwear", "shrug", "shrugs", "capes", "poncho", "shacket",
    "tuxedo", "tuxedos", "business suit",
    # 2026-08-05 (unknown-row apparel audit): "sadri" (14 rows,
    # product_type_name=="sadri", e.g. "Charcoal Grey Multi-Button Sadri",
    # "Black Sadri with Contrast Patch Pockets") is a structured sleeveless
    # ethnic waistcoat/gilet — same category as the already-covered
    # "waistcoat" above. Verified zero collision risk: all 19 catalogue rows
    # containing "sadri" anywhere in prod_name are either pt=="sadri"/
    # "Sadri" (14+1, this fix) or pt=="kurta" (4, already correctly
    # ethnic_top via the "kurta" keyword regardless).
    "sadri",
})
FOOTWEAR_KEYWORDS: frozenset[str] = frozenset({
    "shoes", "sandals", "boots", "heels", "flats", "sneakers",
    "juttis", "jutti", "mojaris", "mojari", "kolhapuris", "kolhapuri",
    "wedges", "loafers", "pumps",
    # 2026-07-30 (unknown-class keyword-coverage audit): "footwear" (4,152
    # rows pt-alone, genuinely shoes/sneakers/juttis/loafers by sample),
    # "chappal"/"chappals" (76 rows, "1 Pair of Chappals"), "mule"/"mules"
    # (~75 rows), "slider"/"sliders"/"slide"/"slides" (~155 rows, "Sliders"/
    # "Snip Cut Slides"/"Flip Flops & Slides" variants), "flip flop"/
    # "flip flops", "mary jane"/"mary janes" (7 rows), "clog"/"clogs" (7
    # rows, "Women's Clogs"), "ballerina"/"ballerinas" (2 rows).
    "footwear", "chappal", "chappals", "mule", "mules", "slider", "sliders",
    "slide", "slides", "flip flop", "flip flops", "mary jane", "mary janes",
    "clog", "clogs", "ballerina", "ballerinas",
    # 2026-08-05 (unknown-row apparel audit): "loafer" (singular — 10 rows,
    # product_type_name=="Men's Loafer", plain given-name listing titles
    # like "Victor"/"Evan" with zero other descriptive text) is a distinct
    # catalogue string from the already-covered plural "loafers". Verified
    # zero collision risk: of the 111 total rows containing "loafer"
    # anywhere in prod_name, 102 are already pt=="footwear" (self-resolving
    # via the bare "footwear" keyword above regardless) and the rest are
    # pt=="Loafers"/"bag" (already correctly resolved via other keywords) —
    # none flip to a wrong class by adding the singular form.
    "loafer",
})
MEN_FORMALWEAR_KEYWORDS: frozenset[str] = frozenset({
    "sherwani", "bandhgala", "nehru jacket",
    # 2026-07-30 (unknown-class keyword-coverage audit): "achkan" (46 rows,
    # e.g. "PINK BROCADE EMBROIDERED ACHKAN" — ethnic men's formal long
    # coat, same category as sherwani/bandhgala).
    #
    # Deliberately NOT bare "jodhpuri" (spec'd but dropped as a verified
    # regression): "jodhpuri" is genuinely ambiguous in this catalogue's own
    # vocabulary, not just an accidental substring collision — 19 real
    # product_type_name=="footwear" rows use it as a legitimate FOOTWEAR
    # style term ("Jodhpuri Mojaris"/"Jodhpuri Boots"), and MEN_FORMALWEAR
    # is checked before FOOTWEAR in classify_anchor()'s priority order, so a
    # bare "jodhpuri" keyword here would misclassify that footwear as
    # men_formalwear. Confirmed to break an EXISTING, deliberately-tested
    # coherence.py contract (TestWesternRegisterGateOffice::
    # test_plain_jodhpuri_mention_outside_outerwear_not_rejected in
    # tests/test_phase_b_gender_slot_coherence.py) — gate 4 there rejects
    # is_ethnic_item(pt, name) candidates for the office register, and
    # is_ethnic_item includes "men_formalwear"; a Jodhpuri Mojaris would be
    # wrongly rejected from an office footwear slot. Using the narrower
    # phrases "jodhpuri suit"/"jodhpuri_suit"/"jodhpuri set" instead
    # (catalogue-verified: 0 footwear-pt collisions) still covers the
    # audited target — product_type_name=="jodhpuri_suit" (346 rows) and
    # =="JODHPURI SET"/"Jodhpuri set"/"open jodhpuri set" (13 rows
    # combined) — while leaving the 45 bare product_type_name=="Jodhpuri"
    # rows unclassified; those are all "Boy's ..." KIDS listings anyway
    # (out of scope per this task's own kids-item exclusion). NOTE:
    # coherence.py's `_JODHPURI_OUTERWEAR_RE` is a separate, narrower
    # mechanism (rejecting Jodhpuri-named OUTERWEAR from the office register
    # specifically) and is untouched by this base-classification change.
    "achkan", "jodhpuri suit", "jodhpuri_suit", "jodhpuri set",
})

# Structural, facet-based western product types capable of formal/glam
# register (checked via product_type_name FACET EQUALITY, same pattern as
# graph.py's "indowestern" gate-4 check — never a substring scan of the
# free-text listing copy). Catalogue-audited 2026-07-30: "dress" rows
# almost never literally say "gown" (4/1580), which is why the OLD
# name-substring allow-list rejected 84% of genuinely reception-
# appropriate dresses. "trousers"/"shirt"/"blazer" are clean, distinct
# facet values in this catalogue (product_type_name=="shirt" has only
# 11/8734 rows whose own name says t-shirt/tee — negligible noise).
# Deliberately EXCLUDES "suit"/"suits": product_type_name=="suits" here
# is 100% ethnic salwar-suit-sets (254/254 rows, all gender=="women",
# e.g. "Burgundy Floral Embroidered Chanderi Silk Straight Suit Set"),
# already correctly classified ethnic_one_piece by classify_anchor via
# ETHNIC_ONE_PIECE_KEYWORDS' "suit set" entry — never a western business
# suit in this catalogue's vocabulary.
_WESTERN_FORMAL_CAPABLE_TYPES: frozenset[str] = frozenset({
    "dress", "jumpsuit", "trousers", "pant", "pants", "shirt", "blazer",
})

# ── Accessory sub-families (Phase B Part 1) ─────────────────────────────────
# Kept as separate small families (rather than one flat ACCESSORY_KEYWORDS set)
# so accessory_query_matches() can require a candidate to share a FAMILY with
# the slot's own query — e.g. a dupatta-seeking slot must never accept a
# handbag, and a "belt watch cap" slot must never accept a dupatta.
# 2026-07-30 catalogue audit (Sukkhi "Kada" bangle + "Goddess Traditional
# Haram" necklace live misses on a generic haldi-look query): audited every
# distinct product_type_name facet value that classify_item() currently
# resolves to "unknown" and is genuinely an accessory/jewellery item. Two root
# causes found, both closed below across all four families that had gaps:
#   1. Pluralization/typo mismatch — _contains_word requires a literal word
#      match, so singular keywords ("bangle", "bracelet", "earring", "anklet",
#      "pendant", "jhumka", "bag", "belt", "watch", "pocket square") never
#      matched their PLURAL catalogue facet values ("Bangles", "Bracelets",
#      "Anklets", "Pendants", "Jhumkas", "Bags", "Belts", "Watches", "Pocket
#      Squares"), and vice versa.
#   2. Missing vocabulary — genuine Indian bridal/traditional jewellery terms
#      (juda/hair-bun ornament, passa, kalangi, borla, hathphool, mathapatti,
#      oddiyanam, nath, chandbalis, mangalsutra, kada, haram, chura) were never
#      covered at all. Every one of these was sampled against the real
#      unified catalogue to confirm genuine jewellery/accessory rows with zero
#      false-positive risk — e.g. "Judo" facet value -> "Multicolor Gold
#      Plated Mishr Juda"/"White Color Gold Plated Jadau Kundan Juda" (a real,
#      if misspelled/OCR'd, facet value for hair-bun jewellery); "Kalangi" ->
#      "Kings of Rajasthan Haritansh Kalangi" (turban ornament); "Oddiyanam/
#      Hip Belts" -> "Bridal Nethra Diamond Like Oddiyanam" (bridal waist
#      jewellery, NOT a functional western belt).
# "bascelet"/"nacklaces"/"imitation jewerllery" are real catalogue TYPOS (the
# facet/name values themselves, not typos of this fix) — included verbatim so
# the match fires against the actual data. Deliberately EXCLUDES garment-class
# facet values (footwear/outerwear/knitwear/swimwear/nightwear/etc.) — a
# separate, much larger classification gap, out of scope here.
_ACCESSORY_DUPATTA_FAMILY: frozenset[str] = frozenset(
    # 2026-08-06 (full accessory-vocabulary audit): "scraf" (3 rows,
    # product_type_name=="Scraf" — a real catalogue typo of "scarf", same
    # typo-tolerance discipline as "bascelet"/"nacklaces" in the jewellery
    # family above). Deliberately EXCLUDES bare "shawl" (262 rows total,
    # sampled: 103 Fashion/100 All Products genuinely shawl accessories,
    # but also 4 knitwear rows that are "Shawl Collar Cardigan"/"Shawl Neck
    # Sweater" — a real garment style descriptor, not a shawl accessory,
    # and knitwear is a _GENERIC_FACET_VALUES entry so those rows are NOT
    # protected by classify_item's pt-alone shortcut the way "choker"/
    # "hasli" above are for their one non-jewellery pt hit). The clean
    # subset (product_type_name=="Shawls" exactly, 5 rows) is caught
    # instead via classify_item()'s own _ACCESSORY_FACET_VALUES below.
    {"dupatta", "stole", "scarf", "scraf"}
)
_ACCESSORY_BAG_FAMILY: frozenset[str] = frozenset(
    {
        "bag", "handbag", "sling", "clutch", "tote", "bags", "potli", "potlis",
        # 2026-07-30 (unknown-class keyword-coverage audit): "wallet"/
        # "wallets" (7+5 rows), "card holder" (5 rows), "backpack"/
        # "backpacks" (2 rows) — self-evidently bag-family accessories.
        "wallet", "wallets", "card holder", "backpack", "backpacks",
        # 2026-08-06 (full accessory-vocabulary audit): "pouch"/"pouches"
        # (4 Fashion-pt rows genuinely new, e.g. "Red Stone Studded
        # Rectangle Phone Pouch"; the rest already resolve via pt=="bag"
        # regardless) — same clutch/potli-adjacent bag category.
        "pouch", "pouches",
    }
)
_ACCESSORY_JEWELLERY_FAMILY: frozenset[str] = frozenset(
    # "pendant" added 2026-07-25 (out-of-sample validation finding): a
    # "Pendant" product_type row (126 in the catalogue, all jewellery) slipped
    # past the new accessory-exclusion gate for "office outfit for men" —
    # was never in ACCESSORY_KEYWORDS at all before this.
    {
        "jewellery", "jewelry", "jhumka", "earrings", "necklace", "bangle", "pendant",
        "ring", "rings", "bangles", "bracelet", "bracelets", "bascelet",
        "necklaces", "nacklaces", "chokers", "earring", "toe ring", "toe rings",
        "nose ring", "nosering", "nosepin", "pendants", "jhumkas",
        "maangtikka", "maangteeka", "maang tika", "mang teeka", "mathapatti",
        "borla", "nath", "chandbalis", "kada", "imitation jewellery",
        "imitation jewerllery", "anklet", "anklets", "payal", "hair maatal",
        "jada", "hair accessories", "hair accessory", "hairband", "oddiyanam",
        "hip belts", "waist belts", "hipbelts", "brooch", "brooches",
        "cufflink", "cufflinks", "hathphool", "kalangi", "passa", "judo",
        "juda", "mangalsutra", "earchain", "earchains", "ear chain",
        "haram", "chura",
        # 2026-07-30 follow-up (found live-testing the fix above, same audit
        # discipline): "kaleera" (80 rows, bridal wrist ornament, e.g.
        # "Traditional Golden Bridal Kaleera") and "varmala"/"jaimala" (5
        # rows combined, wedding garlands, e.g. "Kundan Stone Varmala For
        # Bride Groom") both surfaced live for "bridal look for women" —
        # neither was in the original audit pass. "jadau" (a jewellery-making
        # technique term, 3,179 rows) added too: sampled facet-value
        # distribution is 100% already-jewellery types (Necklace Sets,
        # Earrings, Rings, Bangles, etc.) except 261 rows typed generic
        # "Fashion", which need this keyword to be caught via the
        # combined-text fallback. Distinct word from "jada" above (hair-bun
        # ornament) — "jadau" has no word boundary after "jada", so \bjada\b
        # never accidentally matches it.
        "kaleera", "varmala", "jaimala", "jadau",
        # 2026-07-30 follow-up (unknown-class keyword-coverage audit):
        # "Nose Pin" (space variant, 4 rows) is distinct from the
        # already-covered "nosepin"/"nose ring". "Maang Tikka" (double-k, 3
        # rows) and "Mangtika" (no space, 1-2 rows) are distinct spellings
        # from the already-covered "maangtikka"/"maangteeka"/"maang tika".
        # "Matha Patti" (with space, 5 rows) is distinct from
        # already-covered "mathapatti" (no space). "Kamarband"(3)/
        # "Bajuband"(4)/"Armlet(s)"(2) are genuine ethnic waist/arm
        # jewellery, same category as "oddiyanam"/"hathphool" already
        # covered. "Parandi"(5, hair braid tassel), "Hairbun(s)"(3) are
        # hair ornaments, same category as "hair maatal"/"jada" already
        # covered. "Mala"(3, beaded necklace/rosary), "Neckpiece"/"Neck
        # Piece"(8) are necklace-type jewellery. "Stud"/"Studs"(6) are
        # earring studs. "Manglasutra"(1) is a typo variant of the
        # already-covered "mangalsutra" (letters transposed). "Hand Cuff"
        # remaining rows sampled as "Victorian Dollar Hand Cuff", "Multi
        # Color Stone Hand Cuff" — genuine cuff-bracelet jewellery.
        #
        # Deliberately NOT "accessories"/"accessory" (spec'd but dropped as
        # a verified regression): these generic catch-all words are used
        # throughout THIS FILE's own SlotSpec.search_query register text
        # (e.g. "dupatta jewellery clutch ethnic accessory festive
        # embroidered", "pocket square safa ethnic accessory") as
        # non-family filler tokens, not family-specific vocabulary.
        # split_accessory_query_by_family() (below) scans every word in
        # _ACCESSORY_FAMILIES against the QUERY text too — putting
        # "accessory" in the jewellery family made it match as a second
        # "family" in every existing single-family accessory query that
        # happens to carry the word "accessory" as filler, corrupting the
        # split (confirmed: 3 tests in tests/test_phase_b_gender_slot_
        # coherence.py's TestSplitAccessoryQueryByFamily broke — a query
        # that should stay a single-family no-op got force-split, and a
        # genuine 2-family split gained a spurious 3rd "accessory"-only
        # sub-query). The audit's original reasoning ("this doesn't blur
        # accessory_query_matches() since that's keyed on the SPECIFIC
        # matched family") only checked ONE of the two functions that read
        # _ACCESSORY_FAMILIES — split_accessory_query_by_family() is
        # exactly as family-keyed, and a generic catch-all word placed in
        # any one family breaks its single-vs-multi-family detection for
        # every query containing that word. Fixing this properly (a facet-
        # equality check for bare product_type_name=="Accessories") is a
        # new mechanism, out of this task's keyword-set-only scope.
        "nose pin", "maang tikka", "mangtika",
        "matha patti", "kamarband", "bajuband", "armlet", "armlets",
        "parandi", "hairbun", "hairbuns", "mala", "neckpiece", "neck piece",
        "stud", "studs", "manglasutra", "hand cuff",
        # 2026-08-06 (full accessory-vocabulary audit, closing the leak this
        # time rather than another incremental pass — see classify_item()'s
        # own _ACCESSORY_FACET_VALUES for the catch-all-bucket half of this
        # fix). Every term below was sampled against real catalogue rows,
        # not guessed, and checked for substring collisions against the
        # FULL catalogue (not just the accessory subset) before inclusion:
        #   "choker" (singular — 2,298 rows total incl. "chokers" plural
        #     already covered; sampled the one non-jewellery pt hit,
        #     "Choker Neck ... Top" (product_type_name=="top") -- protected
        #     from ever reaching this combined-text scan by classify_item's
        #     own pt-alone shortcut, since "top" is a specific facet value
        #     that resolves western_top before this family is even checked;
        #     same protection mechanism as the "Shawl Collar Cardigan"
        #     false-positive avoided below by NOT adding bare "shawl").
        #   "hasli" (304 rows, sampled: necklace/jewellery pt overwhelmingly,
        #     zero collision risk found).
        #   "gajra" (25 rows, floral hair ornament, e.g. "Jasmine Flower
        #     Hair Gajra"), "scrunchie"/"scrunchies" (hair tie; the 2
        #     garment-pt hits, "...Palazzos with Scrunchie"/"...Top with
        #     Palazzos & Scrunchie", are bundle listings protected by the
        #     same pt-alone-shortcut mechanism as "choker" above).
        #   "rakhi" (715 rows incl. Raksha Bandhan hamper/combo listings
        #     that still centre on a real wearable rakhi, e.g. "Bond of
        #     Love Bhaiya Bhabhi Rakhi Gift Hamper" — same "bundle keeps its
        #     core item's classification" precedent as "kurta with
        #     dupatta"). Deliberately declined the surrounding pure gift-
        #     merchandise buckets with no wearable core (Gift Hamper, Gift
        #     Card, Jute basket, etc.) — out of this audit's scope, not the
        #     same class of item.
        #   "facelet" (8 rows, all Fashion-pt, catalogue typo/variant of
        #     "bracelet" — sampled 100% genuine bracelet-shaped jewellery).
        #   "neckalce" (31 rows, catalogue typo of "necklace" — distinct
        #     letter transposition from the already-covered "nacklaces").
        #   "hair pin"/"hair pins" (13 rows, most already covered via
        #     "jewellery"/"hair accessory" pt-alone; adds the 4 remaining
        #     Fashion-pt rows).
        #   "hair band" (26 rows — space variant distinct from the already-
        #     covered no-space "hairband").
        #   "maangtika" (single-k, no-space — distinct spelling from the
        #     already-covered "maangtikka"/"maang tika"/"mangtika").
        "choker", "hasli", "gajra", "scrunchie", "scrunchies", "rakhi",
        "facelet", "neckalce", "hair pin", "hair pins", "hair band", "maangtika",
        # 2026-08-06 follow-up (same audit, second sampling pass after the
        # first round of additions above): "kundan" (8,018 rows — a
        # jewellery-SETTING TECHNIQUE term, not a garment word; sampled the
        # full product_type_name distribution, 100% jewellery facet values
        # with zero non-jewellery collision found, unlike "tie"/"chain"/
        # "wrap" which were declined below for exactly that reason).
        # "earcuff"/"earcuffs"/"ear cuff" (200+12+91 rows, an ear-jewellery
        # style — deliberately NOT bare "cuff"/"cuffs", which also turn up
        # in "shirt"/"Joggers & Trackpants" as a genuine sleeve/hem style
        # descriptor). "kangan" (6, bangle-adjacent), "chudi" (1, Rajasthani
        # bangle term distinct from the already-covered "chura"), "jhumkha"/
        # "jhumkhas" (25, spelling variant of the already-covered "jhumka"),
        # "pasa" (8, spelling variant of the already-covered "passa"),
        # "haar"/"rani haar" (78+25, necklace-style terms), "kalgi" (6,
        # turban ornament), "hip chain" (4, waist jewellery, same category
        # as the already-covered "oddiyanam"/"kamarband"), "headband"/
        # "headbands" (14, distinct word from the already-covered "hairband"
        # / "hair band"), "diadem" (2, bridal tiara), "katar" (9, ceremonial
        # groom's dagger accessory), "sheeshphool" (44, hair-parting
        # jewellery), "hair braid" (2, distinct phrase from the already-
        # covered "parandi"). Every term sampled against its full catalogue
        # occurrence, not just the accessory subset, before inclusion.
        "kundan", "earcuff", "earcuffs", "ear cuff", "kangan", "chudi",
        "jhumkha", "jhumkhas", "pasa", "haar", "rani haar", "kalgi",
        "hip chain", "headband", "headbands", "diadem", "katar",
        "sheeshphool", "hair braid",
    }
)
_ACCESSORY_BELT_WATCH_FAMILY: frozenset[str] = frozenset({"belt", "watch", "belts", "watches"})
_ACCESSORY_EYEWEAR_CAP_FAMILY: frozenset[str] = frozenset(
    {
        "sunglasses", "cap",
        # 2026-08-06 (full accessory-vocabulary audit): "bandana"/"bandanas"
        # (4 rows, all product_type_name=="Accessories", e.g. "SNITCH x
        # BISMIL Printed Cotton Bandana") — headwear, same family as cap.
        "bandana", "bandanas",
    }
)
_ACCESSORY_MENSWEAR_FORMAL_FAMILY: frozenset[str] = frozenset(
    # 2026-07-30 (unknown-class keyword-coverage audit): "bow tie" (5 rows,
    # unambiguous). Deliberately excludes bare "bow" — only 3 rows and too
    # ambiguous a word for safe word-boundary matching in this family.
    {
        "pocket square", "safa", "pocket squares", "bow tie",
        # 2026-08-06 (full accessory-vocabulary audit): "handkerchief"/
        # "handkerchiefs" (12 rows, all product_type_name=="Clothing
        # Accessories", e.g. "Cotson Premium Cotton Handkerchief for Men")
        # — same formal-menswear-accessory category as pocket square.
        # "pagri" (16 rows, e.g. "MLS Cotton Silk Pagri") and "pheta" (2
        # rows, "Pheta Fabric Stripes" — a Maharashtrian turban cloth) are
        # both turban terms, same category as the already-covered "safa".
        "handkerchief", "handkerchiefs", "pagri", "pheta",
    }
)

_ACCESSORY_FAMILIES: tuple[frozenset[str], ...] = (
    _ACCESSORY_DUPATTA_FAMILY,
    _ACCESSORY_BAG_FAMILY,
    _ACCESSORY_JEWELLERY_FAMILY,
    _ACCESSORY_BELT_WATCH_FAMILY,
    _ACCESSORY_EYEWEAR_CAP_FAMILY,
    _ACCESSORY_MENSWEAR_FORMAL_FAMILY,
)

# Union of every family — used by classify_item() to detect "this candidate IS
# an accessory of some kind" before checking WHICH family it belongs to.
ACCESSORY_KEYWORDS: frozenset[str] = frozenset().union(*_ACCESSORY_FAMILIES)

# Sub-families that are genuinely unisex in this catalogue (sunglasses, belts,
# watches, caps) — used ONLY as a narrow opt-in fallback when a gendered
# accessory search returns nothing (see is_gender_neutral_accessory below).
_GENDER_NEUTRAL_ACCESSORY_FAMILIES: tuple[frozenset[str], ...] = (
    _ACCESSORY_EYEWEAR_CAP_FAMILY,
    _ACCESSORY_BELT_WATCH_FAMILY,
)

# Western marker words for classes classify_anchor() never flags as "western"
# (footwear/outerwear/unknown) — e.g. is_western_item("Sneakers") is False
# because classify_anchor("Sneakers") returns "footwear", not one of the three
# western_* classes. Used ONLY by ethnic-occasion coherence gates so a sangeet
# look can never accept a pair of sneakers or a denim jacket into the
# footwear/outerwear slot (see is_western_marker_item + coherence.py).
_WESTERN_MARKER_KEYWORDS: frozenset[str] = frozenset(
    {"sneaker", "sneakers", "denim", "bomber", "hoodie", "blazer", "t-shirt", "tshirt"}
)

# Small, conservative novelty/costume denylist (Phase B Part 1 quality guard).
# "cosplay"/"costume" are checked as bare substrings (no real-catalogue false
# positives found — see offline check). The instrument/object words are only
# treated as novelty when paired with "shape"/"shaped" in the same name, so a
# legitimate "V-shape Waist Jegging" is NOT rejected (it contains no object
# word), while "Luxury Piano Shape Statement Handbag" IS rejected.
_NOVELTY_GENERAL_MARKERS: frozenset[str] = frozenset({"cosplay", "costume"})
_NOVELTY_OBJECT_WORDS: frozenset[str] = frozenset(
    {
        "piano", "guitar", "violin", "football", "rhino", "puppy", "dachshund",
        "unicorn", "flamingo", "telephone",
    }
)


def _contains_word(text: str, phrase: str) -> bool:
    """Word-boundary substring match — plain `phrase in text` would let short
    accessory keywords like "bag"/"cap" false-positive inside unrelated words
    (e.g. "Paperbag Waist Pants" contains "bag"; "Capri" contains "cap").  This
    was the live-proven root cause of a pair of trousers filling an
    "accessory" slot — the slot's "bag handbag" query text-matched "paperbag".
    """
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


# 2026-07-30: product_type_name facet VALUES that are themselves generic
# catch-all buckets (no real type information on their own — see
# classify_item's shortcut-skip comment below for the mechanism this
# protects against). Exactly the bare/near-bare pt-alone keywords added by
# the unknown-class keyword-coverage audit that could plausibly co-occur
# with a MORE SPECIFIC, earlier-priority keyword (ethnic/men_formalwear)
# elsewhere in a real item's own name.
#
# 2026-08-05 (indowestern first-class fix): "indowestern" added for a
# DIFFERENT reason than the rest of this set -- the facet value itself is
# informative (see classify_anchor's own indowestern short-circuit below),
# not a generic bucket. It's here purely so classify_item()'s pt-alone
# shortcut doesn't resolve straight from the facet value before the
# combined-text accessory check (step 3) gets a chance to run -- a handful
# of catalogue rows are jewellery mislabeled with product_type_name==
# "indowestern" (e.g. "Multicoloured Gemstone Indo Western Necklace Set"),
# and that combined-text accessory check is what correctly resolves them to
# "accessory" instead of the ethnic_one_piece garment class.
_GENERIC_FACET_VALUES: frozenset[str] = frozenset({
    "outerwear", "footwear", "knitwear", "coord", "bottom", "bottoms",
    "bottomwear", "tracksuit", "track suit", "indowestern",
})

# 2026-08-06 (full accessory-vocabulary audit): product_type_name facet
# VALUES that are themselves genuinely multi-family accessory catch-alls —
# their own name text spans several ACCESSORY_KEYWORDS families at once
# (e.g. "Clothing Accessories" sampled as socks + caps + cufflinks + tie
# pins + handkerchiefs in the SAME facet; "Accessories" as chains + maang
# tikkas + rings + armlets + a sindoor box), so no single keyword or family
# addition could close them — a facet-EQUALITY check on the retailer's own
# already-accessory-labelled facet is the correct mechanism (this is the
# fix a prior audit pass explicitly flagged and deferred — see the
# jewellery family's own "Deliberately NOT 'accessories'/'accessory'"
# comment above for why a SUBSTRING addition would have been wrong: it
# would corrupt split_accessory_query_by_family()'s single-vs-multi-family
# detection for every query that happens to carry the word "accessory" as
# filler text. A facet-equality check on product_type_name has no such
# side effect — it never touches ACCESSORY_KEYWORDS or the family-query
# matching path at all, only this one function's own class resolution).
# "Neklace"/"Pandent"/"Chains"/"Lacha"/"Shawls"/"Muffler" are exact-facet
# TYPOS or singleton facet values (1-5 rows each) with the same "no safe
# keyword generalisation, but the facet itself is unambiguous" shape.
# "Muffler" specifically was tried as a combined-text keyword first and
# REVERTED (2026-08-06 same-day fix): it collided with 8 outerwear rows
# that BUNDLE a muffler with a coat/jacket ("MLS COAT WITH MUFFLER 1PC",
# "Detachable Muffler Wool-Blend Coat") — the same "Crop Top WITH Palazzo"
# bundle-listing shape classify_item's own pt-alone shortcut exists to
# protect against, except outerwear is a _GENERIC_FACET_VALUES entry so
# that shortcut doesn't apply and the combined-text scan saw the bundled
# item's name too. Facet equality has no such risk — it only ever checks
# product_type_name itself, never the freeform name a bundle might mention
# a second item in.
_ACCESSORY_FACET_VALUES: frozenset[str] = frozenset({
    "clothing accessories", "accessories", "neklace", "pandent", "chains",
    "lacha", "shawls", "socks", "muffler", "mufflers",
})


def classify_item(product_type: str, prod_name: str = "") -> str:
    """Classify a CANDIDATE item (not just an anchor) into a slot-compatible class.

    Same classes as classify_anchor(), PLUS "accessory" for bags/belts/watches/
    dupattas/jewellery/etc.  classify_anchor() never returns "accessory" because
    in this catalogue an accessory is never used as a look ANCHOR; this sibling
    function is used for candidate-side slot-type gating in
    composer._find_best_candidate, where accessory candidates DO occur.

    Trusts the catalogue's own `product_type` facet FIRST (checked ALONE, with
    the freeform name blanked out), falling back to the combined product_type +
    name keyword scan (classify_anchor's original behaviour) only when
    product_type alone doesn't resolve to a known class.  This matters because
    many real listings are co-ord/bundle sets whose freeform NAME mentions
    other garment parts too — e.g. product_type="top" but name "... Crop Top
    WITH PALAZZO", or product_type="kurta" but name "... Kurta WITH TROUSERS &
    DUPATTA".  Scanning the combined text let those name-only mentions
    ("palazzo"/"trousers") override the authoritative product_type and put a
    top-typed item in a "bottom" slot — a live-proven variant of the same class
    of bug as the "Paperbag Waist Pants" substring collision, generalised
    beyond "bag"-in-text collisions to "other-garment-word"-in-text collisions.

    Uses word-boundary matching (see _contains_word) so "Paperbag Waist Pants"
    is never misclassified as an accessory just because "bag" is a substring.
    """
    pt = product_type.lower()
    name = prod_name.lower()

    if pt.strip() in _ACCESSORY_FACET_VALUES:
        return "accessory"
    if any(_contains_word(pt, kw) for kw in ACCESSORY_KEYWORDS):
        return "accessory"
    # 2026-07-30 fix (live-proven regression, test_price_outlier_guard.py's
    # eid/men budgeted-look test): the pt-alone shortcut below is safe ONLY
    # when `product_type` is itself a SPECIFIC facet value (top/trousers/
    # dress/shirt/...) — that's the "Crop Top WITH Palazzo" case this
    # shortcut was built for (pt="top" is already the correct, authoritative
    # answer; "palazzo" in the name describes a BUNDLED second item, not
    # this item's own type). It breaks when `product_type` is one of the
    # GENERIC catch-all facet buckets added by the 2026-07-30 unknown-class
    # keyword-coverage audit (outerwear/footwear/knitwear/coord/bottom/
    # bottoms/bottomwear/tracksuit/track suit) — these facet values carry NO
    # real type information on their own (that's WHY they used to resolve
    # "unknown"), so pt-alone resolving via one of THEIR bare keywords must
    # never short-circuit past a MORE SPECIFIC keyword sitting in the item's
    # own full name. Live-proven: product_type=="outerwear",
    # prod_name=="Cream Golden Floral Nehru Jacket" — pt-alone now resolves
    # "outerwear" (bare keyword), silently skipping the MEN_FORMALWEAR_
    # KEYWORDS "nehru jacket" match that classify_anchor(pt, name) [full
    # text] correctly gives ("men_formalwear") — SLOT_ALLOWED_CLASSES
    # doesn't accept men_formalwear for an "outerwear" slot, so this
    # masqueraded the item into eligibility it should never have had,
    # silently changing which candidate won a real composed look.
    _pt_stripped = pt.strip()
    if _pt_stripped not in _GENERIC_FACET_VALUES:
        pt_only_class = classify_anchor(product_type, "")
        if pt_only_class != "unknown":
            return pt_only_class

    combined = pt + " " + name
    if any(_contains_word(combined, kw) for kw in ACCESSORY_KEYWORDS):
        return "accessory"
    return classify_anchor(product_type, prod_name)


# Slot name -> the set of classify_item() classes allowed to fill it.  A
# candidate whose class is not in this set is rejected before scoring —
# this is what makes it impossible for a bottom-classified item (e.g. those
# paperbag-waist trousers) to ever fill the "accessory" slot, regardless of
# what the retrieval layer's text/embedding similarity happened to surface.
# The five sets are pairwise disjoint by construction (see
# TestSlotAllowedClassesDisjoint in tests/test_outfit_package.py).
SLOT_ALLOWED_CLASSES: dict[str, frozenset[str]] = {
    "top": frozenset({"western_top", "ethnic_top"}),
    "bottom": frozenset({"western_bottom", "ethnic_bottom"}),
    "footwear": frozenset({"footwear"}),
    "outerwear": frozenset({"outerwear"}),
    "accessory": frozenset({"accessory"}),
}


# classify_item() classes that can never stand alone as "an outfit" for a
# generic "outfit/look/wear" ask (see graph.py's + eval_strict.py's
# accessory-exclusion gate, both of which import this) — a lone shoe fails
# that bar the same way a lone bag does (2026-07-30 fix: "footwear" is its
# own disjoint class from "accessory" in SLOT_ALLOWED_CLASSES above, so the
# gate's original `!= "accessory"` check never caught a standalone footwear
# item like "Ladies Triveni Kolhapuri Chappal" ranking for "haldi look").
NON_OUTFIT_ITEM_CLASSES: frozenset[str] = frozenset({"accessory", "footwear"})


def is_slot_type_allowed(slot_name: str, product_type: str, prod_name: str = "") -> bool:
    """Hard slot-type gate: reject a candidate whose classified item-type doesn't
    belong to the slot it's being considered for.

    Unknown slot names (should never occur — every SlotSpec.slot_name is one of
    the five keys in SLOT_ALLOWED_CLASSES) fall back to permissive True rather
    than silently rejecting everything.
    """
    allowed = SLOT_ALLOWED_CLASSES.get(slot_name)
    if allowed is None:
        return True
    return classify_item(product_type, prod_name) in allowed


def accessory_query_matches(query: str, product_type: str, prod_name: str) -> bool:
    """Return True if an accessory candidate's text shares a FAMILY with the
    slot query (e.g. a "dupatta ethnic dupatta" query must not accept a
    handbag, and a "belt watch cap" query must not accept a dupatta).

    Permissive (returns True) when the query doesn't recognisably target one of
    the known accessory families — avoids over-rejecting queries not covered
    by this list.  Every accessory SlotSpec.search_query in get_fill_slots()
    below does map onto at least one family (see offline check).
    """
    q = query.lower()
    combined = (product_type + " " + prod_name).lower()
    matched_families = [
        fam for fam in _ACCESSORY_FAMILIES if any(_contains_word(q, kw) for kw in fam)
    ]
    if not matched_families:
        return True
    return any(any(_contains_word(combined, kw) for kw in fam) for fam in matched_families)


def split_accessory_query_by_family(query: str) -> list[str]:
    """Split a multi-family accessory SlotSpec.search_query into one retrieval
    sub-query per accessory family it spans, each retaining that family's own
    matched keyword(s) plus every non-family (register/occasion/gender) word
    from the original query.

    Root-cause fix (2026-07-19, bridal-look jewellery gap): a single accessory
    slot's search_query frequently mixes multiple families in one string —
    e.g. the bridal ethnic_one_piece slot's "dupatta jewellery clutch ethnic
    accessory" spans the DUPATTA + BAG + JEWELLERY families at once.
    Retrieving that combined text as ONE query lets whichever family's
    vocabulary has the strongest lexical/semantic overlap with the query text
    dominate the retrieval window so completely that the other families'
    candidates never enter even a very wide pool — live-proven: the combined
    bridal accessory query returned ZERO jewellery items in the top 500 dense
    hits and top 300 sparse hits against the real unified catalogue, despite
    18,796 real jewellery rows existing and the literal word "jewellery"
    being IN the query (dupatta/clutch listings very often literally repeat
    "ethnic"/"embroidered"/"accessory" in their own catalogue text, so they
    win on lexical/embedding overlap alone). This is a retrieval-POOL defect,
    not a downstream gating defect — accessory_query_matches() already
    permits every one of these families through the slot-type gate; they
    just never arrive as candidates in the first place.

    Splitting into one sub-query per matched family (verified offline: an
    isolated "jewellery ethnic accessory festive embroidered" query surfaces
    18/40 genuine jewellery hits where the combined query surfaced 0/40) and
    merging the resulting pools before scoring gives every family a genuine,
    undiluted retrieval shot. The existing scoring formula in
    _score_candidates (colour/fabric/body-type deltas, occasion register)
    still picks the actual winner — jewellery is never forced to win, only
    given a fair chance to compete.

    Returns an empty list (signalling "no split needed — use the query
    text as-is") when the query matches 0 or 1 accessory family — every
    existing single-family accessory SlotSpec (e.g. "dupatta ethnic dupatta",
    "pocket square safa ethnic accessory") is therefore fully unaffected by
    this function; only genuinely multi-family queries are split.
    """
    q = query.lower()
    matched: list[tuple[frozenset[str], list[str]]] = []
    residual = q
    for family in _ACCESSORY_FAMILIES:
        words_present = [kw for kw in family if _contains_word(q, kw)]
        if words_present:
            matched.append((family, words_present))
            for w in words_present:
                residual = re.sub(rf"\b{re.escape(w)}\b", " ", residual)

    if len(matched) <= 1:
        return []

    residual = re.sub(r"\s+", " ", residual).strip()
    return [
        " ".join(words) + (f" {residual}" if residual else "") for _, words in matched
    ]


def is_gender_neutral_accessory(product_type: str, prod_name: str = "") -> bool:
    """Return True for accessory sub-types that are genuinely unisex in this
    catalogue (sunglasses, belts, watches, caps).

    Used ONLY as a narrow, opt-in fallback in composer._find_best_candidate
    when a slot's gendered search returns zero results — never for garments
    (tops, bottoms, footwear, outerwear), where gender ambiguity is a leak to
    close, not a feature to exploit.
    """
    combined = (product_type + " " + prod_name).lower()
    return any(
        any(_contains_word(combined, kw) for kw in fam)
        for fam in _GENDER_NEUTRAL_ACCESSORY_FAMILIES
    )


# A bag-family word alongside an object/instrument/animal word is the second
# (in addition to "shape"/"shaped") signal that a candidate is a novelty item —
# catches real catalogue rows like "Designer Dachshund Crossbody Bag" that
# don't happen to use the literal word "shape".  Checked against the real
# catalogue: "football" (49 rows, all "Football Shoes"/"Football Shorts") and
# "flamingo" (colour-name rows, "Flamingo Pink ... Shirt") never co-occur with
# any of these bag words, so this AND-combination has zero false positives
# there (see offline audit).
_NOVELTY_BAG_WORDS: frozenset[str] = frozenset(
    {"bag", "handbag", "clutch", "crossbody", "tote", "purse"}
)


def is_novelty_item(prod_name: str) -> bool:
    """Reject conservative novelty/costume items that should never fill a real
    outfit slot (e.g. "Luxury Piano Shape Statement Handbag", "Designer
    Dachshund Crossbody Bag").

    Small, deliberately conservative denylist — false negatives (a novelty item
    that slips through) are safer than false positives that reject real
    garments.  Checked against the real catalogue: no "V-shape Waist Jegging"-
    style false positive (no object word present), no "Novelty Town" brand-name
    false positive (that brand word is not in this denylist at all), and no
    "Football Shoes"/"Flamingo Pink Shirt" false positive (object word present
    but no bag-family word alongside it).
    """
    name = (prod_name or "").lower()
    if any(_contains_word(name, kw) for kw in _NOVELTY_GENERAL_MARKERS):
        return True
    has_object_word = any(_contains_word(name, kw) for kw in _NOVELTY_OBJECT_WORDS)
    if has_object_word and (
        "shape" in name
        or "shaped" in name
        or any(_contains_word(name, kw) for kw in _NOVELTY_BAG_WORDS)
    ):
        return True
    return False


# is_kids_item / _KIDS_MARKER_RE promoted to src.catalogue.cleaning (2026-07-12)
# so the retrieval layer (hybrid_search.py) and plain-search node (graph.py) can
# apply it as an unconditional hard exclusion without importing from the agents
# layer. Import from there — see that module's docstring for the S5 fix history.

# Phase B: an adult, correctly-gendered item can still be casual-register and
# leak into a formal/office look — _default_bottom_query() drops "jeans"/
# "skirt" from the QUERY text for formality>=3 occasions, but that only
# shapes retrieval ranking; it does not stop a casual item from being
# retrieved via other query terms (register tokens, anchor colour, etc.) and
# then accepted as the best-scoring candidate. Live-proven: "black top for
# office for women" -> Style this -> bottom slot filled with "ONLY Women Blue
# Solid Denim Mini Skirts" (adult item, correctly gendered, NOT caught by
# is_kids_item). Deliberately narrow, word-boundary keyword list — conservative
# false-negatives (a casual item slipping through on a word not in this list)
# are safer than false positives that reject legitimate formal items. "denim"
# is checked standalone (not just "denim skirt"/"denim jeans") per design: a
# hypothetical "denim-look tailored trouser" is still rejected — keeping the
# rule simple beats trying to carve out fabric-look exceptions.
#
# 2026-07-30 extension (reception/wedding_guest western-formal-capable-type
# fix): the bare word "casual" is now ALSO a casual marker on its own — real
# catalogue rows like "Grey Solid Uno Fit Casual Trouser"/"Casual Shirt"/
# "Casual Blazer" were live-proven to pass the OLD name-substring formal-
# western allow-list in coherence.py purely because their product_type facet
# happened to literally equal the allow-list keyword, with zero actual
# formality check on the SPECIFIC item. Deliberately excludes "smart casual"/
# "semi casual" via negative lookbehind — catalogue-audited: 467 "semi
# casual" + 6 "smart casual" rows (e.g. "Black Solid Smart Fit Semi Casual
# Shirt") are a real, distinct business-casual register in this catalogue,
# not fully casual, and must stay admissible for formal-register looks.
_CASUAL_MARKER_RE = re.compile(
    r"\b(denim|jeans|mini\s+skirts?|shorts?|joggers?|cargo|distressed|ripped)\b"
    r"|(?<!smart )(?<!smart-)(?<!semi )(?<!semi-)\bcasual\b",
    re.IGNORECASE,
)


def is_casual_marker_item(prod_name: str) -> bool:
    """Return True if `prod_name` carries a casual/denim-register marker word.

    Checked as an ADDITIONAL gate in composer._find_best_candidate for
    formal occasions (occasion.formality >= 3), alongside (never instead of)
    the gender/slot-type/novelty/kids gates — see module docstring on
    _CASUAL_MARKER_RE for the live regression this closes.
    """
    return bool(_CASUAL_MARKER_RE.search(prod_name or ""))


# Rugged/athletic footwear register — live-proven miss (sweep 2026-07-10,
# relevance-adjacent): a sangeet "his look" footwear slot filled with ₹759
# combat boots. The generic casual-marker gate above covers garments (denim/
# cargo/joggers) but had no footwear vocabulary at all. Plain "boots" is
# included deliberately: for Indian festive/formal occasions the appropriate
# menswear registers are oxfords/derbies/loafers/monks/mojaris/juttis — even a
# dress Chelsea boot is an edge case not worth the combat-boot false accepts.
_RUGGED_FOOTWEAR_RE = re.compile(
    r"\b(boots?|sneakers?|trainers?|running|walking|sports?|training|trekking|hiking|"
    r"football|badminton|gym|flip[\s-]?flops?|slippers?|sliders?|crocs?|clogs?)\b",
    re.IGNORECASE,
)


def is_rugged_footwear_item(prod_name: str) -> bool:
    """True if `prod_name` reads as rugged/athletic/at-home footwear — never
    acceptable in a formality >= 3 look's footwear slot."""
    return bool(_RUGGED_FOOTWEAR_RE.search(prod_name or ""))


# Wave 9 (2026-07-23, gym occasion): the INVERSE selection to
# _RUGGED_FOOTWEAR_RE above — a gym look's footwear slot must ONLY accept a
# genuine athletic/sport shoe, never a jutti/mojari/formal oxford/wedding
# heel as a fallback. Deliberately excludes flip-flops/slippers/sliders/
# crocs/clogs (at-home, not gym-appropriate) that _RUGGED_FOOTWEAR_RE lumps
# in with genuine athletic wear. Catalogue audit against the real unified
# catalogue (data/processed/unified/catalogue.parquet): 0 women's rows and 20
# men's rows (all store=flipkart, "Running Shoes"/"Sneakers"/"Training & Gym
# Shoes For Men") match this pattern — a women's gym look's footwear slot is
# therefore expected to go through honest suppression (composer.
# _suppression_reason) far more often than not; that is the correct honest
# behaviour for genuinely thin inventory, never a bug to paper over with a
# non-athletic substitute.
_ATHLETIC_FOOTWEAR_RE = re.compile(
    r"\b(sneakers?|trainers?|running|sports?|training|gym|athletic|workout)\b",
    re.IGNORECASE,
)


def is_athletic_footwear_item(prod_name: str) -> bool:
    """True if `prod_name` reads as a genuine athletic/sport shoe — the ONLY
    footwear register a gym look's footwear slot accepts (see coherence.py's
    athletic-register gate)."""
    return bool(_ATHLETIC_FOOTWEAR_RE.search(prod_name or ""))


# Phase B (product gap 2): a multi-piece SET listing ("Anarkali Sharara Set",
# "Kurta Set with Dupatta", a "Co-Ord Set") is a WHOLE OUTFIT, not a single
# garment — it must never fill a single complement slot (bottom/top/
# outerwear/accessory/etc.), even though it may still be used as a look's own
# SEED/anchor (a kurta set as a look's hero item is fine — compose_outfit's
# seed resolution never calls composer._find_best_candidate, so this gate
# never touches the seed).
#
# Live-proven root cause: product_type_name="sharara" alone (the catalogue's
# OWN facet) classify_item()'s pt-first shortcut resolves straight to
# "ethnic_bottom" WITHOUT ever inspecting the freeform name — so a
# "Quirky Floral Printed Cotton Anarkali Sharara Set" (a 2-piece anarkali top
# + sharara bottom SET) slipped past the bottom slot's hard slot-type gate
# and filled an "office look" bottom slot.
#
# Signals, any sufficient (all verified against the real unified catalogue's
# own "Set"-in-name product rows, and against every "set-not-single" hand
# label in eval/fixtures/strict_gold_labels.yaml — see offline check):
#   1. product_type_name is one of the catalogue's own dedicated set-type
#      values: "Suits"/"Suit Set(s)" (2-3 piece ethnic suit sets: kurta +
#      bottom [+ dupatta]), "coord"/"Co-Ord" (western co-ord sets),
#      "Sets"/"Track-Suit" (misc matching sets e.g. "Winter Set", "Cord Set").
#   2. Freeform name contains "with" AND mentions >= 2 DISTINCT garment nouns
#      (e.g. "Kaftan Kurta with Abstract Patchwork Palazzo") — this
#      catalogue's dominant multi-piece convention, and very often never uses
#      the literal word "set"/"sets" at all (2026-07-11 fix).
#   3. Freeform name contains the word "set(s)" (and is NOT a "Set of N"
#      same-item PACK — e.g. "TAG 7 Women Set of 2 ... Palazzos", a 2-pack of
#      the SAME garment, correctly excluded) AND mentions >= 1 garment noun.
#      2026-07-25 fix: originally required >=2 distinct nouns here too (the
#      same bar as "with"), but that was over-conservative and caused 16
#      real strict-gold misses — this catalogue's single most common
#      multi-piece convention is a bare "<Garment> Set" suffix that never
#      names the second piece in the product NAME at all (e.g. "Orange
#      Floral Printed Cotton Straight Kurta Set", "Plus Size Pink Printed
#      Cotton Straight Kurta Set") — the word "Set" itself is already the
#      strong signal once the "Set of N" pack pattern is excluded.
#   4. Freeform name contains an explicit "N-Piece"/"N Piece" count (e.g.
#      "Sea Green Winter Ethnic 3-Piece Set") — sufficient on its own, since
#      some of these name NO garment noun at all in the truncated name.
#   5. Freeform name contains the literal word "co-ord(s)" — inherently a
#      2-piece-by-definition term, sufficient on its own. Needed because the
#      product_type_name facet often captures only the hero piece (e.g.
#      "URBANIC...Hooded Co-Ords Set" carries product_type_name="top", never
#      matching signal 1's "coord" facet check).
#   6. A recognised TOP noun immediately followed by a recognised BOTTOM/
#      companion noun with NO connector word at all (e.g. "Green Kurta
#      Pajama", "kurta pajama"/"kurta pyjama" alone account for 2,459
#      catalogue rows) — this catalogue's third naming convention. Scoped
#      NARROWLY to TOP-then-BOTTOM specifically (not the full distinct-noun
#      union signals 2/3 use) to avoid false-firing on "Anarkali Kurta" — a
#      ONE_PIECE+TOP synonym pair naming ONE single garment, not two.
_SET_PRODUCT_TYPES: frozenset[str] = frozenset({
    "suits", "suit set", "suit sets", "coord", "co-ord", "sets", "track-suit",
})

_SET_WORD_RE = re.compile(r"\bsets?\b", re.IGNORECASE)
_SET_OF_N_RE = re.compile(r"\bset\s+of\s+\d+\b", re.IGNORECASE)
_N_PIECE_RE = re.compile(r"\b\d+[\s-]?piece\b", re.IGNORECASE)
_COORD_WORD_RE = re.compile(r"\bco-?ords?\b", re.IGNORECASE)
_TOP_THEN_BOTTOM_RE = re.compile(
    r"\b(?:kurta|kurti|kameez|tunic|shirt|top|blouse)\s+"
    r"(?:pajama|pyjama|palazzos?|churidar|salwar|sharara|dhoti|trousers?|pants?|dupatta)\b",
    re.IGNORECASE,
)

# Garment-noun vocabulary reused from the anchor-classification keyword sets
# above — used ONLY to count how many distinct garment types a name mentions.
_SET_GARMENT_NOUN_KEYWORDS: frozenset[str] = (
    ETHNIC_TOP_KEYWORDS
    | ETHNIC_ONE_PIECE_KEYWORDS
    | ETHNIC_BOTTOM_KEYWORDS
    | WESTERN_TOP_KEYWORDS
    | WESTERN_BOTTOM_KEYWORDS
    | WESTERN_ONE_PIECE_KEYWORDS
    | OUTERWEAR_KEYWORDS
    | frozenset({"dupatta"})
)

_WITH_RE = re.compile(r"\bwith\b", re.IGNORECASE)


def is_multi_piece_set(product_type: str, prod_name: str) -> bool:
    """Return True if this item is a multi-piece SET listing (a whole outfit)
    rather than a single garment. See the module comment above
    _SET_PRODUCT_TYPES for the six signals checked, any one sufficient.
    """
    pt = (product_type or "").lower().strip()
    if pt in _SET_PRODUCT_TYPES:
        return True
    name = (prod_name or "").lower()
    if _N_PIECE_RE.search(name) or _COORD_WORD_RE.search(name) or _TOP_THEN_BOTTOM_RE.search(name):
        return True
    has_set = _SET_WORD_RE.search(name) and not _SET_OF_N_RE.search(name)
    has_with = _WITH_RE.search(name)
    if not (has_set or has_with):
        return False
    distinct_nouns = {kw for kw in _SET_GARMENT_NOUN_KEYWORDS if _contains_word(name, kw)}
    if has_set:
        return len(distinct_nouns) >= 1
    return len(distinct_nouns) >= 2


# 2026-07-25 (strict-eval attribute-contradiction follow-up, now the largest
# CODE-FIXABLE miss bucket at 22): no deterministic fit/silhouette-matching
# gate existed anywhere in the plain-search path before this — retrieval
# relies entirely on embedding similarity, which frequently ranks a "Slim
# Fit" item highly for a "straight fit" query since the two phrases are
# semantically close in embedding space despite being product-listing
# OPPOSITES in this catalogue's own marketing vocabulary.
#
# Two representations, chosen per group based on whether the group's members
# are genuinely N-way mutually exclusive or actually two *camps* of near-
# synonyms opposing each other:
#
# - Flat groups (_ATTRIBUTE_CONTRADICTION_FLAT_GROUPS): any two DIFFERENT
#   members oppose each other. Correct only when every member is genuinely
#   distinct from every other (RISE, BREASTED, NECKLINE) — a garment has
#   exactly one neckline shape, one rise, one breasted style.
# - Camp-pair groups (_ATTRIBUTE_CONTRADICTION_CAMP_PAIRS): two frozensets
#   per pair; only CROSS-camp words oppose, same-camp words are compatible
#   synonyms. Required for FIT and SILHOUETTE, where a flat group produced a
#   real false positive: "anarkali" and "a-line" were both dumped into one
#   "silhouette" group as if mutually exclusive, but an anarkali kurta IS
#   a-line by definition (verified against real catalogue desc text, e.g.
#   article 7797797454046 "...heritage anarkali style with a graceful
#   flared silhouette..." — anarkali literally means flared/a-line here).
#   The genuine opposition is flared-family vs fitted/straight-family.
#
# Grounded in the ACTUAL contradiction pairs found across every
# attribute-contradiction hand label in strict_gold_labels.yaml (not
# invented) — verified real hit-rate per word against
# data/processed/unified/catalogue.parquet before inclusion (all >=12 rows,
# most in the hundreds-to-thousands).
_ATTRIBUTE_CONTRADICTION_FLAT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"high waisted", "high-waisted", "high rise", "mid rise", "low rise"}),
    frozenset({"single breasted", "double breasted"}),
    frozenset({
        "v-neck", "halter neck", "boat neck", "round neck", "square neck",
        "mandarin collar", "scoop neck", "sweetheart neck",
    }),
)

_ATTRIBUTE_CONTRADICTION_CAMP_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    # FIT tightness: "slim"/"skinny" (tight camp) vs the mutually-compatible
    # "straight"/"regular"/"relaxed"/"oversized"/"loose"/"tailored" camp.
    (
        frozenset({"slim fit", "skinny fit"}),
        frozenset({
            "straight fit", "regular fit", "relaxed fit", "tailored fit",
            "oversized", "loose fit",
        }),
    ),
    # SILHOUETTE flare: "a-line"/"anarkali"/"fit and flare" are mutually
    # COMPATIBLE (same flared family — anarkali kurtas are a-line by
    # definition), opposing the straight/regular camp. "regular fit" belongs
    # here too (distinct dimension from FIT-tightness above) — real hand
    # label evidence: "'Regular Fit' contradicts 'a-line'" (a kurta's
    # overall cut is either flared/a-line or straight/regular, not both).
    # "straight shape" added 2026-08-07 (compose-wave miss audit) — a real
    # miss ("Straight shape with regular style" desc for an "a-line kurta"
    # query) used this exact phrasing instead of "straight cut"/"straight
    # fit". Catalogue-audited before adding: "straight shape" co-occurs with
    # any flare-camp word (a-line/anarkali/fit and flare) in 0/770 rows —
    # clean, safe to treat as straight-family. ("regular style", the OTHER
    # phrase in that same miss's desc, was NOT added: it co-occurs with a
    # flare-camp word in 171/820 rows — not a safe opposite, would have
    # introduced real false positives, so left out.)
    (
        frozenset({"a-line", "anarkali", "fit and flare", "fit & flare"}),
        frozenset({"straight cut", "straight fit", "regular fit", "straight shape"}),
    ),
    # SILHOUETTE fitted: "bodycon" is its OWN pole (2026-07-25, out-of-sample
    # validation finding), not folded into the straight/regular camp above —
    # an item with the structured facet line "Silhouette: Straight kurta"
    # surfaced for a "bodycon kurta" query in the held-out set. Bodycon
    # (body-hugging throughout) and straight (hangs straight, unflared but
    # NOT tight) are genuinely distinct silhouettes for a kurta, even though
    # both oppose the SAME flared camp above — treating them as compatible
    # with each other was too coarse. Opposes both other poles.
    #
    # "silhouette: straight" (the structured facet-line phrasing, not bare
    # "straight kurta" prose) is deliberately the ONLY straight-family
    # trigger added here — a bare "straight kurta"/"straight kurti" phrase
    # appears in 2252/14184 (16%) of catalogue kurta rows (the default,
    # most-common silhouette description for kurtas generally), which would
    # have been a badly over-broad exclusion for a single niche query
    # pattern; the structured "Silhouette: Straight" facet line is a much
    # narrower, catalogue-verified signal (12/14184 rows). "straight shape"
    # added alongside it 2026-08-07 for the same 0/770-clean reason as above.
    (
        frozenset({"bodycon"}),
        frozenset({"a-line", "anarkali", "fit and flare", "fit & flare"}),
    ),
    (
        frozenset({"bodycon"}),
        frozenset({
            "straight cut", "straight fit", "regular fit", "silhouette: straight",
            "straight shape",
        }),
    ),
    # SILHOUETTE wrap vs bodycon — 2026-08-07 (compose-wave miss audit): a
    # "wrap dress" query surfaced an item explicitly described as "ruched
    # bodycon dress". Catalogue-audited: "wrap" and "bodycon" co-occur in
    # only 4/477 rows mentioning "wrap" (0.8%) — clean enough to treat as
    # opposing silhouettes. Deliberately NOT opposed to the flare camp above
    # (a-line/anarkali) — a wrap silhouette is not reliably distinct from a
    # flared skirt the way it is from bodycon, and that pairing wasn't
    # audited, so it's left alone rather than guessed.
    (
        frozenset({"wrap"}),
        frozenset({"bodycon"}),
    ),
)


def _contains_phrase_flex(text: str, phrase: str) -> bool:
    """Like _contains_word, but a space inside a multi-word `phrase` also
    matches a hyphen in `text` (and vice versa) — real catalogue listings
    routinely hyphenate exactly these compound adjectives ("single-breasted",
    "straight-fit") while the tracked phrases below are written with spaces.
    Scoped to is_attribute_contradiction only, not the shared _contains_word
    (18 other call sites in this module use single-word keywords where the
    distinction doesn't apply — no reason to widen their matching too).
    2026-08-07 (compose-wave miss audit): found via 2 real misses ("single-
    breasted with button closures" not matching tracked "single breasted";
    "The straight-fit, full-sleeve design" not matching tracked "straight
    fit") — both confirmed fixed by this, zero new call sites touched.
    """
    pattern = re.escape(phrase).replace(r"\ ", r"[\s-]")
    return re.search(rf"\b{pattern}\b", text, re.IGNORECASE) is not None


def is_attribute_contradiction(query_text: str, item_name: str, item_desc: str) -> bool:
    """Return True if the CANDIDATE ITEM's own name/desc explicitly states a
    fit/rise/breasted-style/silhouette/neckline word that OPPOSES a word the
    QUERY itself explicitly stated — e.g. query "straight fit kurta" +
    item name "...Slim Fit Kurta".

    Deliberately conservative in both directions:
      - Only fires when the QUERY names a tracked word at all (a query with
        no stated fit/silhouette preference can never trigger this).
      - If the item's own text ALSO contains the query's exact word anywhere
        (even alongside an opposing word — e.g. boilerplate mentioning
        several fit options), that's treated as a genuine match, never a
        contradiction — same "explicit confirmation wins" precedent as the
        query's own hand-labeling rubric.
      - "Unstated, not contradicted" is never penalised here (mirrors the
        hand-labeling rubric's point 2) — an item with NO word from a group
        at all is never flagged for that group.
      - Near-synonyms (e.g. "anarkali" vs "a-line") never oppose each other
        — see _ATTRIBUTE_CONTRADICTION_CAMP_PAIRS.
      - Matching is hyphen/space-flexible (see _contains_phrase_flex) so
        "single-breasted" in real listing text matches tracked "single
        breasted", etc.
    """
    query_lower = (query_text or "").lower()
    text = f"{item_name or ''} {item_desc or ''}".lower()

    for group in _ATTRIBUTE_CONTRADICTION_FLAT_GROUPS:
        stated = next((w for w in group if _contains_phrase_flex(query_lower, w)), None)
        if stated is None:
            continue
        if _contains_phrase_flex(text, stated):
            continue  # item explicitly confirms the query's own word — never a contradiction
        if any(w != stated and _contains_phrase_flex(text, w) for w in group):
            return True

    for camp_a, camp_b in _ATTRIBUTE_CONTRADICTION_CAMP_PAIRS:
        for stated_camp, opposing_camp in ((camp_a, camp_b), (camp_b, camp_a)):
            stated = next((w for w in stated_camp if _contains_phrase_flex(query_lower, w)), None)
            if stated is None:
                continue
            if any(_contains_phrase_flex(text, w) for w in stated_camp):
                continue  # item confirms the query's own camp — never a contradiction
            if any(_contains_phrase_flex(text, w) for w in opposing_camp):
                return True

    return False


def is_western_marker_item(product_type: str, prod_name: str = "") -> bool:
    """Return True if a footwear/outerwear/unknown-class item carries an
    explicit WESTERN marker word (sneaker, denim, bomber, hoodie, blazer,
    t-shirt).  Layered ON TOP of is_western_item (which only covers
    western_top/bottom/one_piece) so an ethnic-only occasion (e.g. sangeet)
    can never accept a pair of sneakers or a denim jacket via the
    footwear/outerwear slot — used only by coherence.py's ethnic gates.
    """
    combined = (product_type + " " + prod_name).lower()
    return any(_contains_word(combined, kw) for kw in _WESTERN_MARKER_KEYWORDS)


def is_formal_capable_western_item(product_type: str, prod_name: str = "") -> bool:
    """Return True if `product_type`'s FACET VALUE (exact match, not a
    substring scan of free text) names a Western garment type capable of
    formal/glam register — the structural signal that replaces the old
    name-substring "gown"/"blazer"/"formal" allow-list scan in
    coherence.py's reception/wedding_guest exception. See
    _WESTERN_FORMAL_CAPABLE_TYPES for the catalogue audit behind this set.
    Always pair with is_casual_marker_item(prod_name) — this only answers
    "is this TYPE formal-capable", not "is this SPECIFIC item casual-
    register despite its type" (a "Casual Trouser" is still a trouser).
    """
    return product_type.lower().strip() in _WESTERN_FORMAL_CAPABLE_TYPES


def resolve_look_gender(
    *,
    intent_gender: str | None,
    session_gender: str | None,
    catalogue_df: pd.DataFrame,
    anchor_id: str | None,
    brand_gender_default: str,
) -> str:
    """Resolve which gender ("men" | "women") to steer a look composition with.

    Precedence (first concrete "men"/"women" signal wins):
      1. intent_gender — explicit gender parsed from the user's own text this turn.
      2. session_gender — gender context carried over from prior turns in the
         same conversation (e.g. a previous "men's shirts" search set filters).
      3. The anchor item's own catalogue `gender` column, when it resolves to
         "men" or "women" — critical for the image-upload owned-anchor path: a
         photo of a men's garment must never silently compose a women's-default
         look just because the brand's configured default happens to be
         "women". Shared by api/routes/image_style.py and graph.py's
         outfit_node so both paths resolve gender identically.
      4. brand_gender_default (config-level fallback; "mixed"/anything else
         coerces to "women" as the least-committal default — never guessed
         per-item).

    Never returns anything other than "men"/"women" — composition always needs
    a concrete slice; "unknown" is never returned here (see gender_allowed for
    how per-item "unknown" rows are excluded downstream).
    """
    if intent_gender in ("men", "women"):
        return intent_gender
    if session_gender in ("men", "women"):
        return session_gender
    if anchor_id is not None and "gender" in catalogue_df.columns and "article_id" in catalogue_df.columns:
        match = catalogue_df.loc[catalogue_df["article_id"] == anchor_id, "gender"]
        if not match.empty and match.iloc[0] is not None:
            anchor_gender = str(match.iloc[0]).lower()
            if anchor_gender in ("men", "women"):
                return anchor_gender
    resolved_default = brand_gender_default or "women"
    return "women" if resolved_default not in ("men", "women") else resolved_default

# Occasions where footwear is required (formality >= 3, ethnic)
# Wave 8: diwali/navratri/karva_chauth/eid added (all formality >= 3, ethnic
# events warrant required footwear). raksha_bandhan deliberately excluded —
# formality 2/EITHER, casual-festive register, footwear stays optional like
# casual/smart_casual/party_evening.
_FORMAL_ETHNIC_OCCASIONS: frozenset[str] = frozenset({
    "sangeet", "haldi", "mehendi", "festive_puja", "wedding_guest",
    "traditional_ethnic", "reception", "engagement",
    "diwali", "navratri", "karva_chauth", "eid",
})

# Women-only ethnic categories — hard reject for men's looks regardless of gender field
WOMEN_ONLY_ETHNIC_KEYWORDS: frozenset[str] = frozenset({
    "dupatta", "saree", "lehenga",
})

# Fabric/embellishment keywords for haldi/mehendi vs sangeet/reception scoring
SANGEET_EMBELLISHMENT_KEYWORDS: frozenset[str] = frozenset({
    "embroidered", "embroidery", "sequin", "zari", "embellished",
    "heavy work", "bridal", "mirror work", "thread work", "beaded",
    "resham", "gota", "kundan",
})
HALDI_LIGHTWEIGHT_KEYWORDS: frozenset[str] = frozenset({
    "cotton", "floral", "tie-dye", "georgette", "chiffon", "printed",
    "casual", "lightweight", "summer", "marigold", "yellow", "daisy",
})

# formality_softener override values (the sibling intent-parser field a
# different fix surfaces from queries like "something comfortable for
# sangeet dancing" or "not too flashy"). Reuses SANGEET_EMBELLISHMENT_KEYWORDS/
# HALDI_LIGHTWEIGHT_KEYWORDS verbatim rather than duplicating them — the
# override's "embellished vs plain/lightweight" text signal is semantically
# identical to sangeet/haldi's existing keyword scan, only the SIGN differs.
FORMALITY_SOFTENER_VALUES: frozenset[str] = frozenset({"minimalist", "comfortable"})


# 2026-07-30: brand-name/ethnic-keyword collision fix. classify_anchor()
# scans the full product_type+name text for keyword matches -- a handful
# of real catalogue BRAND names happen to literally contain an unrelated
# garment keyword, which silently overrides the item's real garment type.
# Confirmed via full-catalogue audit (product_type_name facet vs. combined
# classify_anchor(pt, name) result, keyword match position in the first
# 1-2 words of prod_name): "Jaipur Kurti" (36 rows, sells trousers/skirts/
# jumpsuits/tops under a "Kurti"-named ethnic brand -- 31 collision rows,
# e.g. "Jaipur Kurti Women White Regular Fit Solid Regular Trousers"
# pt=="trousers" wrongly resolving ethnic_top via ETHNIC_TOP_KEYWORDS'
# "kurti"), "SALWAR STUDIO" (25 rows, sells blouses/tops -- 24 collision
# rows via ETHNIC_BOTTOM_KEYWORDS' "salwar"), "Saree Swarg" (5 rows,
# sells tunics/kurtis -- collides with ETHNIC_ONE_PIECE_KEYWORDS' "saree",
# which is checked BEFORE ETHNIC_TOP_KEYWORDS, forcing a genuine tunic/
# kurti to the wrong ethnic SUB-class and excluding it from ever filling
# a "top" slot), "Pepe Jeans" (25 rows, sells knitwear/outerwear too --
# collides with WESTERN_BOTTOM_KEYWORDS' "jeans"). Deliberately a fixed,
# audited denylist, NOT a blanket "trust product_type_name alone" rule --
# that would regress legitimate cases where a generic/overloaded facet
# value genuinely needs a real (non-brand) ethnic descriptor elsewhere in
# the name to reclassify correctly, e.g. "NEUDIS Women ... Flared Maxi
# Lehenga Skirt" (pt=="skirt") is CORRECTLY ethnic_one_piece today via
# the legitimate "Lehenga" word in the name -- a facet-first rule would
# break that. Stripped as a case-insensitive LEADING prefix only (same
# discipline as src/catalogue/normalizer.py's brand-prefix strip) so the
# rest of the name -- including genuine descriptors -- still classifies
# normally.
#
# 2026-08-05 follow-up: re-ran the same audit methodology (compare
# classify_anchor(pt, name) with vs. without a candidate brand-prefix strip
# across the full 112,425-row unified catalogue.parquet, keeping only
# brands where the classification result ACTUALLY CHANGES -- not just
# "brand text happens to contain a keyword substring", which also flags
# false positives like plain descriptive titles with no real brand at all,
# e.g. "Solid Cotton Pyjama For Men" or "Kolhapuri Chappal For Women",
# where the "combined text" result is already correct and there is no
# brand to strip). Found 8 more genuine collisions, all via the same
# mechanism as the original four:
#   "DressBerry" (157 rows, 78 change classification) -- contains "dress"
#     (WESTERN_ONE_PIECE_KEYWORDS), e.g. a plain Top/Trousers/Jeans/Skirt
#     wrongly resolving western_one_piece.
#   "20Dresses" (43 rows, 19 change) -- same "dress" collision.
#   "Akkriti by Pantaloons" (43 rows, 10 change), "Rangmanch by Pantaloons"
#     (76 rows, 7 change), "Ajile by Pantaloons" (16 rows, 6 change),
#     "Honey by Pantaloons" (30 rows, 4 change), "Dreamz by Pantaloons"
#     (2 rows, 2 change) -- "Pantaloons" contains "pant"
#     (WESTERN_BOTTOM_KEYWORDS), e.g. a Sweatshirt/Top/Dupatta wrongly
#     resolving western_bottom. Two sibling Pantaloons house-brands,
#     "Annabelle by Pantaloons" and "SF Jeans by Pantaloons", were audited
#     too and are NOT collisions -- their own rows always hit an
#     earlier-priority keyword (shrug/poncho -> OUTERWEAR, jeans ->
#     WESTERN_BOTTOM via the genuine facet) before the "pant" substring is
#     ever reached, so classify_anchor's priority order already resolves
#     them correctly with no brand-prefix strip needed.
#   "Kraus Jeans" (20 rows, 1 changes) -- "Jeans" collides with
#     WESTERN_BOTTOM_KEYWORDS' "jeans" itself, e.g. a Pullover Sweater
#     wrongly resolving western_bottom.
# Verified negative control still holds after the expansion (NEUDIS Lehenga
# Skirt stays ethnic_one_piece -- see
# TestClassifyAnchorBrandPrefixCollisionRegression2026_07_30 in
# tests/test_outfit_package.py).
#
# Considered but rejected: routing coherence.py's is_ethnic_item/
# is_western_item through classify_item() instead of calling
# classify_anchor() directly (the seemingly more "consistent" fix, since
# classify_item() has its own pt-alone-first shortcut). Simulated across
# the full catalogue: this would flip is_ethnic_item's answer on 4,741
# rows, the overwhelming majority of which are itself a REGRESSION -- e.g.
# it makes the "NEUDIS Lehenga Skirt" negative control itself wrong
# (pt=="skirt" is a specific, non-generic facet value, so classify_item's
# shortcut resolves "western_bottom" from pt ALONE before ever looking at
# the name's genuine "Lehenga" descriptor). classify_item's shortcut is a
# deliberate, accepted trade-off for its own candidate slot-type-gating use
# case (preventing bundle-listing bleed-through, e.g. "Crop Top WITH
# Palazzo"), not a strictly-safer general-purpose classifier -- applying it
# to is_ethnic_item/is_western_item's different job (a garment's own
# ethnic/western-ness for occasion coherence gating) trades a rare
# brand-collision bug for a much larger, common one. The brand-prefix strip
# already lives inside classify_anchor() itself, so every caller --
# classify_item()'s own combined-text fallback included -- already shares
# this single fix at the correct layer; no further routing change needed.
_BRAND_PREFIX_COLLISIONS: tuple[str, ...] = (
    "jaipur kurti", "salwar studio", "saree swarg", "pepe jeans",
    "dressberry", "20dresses", "akkriti by pantaloons",
    "rangmanch by pantaloons", "ajile by pantaloons", "honey by pantaloons",
    "dreamz by pantaloons", "kraus jeans",
)
_BRAND_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(re.escape(b) for b in _BRAND_PREFIX_COLLISIONS) + r")[\s\-_,|]+",
    re.IGNORECASE,
)


def classify_anchor(product_type: str, prod_name: str = "") -> str:
    """Return anchor class: ethnic_top | ethnic_one_piece | ethnic_bottom |
    western_top | western_bottom | western_one_piece | outerwear | footwear | unknown."""
    pt = product_type.lower()
    name = _BRAND_PREFIX_RE.sub("", prod_name.lower())
    combined = pt + " " + name

    # 2026-08-05 (indowestern first-class fix): checked against the exact
    # product_type_name facet value, not a name substring -- "indo-western"/
    # "indowestern" also appears inside 54 trousers, 41 sherwani, 18 kurta,
    # and 16 nightwear rows' free-text names (a real Western-register item
    # merely *styled* indo-western), where a substring match would wrongly
    # reclassify those unrelated items. This was previously a one-off
    # `pt.lower().strip() == "indowestern"` special case living only in
    # coherence.py's office-register gate (which meant every OTHER caller
    # of classify_anchor()/classify_item() -- composer.py's anchor slot
    # composition, is_slot_type_allowed's candidate gating -- saw these 586
    # rows resolve chaotically depending on incidental keyword collisions
    # elsewhere in the name: "Dhoti" -> ethnic_bottom, "...Wide Leg Pant" ->
    # western_bottom via the same "pant"-substring-of-"Pantaloons"-shaped
    # bug as the brand-collision fix above, most rows with no such
    # collision -> unknown (invisible to slot-filling entirely). Promoted
    # to a first-class, unconditional facet-equality short-circuit here so
    # every caller gets ONE consistent, correct answer: these are complete
    # (kurta+churidar/dhoti/trousers) ethnic-crossover ensembles, the same
    # "already a full outfit" semantics as ETHNIC_ONE_PIECE_KEYWORDS' own
    # "suit-set"/"sharara set"/"salwar kameez" entries. Deliberately placed
    # BEFORE the keyword scan (not as an "unknown"-fallback after it) so it
    # also wins over incidental collisions, not just genuine non-matches --
    # this is why "indowestern" is in _GENERIC_FACET_VALUES above: that
    # makes classify_item() route jewellery items mislabeled with this
    # facet (e.g. "...Indo Western Necklace Set") through its own
    # combined-text accessory check FIRST, so they resolve "accessory"
    # rather than ever reaching this unconditional short-circuit.
    if pt.strip() == "indowestern":
        return "ethnic_one_piece"

    # 2026-08-05 (unknown-row apparel audit): re-measured the catalogue's
    # true classify_item()=="unknown" count (4,173 rows after the
    # indowestern fix above, down from the previously-reported 4,622) and
    # sampled every product_type_name bucket over ~20 rows. Most of the mass
    # is genuinely NOT apparel (Fashion/Rakhi/Clothing Accessories/gift-
    # hamper/fragrance/decor-type buckets — largely jewellery or non-apparel
    # merchandise, a separate ACCESSORY_KEYWORDS vocabulary gap, out of this
    # audit's "genuinely apparel" scope) or is apparel-shaped but correctly
    # declined per existing precedent (pt=="vest", 244 rows, sampled as
    # "SayItLoud/VIP/TOM BURG Men Vest (Pack of 2/8/11/12)" — undergarments,
    # same class of item as the already-declined "swimwear"=briefs finding;
    # classifying these would let undergarments fill outfit slots).
    #
    # Five buckets ARE genuine apparel with a clean product_type_name facet
    # (verified: the ENTIRE bucket sampled/reviewed, not just a subset, so
    # this is a facet-EQUALITY match, not a substring scan — the same
    # discipline as the indowestern fix above, deliberately avoiding a
    # substring rule that would reach into unrelated rows elsewhere in the
    # catalogue that merely happen to mention the same word in free text):
    #   "jodhpuri" (45 rows, all "Boy's/Men's ... Jodhpuri" — a structured
    #     ethnic formal jacket-and-trouser ensemble, same category as
    #     sherwani/bandhgala/achkan). NOT the same case as
    #     MEN_FORMALWEAR_KEYWORDS' deliberately-dropped bare "jodhpuri"
    #     substring above — that was about avoiding a false hit inside 19
    #     footwear rows ("Jodhpuri Mojaris"/"Jodhpuri Boots") whose own
    #     product_type_name is "footwear", not "jodhpuri" — a facet-equality
    #     check on the exact value "jodhpuri" never touches those rows.
    #   "pathani suit" (15 rows) / "kids pathani suit" (8 rows) — e.g. "MLS
    #     PATHANI SUIT 2PCS", a complete ethnic kurta-pyjama-style ensemble,
    #     same "already a full outfit" semantics as indowestern.
    #   "business plain suit" (25 rows, 19 unknown / 6 already resolved via
    #     the existing "business suit" keyword coincidentally repeated in
    #     their own name) — e.g. "MLS DOUBLE BREASTED SUIT", a genuine
    #     Western business suit; the facet value itself never contains the
    #     contiguous substring "business suit" ("business PLAIN suit"), so
    #     the existing keyword can't reach it.
    #   "lower" (40 rows incl. "Lowers") — e.g. "Grey Regular Fit Lower For
    #     Men", genuine Western track/lounge-style trousers. NOT the same
    #     case as WESTERN_BOTTOM_KEYWORDS' deliberately-dropped bare
    #     "lower"/"lowers" substring above — that was about a 943-row
    #     "flower"/"sunflower" false-positive risk from scanning free text;
    #     a facet-equality check on the exact value "lower" never touches
    #     those rows either.
    _pt_stripped_lower = pt.strip()
    if _pt_stripped_lower == "jodhpuri":
        return "men_formalwear"
    if _pt_stripped_lower in ("pathani suit", "kids pathani suit"):
        return "ethnic_one_piece"
    if _pt_stripped_lower == "business plain suit":
        return "outerwear"
    if _pt_stripped_lower in ("lower", "lowers"):
        return "western_bottom"

    if any(kw in combined for kw in ETHNIC_ONE_PIECE_KEYWORDS):
        return "ethnic_one_piece"
    if any(kw in combined for kw in ETHNIC_TOP_KEYWORDS):
        return "ethnic_top"
    if any(kw in combined for kw in ETHNIC_BOTTOM_KEYWORDS):
        return "ethnic_bottom"
    if any(kw in combined for kw in MEN_FORMALWEAR_KEYWORDS):
        return "men_formalwear"
    if any(kw in combined for kw in OUTERWEAR_KEYWORDS):
        return "outerwear"
    if any(kw in combined for kw in FOOTWEAR_KEYWORDS):
        return "footwear"
    if any(kw in combined for kw in WESTERN_ONE_PIECE_KEYWORDS):
        return "western_one_piece"
    if any(kw in combined for kw in WESTERN_BOTTOM_KEYWORDS):
        return "western_bottom"
    if any(kw in combined for kw in WESTERN_TOP_KEYWORDS):
        return "western_top"
    return "unknown"


def is_ethnic_item(product_type: str, prod_name: str = "") -> bool:
    """Return True if item is ethnic (kurta, lehenga, saree, etc.)."""
    anchor_class = classify_anchor(product_type, prod_name)
    return anchor_class in ("ethnic_top", "ethnic_one_piece", "ethnic_bottom", "men_formalwear")


def is_western_item(product_type: str, prod_name: str = "") -> bool:
    anchor_class = classify_anchor(product_type, prod_name)
    return anchor_class in ("western_top", "western_bottom", "western_one_piece")


@dataclass
class SlotSpec:
    """Definition of one complement slot to fill."""

    slot_name: str           # e.g. "bottom", "accessory", "footwear"
    search_query: str        # query terms to find candidates
    required: bool = True    # if True, empty slot is a hard failure; if False, optional


def gender_allowed(item_gender: str, look_gender: str) -> bool:
    """Return True if item gender is compatible with look gender.

    "unknown" is excluded from all gendered (men/women) looks — never guessed in.
    "unisex" look accepts everything.
    """
    ig = (item_gender or "unknown").lower()
    lg = look_gender.lower()
    if lg in ("men", "women"):
        return ig == lg
    return True  # unisex


def _occasion_register_tokens(occasion_slug: str) -> str:
    """Return extra register tokens appended to every slot's search_query so
    retrieval favours occasion-appropriate garments — e.g. an office bottom
    slot should surface tailored trousers, not a denim skirt.

    haldi/mehendi keep their own dedicated registers rather than the generic
    ethnic-festive one, since they already have a dedicated lightweight/floral
    colour+fabric bias (colour_score, fabric_score_delta) that would conflict
    with "embroidered" (haldi/mehendi favour light, undone-up looks). reception
    gets its own embellished-evening register, mirroring sangeet's bias.

    Wave 8: diwali/navratri/karva_chauth/raksha_bandhan/eid each get a
    dedicated register mirroring their own colour_score palette override
    above, so retrieval query text and colour scoring stay aligned (same
    pattern as haldi/mehendi/reception).

    Wave 9: gym gets a dedicated "activewear athletic gym sport" register —
    the generic EITHER/formality<3 fallback below would just return "casual",
    which is too weak a signal to steer retrieval away from ordinary
    lounge/casual wear toward genuine activewear.
    """
    if occasion_slug == "haldi":
        return "cotton floral"
    if occasion_slug == "mehendi":
        return "green floral festive"
    if occasion_slug == "reception":
        return "embellished formal evening"
    if occasion_slug == "engagement":
        return "elegant festive"
    if occasion_slug == "diwali":
        return "festive glam gold embellished"
    if occasion_slug == "navratri":
        return "chaniya choli bright colourful dance"
    if occasion_slug == "karva_chauth":
        return "red traditional bridal ethnic"
    if occasion_slug == "raksha_bandhan":
        return "casual festive light"
    if occasion_slug == "eid":
        return "pastel elegant festive"
    if occasion_slug == "gym":
        return "activewear athletic gym sport"
    occ = get_occasion(occasion_slug)
    if occ.ethnic_lean in (ETHNIC_HEAVY, ETHNIC_ONLY):
        return "festive embroidered"
    if occ.formality >= 3:
        return "formal tailored"
    return "casual"


def _append_register(slots: list[SlotSpec], occasion_slug: str) -> list[SlotSpec]:
    """Append the occasion's register tokens to every slot's search_query."""
    register = _occasion_register_tokens(occasion_slug)
    return [SlotSpec(s.slot_name, f"{s.search_query} {register}", s.required) for s in slots]


def _default_bottom_query(occasion_slug: str) -> str:
    """Return the base "bottom" slot query for a western top/outerwear anchor.

    Register-token appending alone ("... formal tailored") isn't enough to keep
    a denim/casual skirt out of a formal look — the literal words "jeans"/
    "skirt" are still IN the query text, so BM25/dense retrieval still surfaces
    them strongly.  For formality>=3, non-ethnic occasions (office, party_
    evening) this drops "jeans"/"skirt" from the query entirely so retrieval is
    steered toward tailored trousers, matching "office bottom must retrieve
    trousers, not a denim skirt".

    Phase B pool-composition fix: "trousers" alone (plus the register's
    generic "formal tailored") is a weak western-only signal — in a larger/
    differently-composed catalogue than this repo's own offline unified
    index, "formal"/"tailored" score close to embroidered ETHNIC formal wear
    too (a "Suit Set"/"Sharara Set" listing's own description text uses
    "formal"/"tailored" register words just as often), which can let an
    ethnic-heavy top-40 retrieval window dominate a western-register bottom
    slot even before any gate runs.  "pants"/"western" add unambiguous
    western-vocabulary lexical weight (BM25) without removing anything.

    Wave 9 gym override: "trousers jeans skirt" (the generic casual fallback
    below) is actively wrong for a gym look's bottom slot — none of those
    three are gym-appropriate, and "jeans"/"skirt" would pull the retrieval
    window toward exactly the wrong register. Returns explicit activewear
    bottom vocabulary instead, mirroring the office/party_evening branch's
    same "replace, don't just append" approach above.
    """
    if occasion_slug == "gym":
        return "leggings joggers track pants cargo pants shorts activewear"
    occ = get_occasion(occasion_slug)
    if occ.ethnic_lean == EITHER and occ.formality >= 3:
        return "trousers pants tailored western formal office wear"
    return "trousers jeans skirt"


def _default_footwear_query(occasion_slug: str, is_men: bool) -> str:
    """Return the base "footwear" slot query for a western top/bottom anchor.

    Wave 9 gym override: the generic "sneakers flats heels casual shoes"
    query text would help retrieval surface exactly the non-athletic footwear
    (flats/heels/loafers) that coherence.py's athletic-register gate then
    correctly rejects — this doesn't relax that gate, it only improves the
    odds of a genuine athletic shoe reaching the candidate pool in the first
    place when one exists (see is_athletic_footwear_item's catalogue-audit
    docstring for how thin that inventory actually is, especially for
    women).
    """
    if occasion_slug == "gym":
        return "sneakers sports shoes running shoes training shoes gym shoes trainers"
    return "sneakers casual shoes loafers men" if is_men else "sneakers flats heels casual shoes women"


def get_fill_slots(
    anchor_class: str,
    gender: str,
    occasion_slug: str,
    body_type: str | None = None,
    body_modifiers: list[str] | None = None,
) -> list[SlotSpec]:
    """Return ordered list of SlotSpecs to fill for a given anchor + gender + occasion.

    Gender: "men" | "women" | "unisex" (treated as women for ethnic, men for men's brands).

    Thin wrapper around _get_fill_slots_base(): appends occasion-register
    tokens (see _occasion_register_tokens) to every slot's search_query so
    retrieval is occasion-aware (formal tailored / festive embroidered /
    casual), without touching the base per-anchor-class slot definitions.

    P3: when body_type/body_modifiers are known, ALSO appends the body type's
    query-augmentation tokens (src.agents.outfit.body_type.query_tokens) —
    mirrors _occasion_register_tokens exactly. No-op (empty string appended)
    when body_type is None and body_modifiers is empty, so this is fully
    backward compatible with every existing call site.
    """
    slots = _get_fill_slots_base(anchor_class, gender, occasion_slug)
    slots = _append_register(slots, occasion_slug)
    return _append_body_type_tokens(slots, body_type, body_modifiers)


def _append_body_type_tokens(
    slots: list[SlotSpec], body_type: str | None, body_modifiers: list[str] | None
) -> list[SlotSpec]:
    """Append body-type query-augmentation tokens to every slot's search_query."""
    tokens = body_type_module.query_tokens(body_type, body_modifiers)
    if not tokens:
        return slots
    return [SlotSpec(s.slot_name, f"{s.search_query} {tokens}", s.required) for s in slots]


def _get_fill_slots_base(anchor_class: str, gender: str, occasion_slug: str) -> list[SlotSpec]:
    """Original per-anchor-class slot definitions (pre occasion-register tokens)."""
    g = gender.lower()
    is_men = g == "men"

    if anchor_class == "ethnic_top":
        if is_men:
            return [
                SlotSpec("bottom", "churidar pyjama dhoti ethnic bottom", required=True),
                SlotSpec("outerwear", "nehru jacket waistcoat ethnic waistcoat", required=False),
                SlotSpec(
                    "footwear", "mojaris juttis kolhapuris ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]
        else:
            return [
                SlotSpec("bottom", "palazzo churidar salwar sharara ethnic bottom", required=True),
                SlotSpec("accessory", "dupatta ethnic dupatta", required=True),
                SlotSpec(
                    "footwear", "juttis heels wedges ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]

    if anchor_class == "ethnic_one_piece":
        # lehenga / saree / anarkali / suit-set — never top/bottom
        return [
            SlotSpec("accessory", "dupatta jewellery clutch ethnic accessory", required=True),
            SlotSpec("footwear", "heels juttis ethnic footwear", required=True),
        ]

    if anchor_class == "men_formalwear":
        # sherwani / bandhgala
        return [
            SlotSpec("bottom", "churidar pyjama ethnic bottom", required=True),
            SlotSpec("footwear", "mojaris juttis ethnic footwear", required=True),
            SlotSpec("accessory", "pocket square safa ethnic accessory", required=False),
        ]

    if anchor_class == "ethnic_bottom":
        # sharara/palazzo as anchor → need ethnic top + dupatta
        if is_men:
            return [
                SlotSpec("top", "kurta ethnic top", required=True),
                SlotSpec(
                    "footwear", "mojaris juttis ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]
        else:
            return [
                SlotSpec("top", "kurta kurti ethnic top kameez", required=True),
                SlotSpec("accessory", "dupatta ethnic dupatta", required=True),
                SlotSpec(
                    "footwear", "juttis heels ethnic footwear",
                    required=occasion_slug in _FORMAL_ETHNIC_OCCASIONS,
                ),
            ]

    if anchor_class == "outerwear":
        return [
            SlotSpec("top", "top shirt blouse", required=True),
            SlotSpec("bottom", _default_bottom_query(occasion_slug), required=True),
        ]

    if anchor_class == "western_one_piece":
        return [
            SlotSpec("outerwear", "jacket cardigan blazer", required=False),
            SlotSpec("footwear", "shoes sandals boots heels", required=False),
            SlotSpec("accessory", "bag handbag", required=False),
        ]

    if anchor_class == "western_bottom":
        footwear_query = _default_footwear_query(occasion_slug, is_men)
        return [
            SlotSpec("top", "top shirt blouse", required=True),
            SlotSpec("outerwear", "jacket blazer coat cardigan", required=False),
            SlotSpec("footwear", footwear_query, required=False),
        ]

    # Default: western_top / unknown
    footwear_query = _default_footwear_query(occasion_slug, is_men)
    accessory_query = (
        "belt watch cap men accessory" if is_men else "handbag sling bag earrings women accessory"
    )
    return [
        SlotSpec("bottom", _default_bottom_query(occasion_slug), required=True),
        SlotSpec("outerwear", "jacket blazer coat cardigan", required=False),
        SlotSpec("footwear", footwear_query, required=False),
        SlotSpec("accessory", accessory_query, required=False),
    ]


def fabric_score_delta(
    item: dict, occasion_slug: str, formality_override: str | None = None
) -> float:
    """Return a score adjustment based on fabric/embellishment keywords for haldi vs sangeet.

    Base behaviour (formality_override absent — unchanged, backward compatible):
    - sangeet/reception/diwali: embellished items score +0.1; lightweight items
      score -0.1. Diwali joins this group (Wave 8) — a festival-of-lights
      evening register reads closer to sangeet/reception's embellished-glam
      bias than haldi/mehendi's undone-up daytime bias.
    - haldi/mehendi/raksha_bandhan: lightweight/floral items score +0.1;
      embellished items score -0.1. Raksha Bandhan joins this group (Wave 8) —
      formality 2/casual-festive reads closer to haldi's light, low-key bias.
    - all other occasions (including wedding_guest, navratri, karva_chauth,
      eid): 0.0 — these have their own dedicated colour_score palette overrides
      instead, and no strong embellishment-vs-lightweight signal.

    formality_override ("minimalist" | "comfortable", see FORMALITY_SOFTENER_VALUES —
    the `formality_softener` value a sibling intent-parser fix surfaces from
    queries like "something comfortable for sangeet dancing" or "not too
    flashy"): when present, OVERRIDES the base occasion-driven sign entirely —
    embellished/heavy items always score -0.1 and lightweight/plain items
    always score +0.1, regardless of occasion_slug. This fixes two confirmed
    live defects: (1) sangeet's base rule unconditionally boosts embellishment
    even when the query explicitly asks for something comfortable/low-key,
    with no way to suppress the bonus; (2) wedding_guest never got ANY
    embellishment-awareness because it isn't in the base sangeet/haldi/
    mehendi/reception occasion list — the override applies uniformly to
    wedding_guest (and any other occasion) once the user has signalled they
    want a toned-down look. wedding_guest's baseline (no override) behaviour
    stays 0.0, unchanged.

    Keyword check is heuristic — searches prod_name + detail_desc.
    """
    text = (
        (item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")
    ).lower()
    has_embellishment = any(kw in text for kw in SANGEET_EMBELLISHMENT_KEYWORDS)
    has_lightweight = any(kw in text for kw in HALDI_LIGHTWEIGHT_KEYWORDS)

    if formality_override in FORMALITY_SOFTENER_VALUES:
        if has_embellishment:
            return -0.1
        if has_lightweight:
            return 0.1
        return 0.0

    if occasion_slug not in (
        "sangeet", "haldi", "mehendi", "reception", "diwali", "raksha_bandhan",
    ):
        return 0.0

    if occasion_slug in ("sangeet", "reception", "diwali"):
        if has_embellishment:
            return 0.1
        if has_lightweight:
            return -0.1
    else:  # haldi, mehendi, raksha_bandhan
        if has_lightweight:
            return 0.1
        if has_embellishment:
            return -0.1
    return 0.0
