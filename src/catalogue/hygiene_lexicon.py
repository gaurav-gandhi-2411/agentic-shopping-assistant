r"""Keyword lexicon for src/catalogue/hygiene.py (data only; the rules live there).

Each row is (garment_type, category, regex). Regexes run on lower-cased text with
non-alphanumerics collapsed to a single space, so ``\b`` is safe and "t-shirt" is "t shirt".
Misspellings are listed explicitly because they occur in the real feeds (see the audit in
reports/wave1b/ and tests/test_catalogue_hygiene.py).
"""

from __future__ import annotations

_NF = "non_fashion"
_J = "jewellery"
_F = "footwear"
_A = "accessory"
_G = "garment"

LEXICON: tuple[tuple[str, str, str], ...] = (
    # non-fashion
    ("gift_card", _NF, r"e ?gift ?cards?|gift ?cards?|e ?gift|gift vouchers?|vouchers?"),
    # A bridal/rakhi hamper is a real gifting product users ask for, so it stays searchable
    # (accessory); only the packaging nouns below are dropped.
    ("hamper", _A, r"hampers?|giftables?"),
    ("gift_box", _NF, r"gift (?:box|boxes|wrap|bags?)"),
    # Bare "trunk(s)" is men's swim/underwear trunks (see swimwear below); only a storage trunk
    # named by what it stores is a non-garment.
    (
        "box",
        _NF,
        # "Zircon Box Set" (voylla) is a jewellery set sold in a box, not a box.
        # "Bridal Hamper Box B" is a hamper (accessory), so "hamper box" is not a box either.
        r"(?<!hamper )box(?:es)?(?! sets?\b)|trousseau|storage|organi[sz]ers?|"
        r"(?:bridal|wedding|saree|sari|lehenga|jewell?ery|trousseau|storage) trunks?",
    ),
    (
        "beauty",
        _NF,
        r"perfumes?|fragrances?|deodorants?|face (?:packs?|wash|serum|cream|gel|scrub)|"
        r"skin packs?|serums?|scrubs?|lotions?|soaps?|shampoos?|body (?:wash|butter|mist)",
    ),
    (
        "decor",
        _NF,
        r"decor|d cor|idols?|rugs?|baskets?|candles?|diffusers?|vases?|mugs?|bottles?|"
        r"cushions?|covers?|incense|wall (?:art|hanging)|photo frames?|"
        r"hidden product|1 basket|platters?",
    ),
    # footwear
    (
        "footwear",
        _F,
        r"footwear|shoes?|sneakers?|sandals?|juttis?|juttie|jutties|mojaris?|mojdi|kolhapuris?|"
        r"chappals?|chappers?|loafers?|mules?|wedges?|heels?|flip flops?|slides?|sliders?|"
        r"slippers?|clogs?|ballerinas?|mary janes?|boots?(?! cut)|pumps?|espadrilles?|moccasins?|"
        r"sliers|peshawaris?|khussas?|paduka|insoles?",
    ),
    # jewellery
    (
        "earrings",
        _J,
        r"ear ?rings?|jhumkas?|jhumkis?|jhumkhi|chandbalis?|studs?|ear ?chains?|ear ?cuffs?|"
        r"kaanbali|danglers?|ear ?climbers?",
    ),
    (
        "necklace",
        _J,
        r"necklaces?|neklaces?|nacklaces?|neckalce|neckace|necklace sets?|chokers?|"
        r"waist charm|pasa|haram|haramu|pendants?|pandents?|"
        r"mangalsutras?|manglasutras?|mangalsutra|manglasutra|mala|neck pieces?|neckpieces?|"
        r"haars?|rani haar|chains?|lockets?",
    ),
    (
        "bangles",
        _J,
        r"bangles?|bangels?|kadas?|cuffs?|hand ?cuffs?|paired bangles?|chooda|choora|chura|"
        r"chuda|chudo|chooras?|churas?",
    ),
    (
        "bracelet",
        _J,
        r"bracelets?|bascelets?|facelets?|hand harness|hand ?chains?|haath ?phool",
    ),
    ("ring", _J, r"rings?|toe rings?|nose rings?|nosepins?|nose pins?|noserings?|naths?|nath"),
    (
        "jewellery",
        _J,
        r"jewell?ery|jewelry|jewellery sets?|jewelry sets?|anklets?|payals?|"
        r"maang ?tika|maang ?teeka|maang ?tikka|mang ?teeka|mang ?tika|mangtika|maangtika|"
        r"maangteeka|matha patti|mathapatti|tikka|hathphool|hath phool|kamar ?bands?|"
        r"hip belts?|hipbelts?|oddiyanam|vanki|arm belts?|bajuband|bajubands?|armlets?|"
        r"kalangi|kalgi|borla|passa|jhoomer|kaleera|kaleeras?|sheesh ?phool|body jewelry|"
        r"hair maatal|maatal|matt?als?|nath|oxidi[sz]ed|polki|polti|kundan sets?|diadem|"
        r"bridal sets?|"
        r"necklace and earrings?|pearl sets?",
    ),
    # accessories
    (
        "dupatta",
        _A,
        r"dupattas?|chunnis?|stoles?|scarf|scarves|scraf|mufflers?|shawls?|wraps?|"
        r"uparna|odhni|ghoonghat",
    ),
    (
        "bag",
        _A,
        r"bags?|handbags?|potlis?|clutch|clutches|backpacks?|wallets?|card holders?|pouch|"
        r"pouches|sling|satchel|totes?|purse|purses|batua|wristlets?",
    ),
    ("belt", _A, r"belts?|waist belts?|cummerbunds?"),
    (
        "hair_accessory",
        _A,
        r"hair (?:accessor(?:y|ies)|bands?|clips?|pins?|sticks?|buns?|bows?|brooch(?:es)?|"
        r"clutchers?)|reusable flowers?|gajras?|hair extensions?|hairbands?|hairbuns?|hairpins?|jada|juda|scrunchies?|headbands?|tiaras?",
    ),
    (
        "accessory",
        _A,
        r"cufflinks?|cuff links?|brooch(?:es)?|brooches|lapel pins?|pocket squares?|"
        r"bow ties?|bowties?|neckties?|ties?(?! (?:dye|up|dyed|front|back|neck|waist|detail|knot))|"
        r"socks?|sunglasses|watches|watch|key ?chains?|keyrings?|umbrellas?|gloves?|"
        r"pagris?|pheta|safa|turbans?|topis?|topi|caps?(?! sleeves?)|hats?|"
        r"saree pins?|stole pins?|button clips?|shirt studs?|collar (?:pins?|stays?)|accessor(?:y|ies)|clothing accessories|masks?|"
        r"suspenders?|braces|tie sets?|bow|bows|bibs?|jaimala|varmala|choti|"
        r"calf length|ankle length|mid calf|crew|low cut",
    ),
    ("rakhi", _A, r"rakhis?|rakhi|lumba"),
    # garments
    ("lehenga", _G, r"lehengas?|lehnga|lacha|ghagras?|chaniya|ghagra choli|lehenga cholis?"),
    ("saree", _G, r"sarees?|saris?|sari|half sarees?"),
    ("choli", _G, r"cholis?|blouses?|blouse|crop tops?"),
    (
        "kurta",
        _G,
        r"kurtas?|kurtis?|kurti|kurta sets?|kurta pyjama|kurta pajama|pathani|angrakha|"
        r"tunics?|kaftans?|phiran|sadri",
    ),
    (
        "suit_set",
        _G,
        r"salwar|shalwar|salwar suits?|suit sets?|suits?|punjabi suits?|churidar|anarkali|"
        r"sharara|gharara|patiala|patiyala|ethnic sets?|co ?ords? sets?|dress material|pathani suits?",
    ),
    (
        "sherwani",
        _G,
        r"sherwanis?|bandhgalas?|jodhpuris?|jodhpuri suits?|jodhpuri sets?|achkan|nehru jackets?|"
        r"bandi|bundi|indo ?western|dhoti kurtas?|angarkha|prince coat",
    ),
    ("blazer", _G, r"blazers?|waistcoats?|waist coats?|tuxedos?|tuxidos?|three piece|breeches"),
    (
        "dress",
        _G,
        r"dress(?:es)?|gowns?|jumpsuits?|playsuits?|rompers?|maxi|frocks?|dungarees?|"
        r"bodysuits?|co ?ords?|coords?|twinsets?",
    ),
    (
        "top",
        _G,
        r"tops?|tees?|t ?shirts?|shirts?|polos?|tanks?|camis?|camisoles?|vests?|bralettes?|"
        r"bras?|sports bras?|sports_bra|henley|skivvy|jerseys?|blousons?|tops blouson|"
        r"modal tee|kids tuxido|overshirts?",
    ),
    (
        "knitwear",
        _G,
        r"pullovers?|sweatshirts?|sweat shirts?|hoodies?|sweaters?|knitwear|cardigans?|shrugs?",
    ),
    (
        "outerwear",
        _G,
        r"jackets?|coats?|shackets?|capes?|ponchos?|ponchu|outerwear|overlays?|bomber|puffer|"
        r"windcheater|capes overlays|winter wear|winterwear",
    ),
    (
        "trousers",
        _G,
        r"trousers?|pants?|harem|heram|palazzos?|plazzos?|leggings?|joggers?|track ?pants?|cargos?|"
        r"cargo pants?|chinos?|culottes?|capris?|lowers?|bottoms?|bottom ?wear|bottomwear|"
        r"salwars?|pyjamas?|pajamas?|lounge pants?|slacks?|dhotis?|dhoti|wide leg|pant style|"
        r"tights?|trackpant|track suits?|tracksuits?|sweatpants?|jeggings?",
    ),
    ("jeans", _G, r"jeans?"),
    # "short" singular is an adjective ("Short and Long Haram"), so only the plural is a garment.
    ("shorts", _G, r"shorts|skorts?|boxers?|briefs?"),
    ("skirt", _G, r"skirts?"),
    (
        "nightwear",
        _G,
        r"night ?wear|night ?suits?|nightsuits?|sleepwear|night ?dress(?:es)?|nighties|nightie|"
        r"night gowns?|loungers?|loungewear|lounge wear|robes?|pj sets?|camisole sets?",
    ),
    (
        "swimwear",
        _G,
        r"swimwear|swimsuits?|bikinis?|swim|innerwear|underwear|lingerie|trunks?",
    ),
)


# Types that carry no category information on their own (feed catch-alls); the title decides.
GENERIC_TYPES: frozenset[str] = frozenset(
    {
        "",
        "fashion",
        "all products",
        "all product",
        "others",
        "other",
        "clothing",
        "accessories",
        "clothing accessories",
        "diwali sale 2025",
        "article",
        "sets",
        "set",
        "combo",
        "bundle",
        "clothing set",
        "suit set",
        "winter wear",
        "n a",
        "none",
        "nan",
        "kids bottomwear",
        "basic women",
        "vintage women",
        "elegant women",
        "chic women",
        "bridal collection s",
        "other essentials",
        "men s jewellery",
    }
)
