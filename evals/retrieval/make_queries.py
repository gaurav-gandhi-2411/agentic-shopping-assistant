"""Generate the 60 frozen retrieval-eval query cases (seed 42, fully deterministic).

    python -m evals.retrieval.make_queries            # (re)write evals/fixtures/retrieval/*.json
    python -m evals.retrieval.make_queries --table    # print the stratification table (markdown)

The 60 cases are stratified, not sampled: case i has occasion = i % 12 and a gender slot = i % 5
(pattern W, M, W, M, either), and 12 and 5 are coprime, so every (occasion, gender-slot) pair
occurs exactly once (24 women / 24 men / 12 either). Phrasing style and budget are balanced
lists (12 each) shuffled with ``random.Random(42)``. Hard cases (20% = 12) are every 5th case
with a rotating offset; they always carry a colour AND a budget, and half get a style term
("indo-western"/"fusion") when the garment is clothing.

The query text is public product wording only (it is later sent to LLM judges); no personal data.
The committed fixtures are the frozen set: regenerate only on purpose and review the diff.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
N_CASES = 60
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "retrieval"

OCCASIONS = [
    "sangeet",
    "haldi",
    "mehendi",
    "wedding_guest",
    "reception_cocktail",
    "office_formal",
    "festive_diwali",
    "casual_daily",
    "college",
    "date_night",
    "travel_party",
    "accessory_footwear",
]
# (plain English, English+Hinglish noun, romanised "... ke liye", keyword form)
OCC_TEXT = {
    "sangeet": ("sangeet", "sangeet night", "sangeet ke liye", "sangeet"),
    "haldi": ("a haldi ceremony", "haldi function", "haldi ke liye", "haldi"),
    "mehendi": ("mehendi", "mehendi function", "mehendi ke liye", "mehendi"),
    "wedding_guest": ("a wedding", "shaadi", "shaadi ke liye", "wedding guest"),
    "reception_cocktail": ("a reception", "reception party", "reception ke liye", "cocktail"),
    "office_formal": ("office", "office wear", "office ke liye", "office formal"),
    "festive_diwali": ("diwali", "diwali puja", "diwali ke liye", "diwali festive"),
    "casual_daily": ("daily wear", "daily use", "roz pehenne ke liye", "casual daily"),
    "college": ("college", "college roz", "college ke liye", "college"),
    "date_night": ("date night", "date night", "date night ke liye", "date night"),
    "travel_party": ("a party", "party", "party ke liye", "party"),
    "accessory_footwear": ("a wedding", "shaadi", "shaadi ke liye", "wedding"),
}
# occasion -> gender slot -> garments (first = lower-priced option)
GARMENTS: dict[str, dict[str, list[str]]] = {
    "sangeet": {
        "women": ["anarkali gown", "lehenga"],
        "men": ["kurta set", "sherwani"],
        "either": ["ethnic outfit"],
    },
    "haldi": {
        "women": ["kurta set", "sharara set"],
        "men": ["kurta", "kurta pyjama set"],
        "either": ["outfit"],
    },
    "mehendi": {
        "women": ["palazzo kurta set", "sharara set"],
        "men": ["kurta", "nehru jacket set"],
        "either": ["outfit"],
    },
    "wedding_guest": {
        "women": ["saree", "lehenga"],
        "men": ["kurta set", "bandhgala"],
        "either": ["outfit"],
    },
    "reception_cocktail": {
        "women": ["gown", "saree"],
        "men": ["blazer", "suit"],
        "either": ["outfit"],
    },
    "office_formal": {
        "women": ["shirt", "trousers"],
        "men": ["formal shirt", "trousers"],
        "either": ["blazer"],
    },
    "festive_diwali": {
        "women": ["kurta set", "anarkali"],
        "men": ["kurta", "kurta pyjama set"],
        "either": ["ethnic wear"],
    },
    "casual_daily": {
        "women": ["cotton kurta", "top"],
        "men": ["t-shirt", "casual shirt"],
        "either": ["co-ord set"],
    },
    "college": {"women": ["kurti", "jeans"], "men": ["hoodie", "shirt"], "either": ["sneakers"]},
    "date_night": {"women": ["dress", "top"], "men": ["shirt", "blazer"], "either": ["outfit"]},
    "travel_party": {
        "women": ["jumpsuit", "dress"],
        "men": ["shirt", "jacket"],
        "either": ["jacket"],
    },
    "accessory_footwear": {
        "women": ["jhumka earrings", "juttis"],
        "men": ["kolhapuri sandals", "loafers"],
        "either": ["necklace set", "sneakers"],
    },
}
HIGH_END = {"lehenga", "sherwani", "saree", "bandhgala", "suit", "gown", "sharara set"}
NON_CLOTHING = {
    "jhumka earrings",
    "juttis",
    "kolhapuri sandals",
    "loafers",
    "necklace set",
    "sneakers",
}
COLOURS = {
    "haldi": ["yellow", "white", "orange"],
    "mehendi": ["green", "yellow", "pink"],
    "accessory_footwear": ["gold", "silver", "black"],
}
# q15 drew "green outfit for mehendi under ₹1000", a near-duplicate of the strict-gold query
# "green outfit for mehendi" (a label-set query this eval must not overlap). Same stratum
# (mehendi, gender-neutral, plain English, <= 1000), different colour. The override is applied
# after the rng draw, so no other case changes.
COLOUR_OVERRIDES = {"q15": "yellow"}
DEFAULT_COLOURS = ["black", "blue", "red", "maroon", "pink", "white", "navy", "green", "grey"]
BUDGETS: list[int | None] = [None, 1000, 1500, 3000, 5000]
STYLES = ["plain_en", "en_hinglish", "romanised_hinglish", "keyword_typo", "stylist_descriptive"]
GENDER_SLOTS = ["women", "men", "women", "men", "either"]  # i % 5
STYLE_TERMS = ["indo-western", "fusion"]


def _typo(word: str, rng: random.Random) -> str:
    """Drop one interior letter (deterministic via rng) for the keyword/typo phrasing."""
    if len(word) < 5:
        return word
    k = rng.randrange(1, len(word) - 1)
    return word[:k] + word[k + 1 :]


def _phrase(  # noqa: PLR0913 - one flat template function is clearer than a class here
    style: str,
    garment: str,
    colour: str | None,
    occ: str,
    gender: str,
    budget: int | None,
    rng: random.Random,
) -> str:
    """Render one query string in the requested phrasing style."""
    plain, hin, roman, kw = OCC_TEXT[occ]
    col = f"{colour} " if colour else ""
    gp = {"women": "women's ", "men": "men's ", "either": ""}[gender]
    if style == "plain_en":
        bud = f" under ₹{budget}" if budget else ""
        return f"{gp}{col}{garment} for {plain}{bud}"
    if style == "en_hinglish":
        bud = f", budget {budget} max" if budget else ", kuch accha dikhao"
        return f"Need {gp}{col}{garment} for {hin} yaar{bud}"
    if style == "romanised_hinglish":
        who = {"women": " ladkiyon ke liye", "men": " ladkon ke liye", "either": ""}[gender]
        bud = f", {budget} se kam mein" if budget else ""
        return f"{roman} {col}{garment} chahiye{who}{bud}"
    if style == "keyword_typo":
        c = _typo(colour, rng) if colour and rng.random() < 0.5 else (colour or "")
        g = _typo(garment.split()[0], rng) + "".join(" " + w for w in garment.split()[1:])
        gw = {"women": "women", "men": "men", "either": ""}[gender]
        bud = f" <{budget}" if budget else ""
        return " ".join(x for x in (c, g, kw, gw) if x) + bud
    who = {"women": "a woman", "men": "a man", "either": "someone"}[gender]
    bud = f" Please keep it under ₹{budget}." if budget else ""
    return (
        f"Can you suggest {col}{garment} for {who} to wear to {plain}? "
        f"I want it to look elegant but stay comfortable.{bud}"
    )


def build_cases() -> list[dict]:
    """Return the 60 case dicts, deterministic for SEED."""
    rng = random.Random(SEED)
    styles = [STYLES[i % 5] for i in range(N_CASES)]
    budgets = [BUDGETS[i % 5] for i in range(N_CASES)]
    rng.shuffle(styles)
    rng.shuffle(budgets)
    hard = {k * 5 + (k % 5) for k in range(N_CASES // 5)}
    for i in sorted(hard):  # hard cases need a budget: swap with a non-hard budgeted case
        if budgets[i] is None:
            j = next(j for j in range(N_CASES) if j not in hard and budgets[j] is not None)
            budgets[i], budgets[j] = budgets[j], budgets[i]
    cases = []
    for i in range(N_CASES):
        occ, gender = OCCASIONS[i % 12], GENDER_SLOTS[i % 5]
        garments = GARMENTS[occ][gender]
        budget = budgets[i]
        garment = garments[i // 12 % len(garments)]
        if budget is not None and budget <= 1500 and garment in HIGH_END and len(garments) > 1:
            garment = next((g for g in garments if g not in HIGH_END), garment)
        is_hard = i in hard
        pool = COLOURS.get(occ, DEFAULT_COLOURS)
        colour = rng.choice(pool) if (is_hard or i % 2 == 0) else None
        colour = COLOUR_OVERRIDES.get(f"q{i + 1:02d}", colour)  # applied after rng: no drift
        shown_garment = garment
        style_term = None
        if is_hard and garment not in NON_CLOTHING and (i // 5) % 2 == 0:
            style_term = STYLE_TERMS[(i // 10) % 2]
            shown_garment = f"{style_term} {garment}"
        text = _phrase(styles[i], shown_garment, colour, occ, gender, budget, rng)
        cases.append(
            {
                "id": f"q{i + 1:02d}",
                "text": text,
                "language_style": styles[i],
                "occasion": occ,
                "gender": gender,
                "budget_inr": budget,
                "hard": is_hard,
                "constraints": {
                    "gender": None if gender == "either" else gender,
                    "colour": colour,
                    "category": garment,
                    "max_price": budget,
                    "style_term": style_term,
                },
                "notes": (
                    "hard: multi-constraint (occasion + colour + budget"
                    + (f" + style term '{style_term}'" if style_term else "")
                    + ")"
                    if is_hard
                    else "standard"
                ),
            }
        )
    return cases


def stratification_table(cases: list[dict]) -> str:
    """Markdown marginal counts per stratification axis."""
    out = []
    axes = [
        ("occasion", lambda c: c["occasion"]),
        ("gender", lambda c: c["gender"]),
        ("budget_inr", lambda c: str(c["budget_inr"])),
        ("language_style", lambda c: c["language_style"]),
        ("hard", lambda c: str(c["hard"])),
        ("has colour", lambda c: str(bool(c["constraints"]["colour"]))),
    ]
    for name, key in axes:
        counts = Counter(key(c) for c in cases)
        out.append(
            f"| {name} | " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + " |"
        )
    return "| axis | counts (n=%d) |\n|---|---|\n" % len(cases) + "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", action="store_true", help="print stratification table only")
    args = ap.parse_args()
    cases = build_cases()
    if args.table:
        print(stratification_table(cases))
        return
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for c in cases:
        (FIXTURE_DIR / f"{c['id']}.json").write_text(
            json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(cases)} fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
