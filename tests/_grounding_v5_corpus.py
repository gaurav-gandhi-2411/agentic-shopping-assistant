"""Deterministic corpora for the V5 grounding acceptance rules (docs/grounding-v5-acceptance.md).

No LLM, no network, no randomness: every case is built from the fixture items below by plain
templates. An adversarial case lists ``bad_spans`` (exact substrings that must NOT survive in the
grounded output); a clean case lists the ``expected`` output (input with only id tokens
substituted, computed here independently of src/agents/grounded_render.py).
"""

from __future__ import annotations

import re
from typing import Any

ITEMS: list[dict[str, Any]] = [
    {
        "article_id": "1001",
        "display_name": "Riviera Cotton Sundress",
        "price_inr": 2999.0,
        "colour": "blue",
        "product_type": "dress",
        "detail_desc": "Relaxed cotton sundress.",
    },
    {
        "article_id": "1002",
        "display_name": "Gloss Wrap Skirt",
        "price_inr": 1299.0,
        "colour": "black",
        "product_type": "skirt",
        "detail_desc": "Satin wrap skirt.",
    },
    {
        "article_id": "1003",
        "display_name": "Indigo Block Print Kurta",
        "price_inr": 1499.0,
        "colour": "indigo",
        "product_type": "kurta",
        "detail_desc": "Hand block printed kurta.",
    },
    {
        "article_id": "1004",
        "display_name": "Banarasi Silk Saree",
        "price_inr": 8499.0,
        "colour": "red",
        "product_type": "saree",
        "detail_desc": "Woven silk saree.",
        "in_stock": False,
    },
    {
        "article_id": "1005",
        "display_name": "Linen Co-ord Set",
        "price_inr": 3499.0,
        "colour": "beige",
        "product_type": "set",
        "detail_desc": "Linen two piece set.",
        "sizes": ["S", "M"],
        "in_stock": True,
    },
    {
        "article_id": "1006",
        "display_name": "Embroidered Juttis",
        "price_inr": 999.0,
        "colour": "gold",
        "product_type": "juttis",
        "detail_desc": "Embroidered flat juttis.",
        "mrp_inr": 1249.0,
        "discount_pct": 20.0,
    },
]
BY_ID = {it["article_id"]: it for it in ITEMS}
_TOKEN = re.compile(r"\[(item|price):(\d+)\]")


def expected_render(text: str) -> str:
    """Independent re-implementation of id substitution used as the clean-corpus oracle."""

    def sub(m: re.Match[str]) -> str:
        it = BY_ID[m.group(2)]
        return it["display_name"] if m.group(1) == "item" else f"₹{it['price_inr']:,.0f}"

    return _TOKEN.sub(sub, text)


def _case(
    cid: str,
    category: str,
    reply: str,
    bad: list[str],
    keep: list[str] | None = None,
    user: str = "",
) -> dict[str, Any]:
    return {
        "id": cid,
        "category": category,
        "reply": reply,
        "bad_spans": bad,
        "must_keep": keep or [],
        "user_text": user,
    }


def adversarial() -> list[dict[str, Any]]:
    """>= 60 replies, each containing at least one ungrounded claim."""
    cases: list[dict[str, Any]] = []
    n = 0

    def add(
        category: str, reply: str, bad: list[str], keep: list[str] | None = None, user: str = ""
    ) -> None:
        nonlocal n
        n += 1
        cases.append(_case(f"adv{n:03d}", category, reply, bad, keep, user))

    for it in ITEMS:  # wrong price, via id token and via plain name
        i = it["article_id"]
        wrong = int(it["price_inr"]) + 137  # never equals another fixture price
        add("wrong_price", f"You will love [item:{i}] at ₹{wrong:,} today.", [f"₹{wrong:,}"])
        add(
            "wrong_price",
            f"The {it['display_name']} costs ₹{wrong:,} which is a steal.",
            [f"₹{wrong:,}"],
        )
    for sym, amt in (("$", "25"), ("€", "30"), ("£", "20"), ("USD ", "40")):  # wrong currency
        add("wrong_currency", f"[item:1002] is only {sym}{amt} right now.", [f"{sym}{amt}"])
    for form in (
        "₹1,400",
        "Rs. 1400",
        "INR 1400",
        "1.4k",
        "Rs 1,400/-",
        "1400 rupees",
        "Rs.1450",
        "INR1450",
        "rs 1450",
        "1.45k",
        "₹1450 only",
        "one thousand four hundred rupees",
        "INR:1100",
        "Rupees-1100",
    ):
        add("price_format_variant", f"The Gloss Wrap Skirt is available for {form}.", [form])
    for sizes in ("S, M and L", "XS, S, M", "XL"):  # invented size
        add("invented_size", f"[item:1002] comes in sizes {sizes} so you can pick yours.", [sizes])
    add("invented_size", "[item:1003] runs small so order a size up.", ["runs small"])
    add("invented_size", "[item:1005] is available in XL as well.", ["XL"])
    add("invented_size", "Grab [item:1001] in UK 8 for a perfect fit.", ["UK 8"])
    add("invented_size", "[item:1006] is true to size for most shoppers.", ["true to size"])
    add("invented_size", "[item:1003] comes in size medium for most people.", ["size medium"])
    add("stock", "[item:1004] is in stock and ready to ship.", ["in stock"])
    add("stock", "[item:1001] is in stock right now.", ["in stock"])
    add("stock", "Hurry, only 2 left of [item:1003].", ["only 2 left"])
    add("stock", "[item:1005] is sold out at the moment.", ["sold out"])
    add("stock", "[item:1002] is back in stock.", ["back in stock"])
    add("stock", "[item:1006] has limited stock so grab it.", ["limited stock"])
    add("discount", "[item:1001] is 20% off today.", ["20% off"])
    add("discount", "[item:1003] is on sale this week.", ["on sale"])
    add("discount", "[item:1006] is 30% off, a great deal.", ["30% off"])
    add("discount", "Flat 40% off on [item:1002].", ["40% off"])
    add("discount", "Save ₹500 on [item:1005] with this offer.", ["Save ₹500"])
    add("discount", "[item:1004] has a special offer running.", ["special offer"])
    add("discount", "[item:1002] is half price this week.", ["half price"])
    for k, bad_id in enumerate(("9999", "4242", "abc123")):  # id not shown
        add("id_not_shown", f"I recommend [item:{bad_id}] for the occasion.", [bad_id])
        add("id_not_shown", f"Try [item:1001] at [price:{bad_id}] for a bargain.", [bad_id])
    add("id_not_shown", "A nice option is [size:1001] for you.", ["[size"])
    add("id_not_shown", "A nice option is [item:1001 for you.", ["[item"])
    for bare in (1700, 2150, 899):  # numeric hallucination inside prose
        add("numeric_in_prose", f"The Gloss Wrap Skirt is yours for {bare} only.", [str(bare)])
    add("numeric_in_prose", "Reviewers call it a steal at 1750 for what you get.", ["1750"])
    add("price_wrong_item", "[item:1001] at [price:1002] is great value.", ["is great value"])
    add("price_wrong_item", "The Riviera Cotton Sundress is priced at ₹1,299.", ["priced at"])
    add("price_wrong_item", "[item:1004] at [price:1006] is a great pick.", ["is a great pick"])
    add("delivery", "[item:1002] comes with free shipping across India.", ["free shipping"])
    add("delivery", "[item:1003] delivers in 3 days.", ["3 days"])
    add("delivery", "[item:1001] is available for pickup today.", ["pickup"])
    for first, second, bad in (  # a good sentence next to a bad one: the good one must survive
        ("[item:1001] at [price:1001] is a lovely pick.", "It is also 25% off.", "25% off"),
        ("[item:1002] pairs well with a plain top.", "It is in stock in size M.", "in stock"),
        ("[item:1003] suits a daytime event.", "Get it for ₹999 only.", "₹999"),
        ("[item:1005] is a relaxed set.", "Sizes S, M, L are available.", "S, M, L"),
    ):
        add("mixed", f"{first} {second}", [bad], keep=[expected_render(first)])
    return cases


def clean() -> list[dict[str, Any]]:
    """>= 30 clean replies; output must equal expected_render(reply)."""
    replies: list[tuple[str, str]] = [  # (reply, user_text)
        ("For the beach, [item:1001] is a great pick; the relaxed cut works over swimwear.", ""),
        ("[item:1001] at [price:1001] suits a sunny brunch.", ""),
        ("[item:1002] at [price:1002] pairs well with a simple top.", ""),
        ("[item:1003] at [price:1003] is easy to dress up or down.", ""),
        ("[item:1004] at [price:1004] is a festive statement.", ""),
        ("[item:1005] at [price:1005] is a polished everyday set.", ""),
        ("[item:1006] at [price:1006] finishes an ethnic look.", ""),
        ("The Gloss Wrap Skirt is priced at ₹1,299 and drapes well.", ""),
        ("The Gloss Wrap Skirt is priced at Rs. 1299 and drapes well.", ""),
        ("The Gloss Wrap Skirt is priced at INR 1299 and drapes well.", ""),
        ("The Gloss Wrap Skirt is about 1.3k and drapes well.", ""),
        ("The Gloss Wrap Skirt is Rs 1,299/- and drapes well.", ""),
        (
            "I don't have size data so I couldn't filter by size; here are the options available.",
            "",
        ),
        ("I don't have stock information, so check the product page for availability.", ""),
        ("I don't have size or fit information; check the product page.", "do you have size M"),
        ("I don't have pricing information for that one; check the product page.", ""),
        (
            "These are within your budget of ₹2000: [item:1002] and [item:1003].",
            "show me something under ₹2000",
        ),
        ("[item:1003] is under ₹2,000 as you asked, and [item:1002] is too.", "kurta under ₹2,000"),
        ("[item:1005] is available in S and M.", ""),
        ("[item:1005] is in stock.", ""),
        ("[item:1004] is currently out of stock.", ""),
        ("[item:1006] is 20% off its listed price.", ""),
        ("The indigo block print feels fresh for a daytime event.", ""),
        ("Pair [item:1001] with flat sandals for a relaxed holiday look.", ""),
        ("Both [item:1002] and [item:1006] are in warm festive tones.", ""),
        ("Here is a clean, understated option: [item:1005]. Add [item:1006] to finish.", ""),
        ("[item:1003] works for a mehendi; [item:1004] works for a wedding. Pick one.", ""),
        ("Nothing here matches exactly, but [item:1001] is the closest available match.", ""),
        ("Pick one and I can put together a complete look around it.", ""),
        ("[item:1002] is a lighter pick than [item:1003], so mix and match.", ""),
        ("You asked for a lehenga; the closest available are [item:1004] and [item:1001].", ""),
        ("[item:1001] is breezy and [item:1002] is smart, so it depends on your plans.", ""),
        ("Happy to narrow this down by colour or occasion if you tell me more.", ""),
        ("Great taste; [item:1003] keeps things simple while [item:1005] adds structure.", ""),
    ]
    out: list[dict[str, Any]] = []
    for k, (reply, user) in enumerate(replies, 1):
        out.append(
            {
                "id": f"clean{k:03d}",
                "reply": reply,
                "user_text": user,
                "expected": expected_render(reply),
            }
        )
    return out
