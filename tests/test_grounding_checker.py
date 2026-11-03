"""Unit tests for evals/grounding/checker.py (W10 grounding checker).

The checker is the measurement instrument for E5, so these tests pin both directions: grounded
text must pass (false-positive guard) and ungrounded text must be flagged (false-negative guard).
"""
from __future__ import annotations

import copy
import time

import pytest

from evals.grounding.checker import check
from evals.grounding.measure_checker import HOLDOUT, LABELS, measure

A = {"article_id": "a1", "display_name": "Riviera Sundress", "prod_name": "Riviera Sundress",
     "price_inr": 2999.0, "detail_desc": "Blue sundress. 100% cotton."}
B = {"article_id": "b1", "display_name": "Gloss Wrap Skirt", "prod_name": "Gloss Wrap Skirt",
     "price_inr": 1499.0, "detail_desc": "Wrap skirt."}
LEVIS = {"article_id": "c1", "display_name": "Levi's 511 Slim Fit Jeans",
         "prod_name": "Levi's 511 Slim Fit Jeans", "price_inr": 3299.0, "detail_desc": ""}
FUTURE = {"article_id": "g1", "display_name": "Slim Chinos", "prod_name": "Slim Chinos",
          "price_inr": 1599.0, "mrp_inr": 1999.0, "discount_pct": 20, "sizes": ["30", "32", "34"],
          "in_stock": True, "stock_count": 2, "delivery_days": [3, 5], "free_shipping": True}


def kinds(text: str, items: list[dict], **kw: object) -> list[str]:
    return [v.kind for v in check(text, items, **kw)]  # type: ignore[arg-type]


# ----------------------------------------------------------------- price: grounded forms


@pytest.mark.parametrize("text", [
    "The Riviera Sundress is ₹2,999.",
    "The Riviera Sundress is ₹ 2999.",
    "The Riviera Sundress is Rs 2999.",
    "The Riviera Sundress is Rs. 2,999.",
    "The Riviera Sundress is Rs.2999/-.",
    "The Riviera Sundress is INR 2999.",
    "The Riviera Sundress is 2,999 rupees.",
    "The Riviera Sundress is 2999 INR.",
    "The Riviera Sundress is ₹2,999.00.",
    "The Riviera Sundress is priced at 2999.",
    "The Riviera Sundress is ₹3k.",          # 3000 is within Rs 1 of 2999
    "The Riviera Sundress is ₹2,999.40.",
    "The Riviera Sundress costs 2,998.",
])
def test_grounded_price_formats_pass(text: str) -> None:
    assert check(text, [A]) == []


@pytest.mark.parametrize("text", [
    "The Riviera Sundress is ₹2,997.",       # 2 off: outside the Rs 1 tolerance
    "The Riviera Sundress is ₹3,001.",
    "The Riviera Sundress is ₹299.",
    "The Riviera Sundress is ₹29,990.",
    "The Riviera Sundress is $30.",
    "The Riviera Sundress is 4 thousand rupees.",
    "The Riviera Sundress is about 3.5k.",
])
def test_ungrounded_price_flagged(text: str) -> None:
    got = check(text, [A])
    assert got and got[0].kind in {"price", "price_wrong_item"}


def test_indian_grouping_and_lakh() -> None:
    big = dict(A, price_inr=100000.0)
    assert check("It is ₹1,00,000.", [big]) == []
    assert check("It is ₹1 lakh.", [big]) == []
    assert check("It is ₹1.5 lakh.", [big]) != []


def test_range_both_endpoints_checked() -> None:
    assert check("From ₹1,499 to ₹2,999.", [A, B]) == []
    got = check("Between ₹999 and ₹1,499.", [A, B])
    assert [v.text for v in got] == ["₹999"]
    got = check("In the ₹999–₹1,499 range.", [B])
    assert [v.text for v in got] == ["₹999"]
    assert [v.text for v in check("Pieces at ₹1,499-3,500.", [B])] == ["3,500"]


def test_price_ignores_non_price_numbers() -> None:
    assert check("I found 5 options, 2 colours, 3 pieces from Myntra.", [A]) == []
    assert check("Delivered nowhere; model is 5'8 and the dupatta is 2.43 meters long.", [A]) != []
    assert check("The dupatta is 2.43 meters and 88 cm wide.", [A]) == []


def test_wrong_item_price_is_its_own_kind() -> None:
    got = check("The Riviera Sundress is yours at ₹1,499.", [A, B])
    assert [v.kind for v in got] == ["price_wrong_item"]


def test_nearest_item_attribution_with_two_items() -> None:
    ok = "The Riviera Sundress is ₹2,999 and the Gloss Wrap Skirt is ₹1,499."
    swapped = "The Riviera Sundress is ₹1,499 and the Gloss Wrap Skirt is ₹2,999."
    assert check(ok, [A, B]) == []
    assert [v.kind for v in check(swapped, [A, B])] == ["price_wrong_item"] * 2


def test_sum_only_counts_in_a_total_sentence() -> None:
    assert check("Together they come to ₹4,498.", [A, B]) == []
    assert kinds("The Riviera Sundress is ₹4,498.", [A, B]) != []


def test_extra_amounts_back_code_computed_totals() -> None:
    assert check("Your look totals ₹7,000.", [A], extra_amounts=[7000.0]) == []
    assert check("Your look totals ₹7,000.", [A]) != []


# ----------------------------------------------------------------- names, ids, years


def test_number_in_product_name_is_not_a_price() -> None:
    assert check("Levi's 511 Slim Fit Jeans are a classic.", [LEVIS]) == []
    assert check("The Levi's 511 is a classic fit.", [LEVIS]) == []


def test_name_number_exemption_never_covers_a_rupee_amount() -> None:
    assert kinds("The Levi's 511 is ₹511.", [LEVIS]) != []


def test_name_number_exemption_needs_the_name_context() -> None:
    assert kinds("These jeans are priced at 511.", [LEVIS]) != []


def test_ids_and_years_are_not_prices() -> None:
    assert check("Article 17048614 and SKU 99887766 are on the card.", [A]) == []
    assert check("A pick from the 2024 collection, new for Diwali 2025.", [A]) == []


# ----------------------------------------------------------------- the budget echo rule


def test_budget_echo_allowed_only_in_budget_context() -> None:
    assert check("Here are pieces under ₹3000.", [A], user_text="kurtas under ₹3000") == []
    assert check("Within your ₹3,000 budget.", [B], user_text="under 3000") == []
    assert check("Between ₹1,000 and ₹2,000 you will find it.", [B], user_text="between 1000 and 2000") == []


def test_budget_echo_does_not_cover_an_item_price() -> None:
    # user said 3000; the reply attributes 3000 to an item priced 1499
    assert kinds("The Gloss Wrap Skirt is ₹3,000.", [B], user_text="under ₹3000") == ["price"]


def test_budget_echo_requires_the_user_to_have_said_it() -> None:
    assert kinds("Here are pieces under ₹3000.", [B], user_text="show me kurtas") == ["price"]


def test_budget_echo_matches_exact_amount_only() -> None:
    assert kinds("Here are pieces under ₹4000.", [B], user_text="under 3000") == ["price"]


# ----------------------------------------------------------------- percent / discount


@pytest.mark.parametrize("text", [
    "Get 30% off today!", "Flat 50 percent off.", "Up to 70% off everything.",
    "It is 15% cheaper than the other one.", "Save 25% on this.", "A 20% discount applies.",
])
def test_percent_discounts_flagged(text: str) -> None:
    assert "percent" in kinds(text, [A])


def test_material_percent_is_not_a_discount() -> None:
    assert check("It is 100% cotton and 5% elastane.", [A]) == []


def test_percent_matching_item_description_passes() -> None:
    item = dict(A, detail_desc="Fit: 95% viscose, 5% spandex")
    assert check("A 95% viscose blend.", [item]) == []


@pytest.mark.parametrize("text", [
    "It is on sale.", "The mid-season sale is on.", "Use the coupon at checkout.",
    "A great deal on this one!", "Clearance prices!", "Cashback of a lifetime.",
    "Save ₹500 on this.", "Buy 2 get 1 free.", "A special offer for you.",
])
def test_discount_language_flagged(text: str) -> None:
    assert "discount" in kinds(text, [A])


def test_wholesale_and_substrings_do_not_trigger() -> None:
    assert check("Wholesale cotton blends, and a resale-friendly cut.", [A]) == []


def test_discount_backed_by_item_fields() -> None:
    assert check("The Slim Chinos are 20% off, now ₹1,599 instead of ₹1,999.", [FUTURE]) == []
    assert "percent" in kinds("The Slim Chinos are 35% off.", [FUTURE])


def test_discount_pct_derived_from_mrp() -> None:
    item = {"display_name": "Kurta Pro", "price_inr": 800.0, "mrp_inr": 1000.0}
    assert check("Kurta Pro is 20% off.", [item]) == []
    assert "percent" in kinds("Kurta Pro is 30% off.", [item])


# ----------------------------------------------------------------- sizes


@pytest.mark.parametrize("text", [
    "Available in size M.", "Sizes S, M and L are available.", "It comes in UK 8.",
    "Size: 32", "XL pieces sell out.", "Order a size up.", "It runs small.", "True to size.",
    "Pair it in size 7.", "Free size, fits all.", "Available in XXL.", "In S, M, L and XL.",
    "The model wears size S.",
])
def test_size_claims_flagged_when_no_sizes_field(text: str) -> None:
    assert "size" in kinds(text, [A])


def test_size_claims_backed_by_sizes_field() -> None:
    assert check("Available in sizes 30, 32 and 34.", [FUTURE]) == []
    assert "size" in kinds("Available in size 36.", [FUTURE])
    item = dict(A, sizes=["S", "M", "L"])
    assert check("Comes in S, M and L.", [item]) == []
    assert "size" in kinds("Comes in S, M, L and XL.", [item])


def test_free_size_backed_by_field() -> None:
    saree = dict(A, sizes=["Free Size"])
    assert check("It is a free size saree.", [saree]) == []


def test_letters_inside_words_are_not_sizes() -> None:
    assert check("The Meera set pairs with Mojari in a Maroon shade, Medium-weight Linen.", [A]) == []
    assert check("Sizzling looks in a Smart, Modern, Lightweight style.", [A]) == []


# ----------------------------------------------------------------- stock


@pytest.mark.parametrize("text,kind_", [
    ("It is in stock.", "stock"), ("Sold out, sorry.", "stock"), ("Only 3 left!", "stock"),
    ("Back in stock!", "stock"), ("Limited stock.", "stock"), ("Out of stock right now.", "stock"),
    ("Selling fast.", "stock"), ("Only 2 pieces left.", "stock"), ("The last 2 are here.", "stock"),
])
def test_stock_claims_flagged(text: str, kind_: str) -> None:
    assert kinds(text, [A]) == [kind_]


def test_stock_claims_must_match_state() -> None:
    gone = dict(A, in_stock=False)
    here = dict(A, in_stock=True, stock_count=2)
    assert check("It is sold out.", [gone]) == []
    assert kinds("It is in stock.", [gone]) == ["stock"]
    assert check("It is in stock.", [here]) == []
    assert kinds("It is sold out.", [here]) == ["stock"]
    assert check("Only 2 left.", [here]) == []
    assert kinds("Only 5 left.", [here]) == ["stock"]


# ----------------------------------------------------------------- delivery / shipping


@pytest.mark.parametrize("text", [
    "Free shipping on this one.", "It ships in 3-5 days.", "Delivered by Friday.",
    "Cash on delivery is available.", "Next-day delivery.", "Easy returns within 7 days.",
    "We dispatch today.", "Express shipping available.", "Free delivery across India.",
])
def test_delivery_claims_flagged(text: str) -> None:
    assert "delivery" in kinds(text, [A])


def test_delivery_backed_by_fields() -> None:
    assert check("Free shipping, delivered in 3-5 days.", [FUTURE]) == []
    assert "delivery" in kinds("Delivered in 9 days.", [FUTURE])
    assert "delivery" in kinds("Free shipping.", [dict(FUTURE, free_shipping=False)])


# ----------------------------------------------------------------- denial semantics


@pytest.mark.parametrize("text", [
    "I don't have size information, check the product page.",
    "I can't confirm free shipping; check the store.",
    "I don't have any discount information, so I can't say if it is on sale.",
    "Delivery details: check the product page.",
    "I'm unable to tell if it is in stock.",
    "There is no data on delivery times.",
])
def test_honest_disclaimers_pass(text: str) -> None:
    assert check(text, [A]) == []


def test_denial_covers_user_echoed_numbers_only() -> None:
    assert check("I can't promise 30% off.", [A], user_text="give me 30% off") == []
    assert "percent" in kinds("We don't have the 40% off deal.", [A], user_text="show kurtas")
    assert check("I don't have size data, so I couldn't filter by size M.", [A], user_text="size M") == []
    assert "size" in kinds("I don't have size data, so I couldn't filter by size XL.", [A],
                           user_text="size M")


def test_denial_is_clause_local() -> None:
    assert "stock" in kinds("I can't see stock levels, but it says in stock on the page.", [A])
    assert "delivery" in kinds("I don't have size data; it ships free.", [A])


def test_denial_in_another_sentence_does_not_excuse_a_claim() -> None:
    assert "stock" in kinds("I don't have sizes. It is in stock.", [A])


# ----------------------------------------------------------------- hostile / robustness


@pytest.mark.parametrize("text", ["", "   ", "\n\n", None])
def test_empty_text(text: str | None) -> None:
    assert check(text or "", [A]) == []


def test_no_items_means_every_number_is_ungrounded() -> None:
    assert kinds("It is ₹999.", []) == ["price"]
    assert kinds("I found some options for you.", []) == []


def test_items_with_missing_or_odd_fields() -> None:
    weird = [{}, {"price_inr": None}, {"price_inr": "free", "display_name": None},
             {"price_inr": True, "display_name": "X"}, {"display_name": 12345}]
    assert kinds("It is ₹999.", weird) == ["price"]


def test_unicode_and_markdown_noise() -> None:
    assert check("**Riviera Sundress** — ₹2,999 ✨ (नया संग्रह)", [A]) == []
    assert kinds("**Riviera Sundress** — ₹3,999 ✨", [A]) != []


def test_pure_and_deterministic() -> None:
    items = [copy.deepcopy(A), copy.deepcopy(B)]
    before = copy.deepcopy(items)
    text = "Get 30% off. The Riviera Sundress is ₹1,499. Ships free in 3 days. Size M. Only 2 left."
    first = check(text, items, user_text="x")
    assert check(text, items, user_text="x") == first
    assert items == before


def test_pathological_input_is_fast() -> None:
    text = ("₹1," * 2000) + ("9" * 5000) + (" size " * 2000) + ("1 - " * 3000)
    t0 = time.perf_counter()
    check(text, [A, B])
    assert time.perf_counter() - t0 < 5.0


def test_violation_spans_point_into_the_text() -> None:
    text = "Sale! The Riviera Sundress is ₹1,499 and in stock."
    for v in check(text, [A, B]):
        assert text[v.start:v.end] == v.text


# ----------------------------------------------------------------- the checker's own P/R


def test_hand_labelled_sets_have_no_disagreements() -> None:
    for path, minimum_cases in ((LABELS, 40), (HOLDOUT, 20)):
        res = measure(path)
        assert res["n_cases"] >= minimum_cases
        assert res["disagreements"] == [], res["disagreements"]
        assert res["violation_level"]["precision"] == 1.0
        assert res["violation_level"]["recall"] == 1.0


def test_near_identical_item_names_get_distinct_positions() -> None:
    # Found by the first real-agent run: "X Suit" and "X Suit 2PCS" both matched at offset 0,
    # so the first amount was attributed to the wrong item.
    one = {"display_name": "MLS Pathani Suit", "prod_name": "MLS Pathani Suit", "price_inr": 4999.0}
    two = {"display_name": "MLS Pathani Suit 2PCS", "prod_name": "MLS Pathani Suit 2PCS",
           "price_inr": 4299.0}
    text = "MLS Pathani Suit at ₹4999 is a lovely pick, and MLS Pathani Suit 2PCS (₹4299) pairs well."
    assert check(text, [one, two]) == []


def test_shared_name_fragments_do_not_create_false_mentions() -> None:
    # Found on real catalogue rows: "Floral Chintz" appears in several names and a 3-word prefix
    # ("MLS Pathani Suit") is shared; neither may steal the attribution of the price.
    nadia = {"display_name": "Nadia Floral Chintz Wrap Tunic Dress ( dress)", "price_inr": 1699.0}
    amala = {"display_name": "Amala Floral Chintz Ruffled Long Dress ( dress)", "price_inr": 1479.0}
    assert check("Nadia Floral Chintz Wrap Tunic Dress ( dress) is a good start at ₹1699.",
                 [nadia, amala]) == []
    white = {"display_name": "MLS PATHANI SUIT (White Pathani Suit)", "price_inr": 6499.0}
    one = {"display_name": "MLS PATHANI SUIT ( PATHANI SUIT)", "price_inr": 4999.0}
    two = {"display_name": "MLS PATHANI SUIT 2PCS ( PATHANI SUIT)", "price_inr": 4299.0}
    text = ("MLS PATHANI SUIT ( PATHANI SUIT) at ₹4999 is a lovely pick, and "
            "MLS PATHANI SUIT 2PCS ( PATHANI SUIT) (₹4299) pairs well.")
    assert check(text, [one, two, white]) == []
    swapped = text.replace("₹4999", "₹4299 ").replace("(₹4299)", "(₹4999)")
    assert [v.kind for v in check(swapped, [one, two, white])] == ["price_wrong_item"] * 2


def test_keywords_inside_a_shown_item_name_are_not_claims() -> None:
    # Found on a real catalogue row: "Diwali Sale Other Essentials D25816 ( Diwali Sale 2025)".
    sale = {"display_name": "Diwali Sale Other Essentials D25816 ( Diwali Sale 2025)",
            "price_inr": 2999.0}
    assert check("Diwali Sale Other Essentials D25816 ( Diwali Sale 2025) is ₹2999.", [sale]) == []
    # ...but the exemption is for the name only, not for claims next to it
    assert "discount" in kinds("Diwali Sale Other Essentials D25816 ( Diwali Sale 2025) is on sale.",
                               [sale])
    assert "percent" in kinds("Diwali Sale Other Essentials D25816 ( Diwali Sale 2025) is 20% off.",
                              [sale])
    cotton = {"display_name": "Diara|100% Cotton Scallop Blouse ( blouse)", "price_inr": 899.0}
    assert check("Diara|100% Cotton Scallop Blouse ( blouse) at ₹899 is lovely.", [cotton]) == []


# ----------------------------------------------------------------- recall holes (post-verifier)
# Tuned-on cases: written after the verifier found these misses, so they pin the fix and are NOT
# an independent estimate of recall.


@pytest.mark.parametrize("text", [
    "Sirf 999 rupaye hai.",
    "Bas rupaye 999 mein mil jayega.",
    "It is 1200 rupaiya only.",
    "It is 1200 rupaiye.",
    "It is 1200 rupay.",
    "It is Rupaye 1200.",
    "It is 1200 rs.",
    "It is 1200 Rs.",
    "It is 1200 INR.",
    "It is 1200 rupees.",
    "sirf 1200 hai.",
    "The kurta? 1500 only.",
    "The kurta is 1,500 only.",
    "The kurta is 1500/- only.",
    "The kurta is 1500 /-.",
    "The kurta? 1500 sirf.",
    "The kurta is 1.5k only.",
])
def test_hinglish_and_trailing_cue_prices_are_flagged(text: str) -> None:
    assert kinds(text, [A]) == ["price"]


@pytest.mark.parametrize("text", [
    "Riviera Sundress sirf 2999 rupaye ka hai.",
    "Riviera Sundress sirf 2999 hai.",
    "The Riviera Sundress? 2999 only.",
    "The Riviera Sundress is 2999 rupaiya only.",
    "The Riviera Sundress is Rupaye 2999.",
    "The Riviera Sundress is 2,999 only.",
])
def test_hinglish_and_trailing_cue_grounded_prices_pass(text: str) -> None:
    assert check(text, [A]) == []


@pytest.mark.parametrize("text", [
    "I found 3 options only.",
    "Only 2 colours are shown.",
    "I can show 5 only.",
    "I picked 12 looks only for you.",
    "Collection of 2020 only.",
    "Order 1234567 only.",
    "That is 50% only.",
    "Only 2 pieces, sirf 2.",
])
def test_count_and_non_price_only_phrases_are_not_prices(text: str) -> None:
    assert "price" not in kinds(text, [A])
