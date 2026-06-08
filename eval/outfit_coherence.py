from __future__ import annotations

"""Outfit coherence eval — 36 anchors.

Runnable as:
    python -m eval.outfit_coherence
    python eval/outfit_coherence.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure repo root is on sys.path regardless of invocation style
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.outfit.composer import compose_outfit  # noqa: E402


# ── Synthetic item pool ─────────────────────────────────────────────────────

_POOL_DICTS: list[dict[str, Any]] = [
    # Western men
    {
        "article_id": "W001",
        "prod_name": "Navy Slim Fit Shirt",
        "product_type": "Shirt",
        "colour": "Blue",
        "index_group_name": "Menswear",
        "price_inr": 999.0,
        "detail_desc": "cotton slim fit formal shirt",
        "pdp_handle": "w001",
    },
    {
        "article_id": "W002",
        "prod_name": "White Oxford Shirt",
        "product_type": "Shirt",
        "colour": "White",
        "index_group_name": "Menswear",
        "price_inr": 1299.0,
        "detail_desc": "formal white shirt",
        "pdp_handle": "w002",
    },
    {
        "article_id": "W003",
        "prod_name": "Black Skinny Jeans",
        "product_type": "Jeans",
        "colour": "Black",
        "index_group_name": "Menswear",
        "price_inr": 1499.0,
        "detail_desc": "slim fit black jeans",
        "pdp_handle": "w003",
    },
    {
        "article_id": "W004",
        "prod_name": "Grey Blazer",
        "product_type": "Blazer",
        "colour": "Grey",
        "index_group_name": "Menswear",
        "price_inr": 2499.0,
        "detail_desc": "formal grey blazer",
        "pdp_handle": "w004",
    },
    # Western women
    {
        "article_id": "W005",
        "prod_name": "Floral Midi Dress",
        "product_type": "Dress",
        "colour": "Pink",
        "index_group_name": "Ladieswear",
        "price_inr": 1999.0,
        "detail_desc": "floral cotton midi dress",
        "pdp_handle": "w005",
    },
    {
        "article_id": "W006",
        "prod_name": "White Blouse",
        "product_type": "Blouse",
        "colour": "White",
        "index_group_name": "Ladieswear",
        "price_inr": 899.0,
        "detail_desc": "casual white blouse",
        "pdp_handle": "w006",
    },
    {
        "article_id": "W007",
        "prod_name": "Black Trousers",
        "product_type": "Trousers",
        "colour": "Black",
        "index_group_name": "Ladieswear",
        "price_inr": 1199.0,
        "detail_desc": "formal black trousers",
        "pdp_handle": "w007",
    },
    {
        "article_id": "W008",
        "prod_name": "Denim Jacket",
        "product_type": "Jacket",
        "colour": "Blue",
        "index_group_name": "Ladieswear",
        "price_inr": 1799.0,
        "detail_desc": "casual denim jacket",
        "pdp_handle": "w008",
    },
    {
        "article_id": "W009",
        "prod_name": "Red Crop Top",
        "product_type": "Top",
        "colour": "Red",
        "index_group_name": "Ladieswear",
        "price_inr": 599.0,
        "detail_desc": "casual red crop top",
        "pdp_handle": "w009",
    },
    {
        "article_id": "W010",
        "prod_name": "White Sneakers",
        "product_type": "Shoes",
        "colour": "White",
        "index_group_name": "Ladieswear",
        "price_inr": 1499.0,
        "detail_desc": "casual white sneakers",
        "pdp_handle": "w010",
    },
    # Ethnic women
    {
        "article_id": "E001",
        "prod_name": "Royal Blue Anarkali Suit",
        "product_type": "Anarkali",
        "colour": "Blue",
        "index_group_name": "Ladieswear",
        "price_inr": 3499.0,
        "detail_desc": "embellished zari anarkali festive",
        "pdp_handle": "e001",
    },
    {
        "article_id": "E002",
        "prod_name": "Pink Embroidered Lehenga",
        "product_type": "Lehenga",
        "colour": "Pink",
        "index_group_name": "Ladieswear",
        "price_inr": 7999.0,
        "detail_desc": "heavily embroidered bridal lehenga sequin zari",
        "pdp_handle": "e002",
    },
    {
        "article_id": "E003",
        "prod_name": "Mustard Silk Dupatta",
        "product_type": "Dupatta",
        "colour": "Yellow",
        "index_group_name": "Ladieswear",
        "price_inr": 799.0,
        "detail_desc": "silk dupatta ethnic accessory",
        "pdp_handle": "e003",
    },
    {
        "article_id": "E004",
        "prod_name": "Green Palazzo Pants",
        "product_type": "Palazzo",
        "colour": "Green",
        "index_group_name": "Ladieswear",
        "price_inr": 999.0,
        "detail_desc": "ethnic palazzo bottom",
        "pdp_handle": "e004",
    },
    {
        "article_id": "E005",
        "prod_name": "Pink Kurti",
        "product_type": "Kurti",
        "colour": "Pink",
        "index_group_name": "Ladieswear",
        "price_inr": 899.0,
        "detail_desc": "casual cotton kurti",
        "pdp_handle": "e005",
    },
    {
        "article_id": "E006",
        "prod_name": "Blue Cotton Kurta",
        "product_type": "Kurta",
        "colour": "Blue",
        "index_group_name": "Ladieswear",
        "price_inr": 1299.0,
        "detail_desc": "cotton printed kurta casual",
        "pdp_handle": "e006",
    },
    {
        "article_id": "E007",
        "prod_name": "Gold Juttis",
        "product_type": "Juttis",
        "colour": "Gold",
        "index_group_name": "Ladieswear",
        "price_inr": 999.0,
        "detail_desc": "ethnic juttis footwear",
        "pdp_handle": "e007",
    },
    {
        "article_id": "E008",
        "prod_name": "Red Churidar",
        "product_type": "Churidar",
        "colour": "Red",
        "index_group_name": "Ladieswear",
        "price_inr": 599.0,
        "detail_desc": "ethnic churidar bottom",
        "pdp_handle": "e008",
    },
    {
        "article_id": "E009",
        "prod_name": "Yellow Floral Lehenga",
        "product_type": "Lehenga",
        "colour": "Yellow",
        "index_group_name": "Ladieswear",
        "price_inr": 4999.0,
        "detail_desc": "cotton floral printed lightweight lehenga yellow haldi",
        "pdp_handle": "e009",
    },
    {
        "article_id": "E010",
        "prod_name": "Orange Tie-Dye Kurta",
        "product_type": "Kurta",
        "colour": "Orange",
        "index_group_name": "Ladieswear",
        "price_inr": 1199.0,
        "detail_desc": "tie-dye cotton floral kurta lightweight haldi",
        "pdp_handle": "e010",
    },
    {
        "article_id": "E011",
        "prod_name": "Turquoise Dupatta with Zari",
        "product_type": "Dupatta",
        "colour": "Turquoise",
        "index_group_name": "Ladieswear",
        "price_inr": 1499.0,
        "detail_desc": "zari embellished dupatta",
        "pdp_handle": "e011",
    },
    {
        "article_id": "E012",
        "prod_name": "Magenta Sharara",
        "product_type": "Sharara",
        "colour": "Pink",
        "index_group_name": "Ladieswear",
        "price_inr": 1799.0,
        "detail_desc": "ethnic sharara palazzo sangeet festive",
        "pdp_handle": "e012",
    },
    {
        "article_id": "E013",
        "prod_name": "Embellished Block Heels",
        "product_type": "Heels",
        "colour": "Gold",
        "index_group_name": "Ladieswear",
        "price_inr": 1599.0,
        "detail_desc": "ethnic heels juttis festive footwear",
        "pdp_handle": "e013",
    },
    {
        "article_id": "E014",
        "prod_name": "Red Embellished Saree",
        "product_type": "Saree",
        "colour": "Red",
        "index_group_name": "Ladieswear",
        "price_inr": 5999.0,
        "detail_desc": "embellished zari silk saree traditional",
        "pdp_handle": "e014",
    },
    {
        "article_id": "E015",
        "prod_name": "Jewellery Set",
        "product_type": "Jewellery",
        "colour": "Gold",
        "index_group_name": "Ladieswear",
        "price_inr": 1299.0,
        "detail_desc": "ethnic jewellery accessory gold",
        "pdp_handle": "e015",
    },
    # Ethnic men
    {
        "article_id": "M001",
        "prod_name": "Royal Blue Kurta",
        "product_type": "Kurta",
        "colour": "Blue",
        "index_group_name": "Menswear",
        "price_inr": 1299.0,
        "detail_desc": "cotton kurta festive ethnic men",
        "pdp_handle": "m001",
    },
    {
        "article_id": "M002",
        "prod_name": "Off-White Churidar",
        "product_type": "Churidar",
        "colour": "White",
        "index_group_name": "Menswear",
        "price_inr": 799.0,
        "detail_desc": "churidar pyjama ethnic bottom men",
        "pdp_handle": "m002",
    },
    {
        "article_id": "M003",
        "prod_name": "Navy Nehru Jacket",
        "product_type": "Nehru Jacket",
        "colour": "Blue",
        "index_group_name": "Menswear",
        "price_inr": 1799.0,
        "detail_desc": "nehru jacket waistcoat ethnic outerwear men",
        "pdp_handle": "m003",
    },
    {
        "article_id": "M004",
        "prod_name": "Brown Mojaris",
        "product_type": "Mojaris",
        "colour": "Brown",
        "index_group_name": "Menswear",
        "price_inr": 999.0,
        "detail_desc": "mojaris juttis ethnic footwear men",
        "pdp_handle": "m004",
    },
    {
        "article_id": "M005",
        "prod_name": "Maroon Sherwani",
        "product_type": "Sherwani",
        "colour": "Red",
        "index_group_name": "Menswear",
        "price_inr": 9999.0,
        "detail_desc": "embellished sherwani wedding formal men",
        "pdp_handle": "m005",
    },
    {
        "article_id": "M006",
        "prod_name": "Gold Pyjama",
        "product_type": "Pyjama",
        "colour": "Gold",
        "index_group_name": "Menswear",
        "price_inr": 899.0,
        "detail_desc": "ethnic pyjama churidar bottom men",
        "pdp_handle": "m006",
    },
    # Poison item — dupatta for men; must be rejected by gender gate
    {
        "article_id": "M007",
        "prod_name": "Black Ethnic Dupatta",
        "product_type": "Dupatta",
        "colour": "Black",
        "index_group_name": "Menswear",
        "price_inr": 499.0,
        "detail_desc": "dupatta ethnic",
        "pdp_handle": "m007",
    },
    # Accessories
    {
        "article_id": "A001",
        "prod_name": "Beige Clutch Bag",
        "product_type": "Clutch",
        "colour": "Beige",
        "index_group_name": "Ladieswear",
        "price_inr": 999.0,
        "detail_desc": "ethnic clutch bag accessory",
        "pdp_handle": "a001",
    },
    # Budget items
    {
        "article_id": "B001",
        "prod_name": "Yellow Cotton Dupatta",
        "product_type": "Dupatta",
        "colour": "Yellow",
        "index_group_name": "Ladieswear",
        "price_inr": 299.0,
        "detail_desc": "cotton dupatta lightweight casual",
        "pdp_handle": "b001",
    },
    {
        "article_id": "B002",
        "prod_name": "Green Palazzo",
        "product_type": "Palazzo",
        "colour": "Green",
        "index_group_name": "Ladieswear",
        "price_inr": 499.0,
        "detail_desc": "ethnic palazzo bottom budget",
        "pdp_handle": "b002",
    },
    {
        "article_id": "B003",
        "prod_name": "Blue Churidar",
        "product_type": "Churidar",
        "colour": "Blue",
        "index_group_name": "Ladieswear",
        "price_inr": 399.0,
        "detail_desc": "churidar bottom budget ethnic",
        "pdp_handle": "b003",
    },
]


def _build_catalogue_df(pool: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a minimal DataFrame from the item pool that compose_outfit can consume."""
    rows = []
    for item in pool:
        colour = item["colour"]
        pt = item["product_type"]
        ig = item["index_group_name"]
        row = {
            "article_id": item["article_id"],
            "prod_name": item["prod_name"],
            "display_name": item["prod_name"],
            "product_type_name": pt,
            "colour_group_name": colour,
            "index_group_name": ig,
            "department_name": "N/A",
            "detail_desc": item.get("detail_desc", ""),
            "price_inr": item.get("price_inr", 0.0),
            "pdp_handle": item.get("pdp_handle", ""),
            "image_url": None,
            "pdp_live": True,
            # facets dict accessed by composer._row_to_item
            "facets": {
                "colour_group_name": colour,
                "product_type_name": pt,
                "department_name": "N/A",
                "index_group_name": ig,
                "garment_group_name": "N/A",
            },
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── MiniRetriever ────────────────────────────────────────────────────────────

class MiniRetriever:
    """Lightweight mock retriever that returns pre-seeded item pools by keyword match.

    Behaves like HybridRetriever structurally (duck-typed) so compose_outfit
    can call self._pool without importing the real retrieval stack.
    """

    def __init__(self, item_pool: list[dict[str, Any]]) -> None:
        self._pool = item_pool

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Return items whose prod_name or product_type contains any query word."""
        q_words = query.lower().split()
        hits: list[dict[str, Any]] = []
        for i, item in enumerate(self._pool):
            text = (
                item.get("prod_name", "") + " " + item.get("product_type", "")
            ).lower()
            if any(w in text for w in q_words):
                scored_item = {**item, "score": 1.0 - i * 0.01}
                hits.append(scored_item)
        return hits[:top_k]


# ── Anchor case dataclass ────────────────────────────────────────────────────

@dataclass
class AnchorCase:
    """Specification for one coherence eval case."""

    case_id: int
    anchor_article_id: str
    occasion: str
    gender: str
    pass_description: str
    budget_inr: float | None = None
    expected_slots: list[str] = field(default_factory=list)
    forbidden_slots: list[str] = field(default_factory=list)
    # Keywords that must NOT appear in any complement's prod_name or product_type
    forbidden_keywords_in_complements: list[str] = field(default_factory=list)
    # {slot_name: [kw1, kw2]} — at least one kw must appear in slot item's name/type
    required_keywords_in_slot: dict[str, list[str]] = field(default_factory=dict)
    # If True, failing forbidden_keywords or forbidden_slots is a hard gate; else warn
    is_hard_gate: bool = True
    # If True, only assert no crash; don't score expected_slots
    crash_only: bool = False
    # If True, check that budget_total_inr <= budget_inr
    check_budget: bool = False


# ── Build 36 anchor cases ────────────────────────────────────────────────────

def _build_cases() -> list[AnchorCase]:
    return [
        # ── Western anchors 1–12 ────────────────────────────────────────────
        AnchorCase(
            case_id=1,
            anchor_article_id="W001",
            occasion="office",
            gender="men",
            pass_description="Men's shirt as office anchor → bottom filled; no ethnic items",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta", "kurta", "churidar"],
        ),
        AnchorCase(
            case_id=2,
            anchor_article_id="W004",
            occasion="office",
            gender="men",
            pass_description="Grey blazer (outerwear) → fills top + bottom; no dupatta",
            expected_slots=["top", "bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=3,
            anchor_article_id="W005",
            occasion="party_evening",
            gender="women",
            pass_description="Floral dress (western_one_piece) → no top/bottom slots",
            forbidden_slots=["bottom", "top"],
        ),
        AnchorCase(
            case_id=4,
            anchor_article_id="W006",
            occasion="smart_casual",
            gender="women",
            pass_description="White blouse → bottom filled; no dupatta",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=5,
            anchor_article_id="W007",
            occasion="casual",
            gender="women",
            pass_description="Black trousers (western_bottom) → top filled; no dupatta",
            expected_slots=["top"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=6,
            anchor_article_id="W008",
            occasion="smart_casual",
            gender="women",
            pass_description="Denim jacket (outerwear) → top + bottom filled",
            expected_slots=["top", "bottom"],
        ),
        AnchorCase(
            case_id=7,
            anchor_article_id="W009",
            occasion="casual",
            gender="women",
            pass_description="Red crop top → bottom filled; no dupatta/churidar",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta", "churidar"],
        ),
        AnchorCase(
            case_id=8,
            anchor_article_id="W003",
            occasion="casual",
            gender="men",
            pass_description="Black jeans (western_bottom) → top filled; no ethnic",
            expected_slots=["top"],
            forbidden_keywords_in_complements=["dupatta", "churidar"],
        ),
        AnchorCase(
            case_id=9,
            anchor_article_id="W001",
            occasion="party_evening",
            gender="men",
            pass_description="Men's shirt at party_evening → bottom filled; no dupatta",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=10,
            anchor_article_id="W002",
            occasion="office",
            gender="men",
            pass_description="White oxford shirt → bottom filled; no dupatta",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=11,
            anchor_article_id="W005",
            occasion="casual",
            gender="women",
            pass_description="Floral dress at casual → no crash; western_one_piece fills optional slots only",
            forbidden_slots=[],
        ),
        AnchorCase(
            case_id=12,
            anchor_article_id="W006",
            occasion="office",
            gender="women",
            pass_description="White blouse at office → bottom filled; no dupatta",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        # ── Ethnic women 13–24 ──────────────────────────────────────────────
        AnchorCase(
            case_id=13,
            anchor_article_id="E001",
            occasion="sangeet",
            gender="women",
            pass_description="Anarkali at sangeet (ethnic_only) → accessory+footwear; no top/bottom for one_piece; no western",
            expected_slots=["accessory", "footwear"],
            forbidden_slots=["top", "bottom"],
            forbidden_keywords_in_complements=["shirt", "jeans", "blazer", "trousers"],
        ),
        AnchorCase(
            case_id=14,
            anchor_article_id="E002",
            occasion="sangeet",
            gender="women",
            pass_description="Lehenga at sangeet → accessory+footwear; accessory must be dupatta",
            expected_slots=["accessory", "footwear"],
            forbidden_slots=["top", "bottom"],
            required_keywords_in_slot={"accessory": ["dupatta"]},
        ),
        AnchorCase(
            case_id=15,
            anchor_article_id="E006",
            occasion="festive_puja",
            gender="women",
            pass_description="Women's kurta at festive_puja → bottom+accessory; accessory=dupatta; no western",
            expected_slots=["bottom", "accessory"],
            required_keywords_in_slot={"accessory": ["dupatta"]},
            forbidden_keywords_in_complements=["shirt", "jeans", "blazer"],
        ),
        AnchorCase(
            case_id=16,
            anchor_article_id="E005",
            occasion="festive_puja",
            gender="women",
            pass_description="Women's kurti at festive_puja → bottom+accessory; accessory=dupatta",
            expected_slots=["bottom", "accessory"],
            required_keywords_in_slot={"accessory": ["dupatta"]},
        ),
        AnchorCase(
            case_id=17,
            anchor_article_id="E012",
            occasion="festive_puja",
            gender="women",
            pass_description="Sharara as anchor → top filled with kurta/kurti; not western top",
            expected_slots=["top", "accessory"],
            required_keywords_in_slot={"top": ["kurta", "kurti"]},
            forbidden_keywords_in_complements=["shirt", "blouse"],
        ),
        AnchorCase(
            case_id=18,
            anchor_article_id="E014",
            occasion="traditional_ethnic",
            gender="women",
            pass_description="Saree (ethnic_one_piece) at traditional_ethnic → accessory+footwear only; no western",
            expected_slots=["accessory", "footwear"],
            forbidden_slots=["top", "bottom"],
            forbidden_keywords_in_complements=["shirt", "jeans"],
        ),
        AnchorCase(
            case_id=19,
            anchor_article_id="E009",
            occasion="haldi_mehendi",
            gender="women",
            pass_description="Yellow floral lehenga at haldi_mehendi → accessory+footwear; no top/bottom",
            expected_slots=["accessory", "footwear"],
            forbidden_slots=["top", "bottom"],
        ),
        AnchorCase(
            case_id=20,
            anchor_article_id="E010",
            occasion="haldi_mehendi",
            gender="women",
            pass_description="Orange tie-dye kurta at haldi_mehendi → bottom+accessory; accessory=dupatta",
            expected_slots=["bottom", "accessory"],
            required_keywords_in_slot={"accessory": ["dupatta"]},
        ),
        AnchorCase(
            case_id=21,
            anchor_article_id="E001",
            occasion="wedding_guest",
            gender="women",
            pass_description="Anarkali at wedding_guest → accessory+footwear; no top/bottom; no western",
            expected_slots=["accessory", "footwear"],
            forbidden_slots=["top", "bottom"],
            forbidden_keywords_in_complements=["shirt", "jeans"],
        ),
        AnchorCase(
            case_id=22,
            anchor_article_id="E006",
            occasion="wedding_guest",
            gender="women",
            pass_description="Women's kurta at wedding_guest → bottom+accessory; accessory=dupatta; no western",
            expected_slots=["bottom", "accessory"],
            required_keywords_in_slot={"accessory": ["dupatta"]},
            forbidden_keywords_in_complements=["jeans", "trousers", "shirt"],
        ),
        AnchorCase(
            case_id=23,
            anchor_article_id="E006",
            occasion="casual",
            gender="women",
            pass_description="Women's kurta at casual → bottom+accessory (ethnic anchor fills ethnic items at EITHER occasion)",
            expected_slots=["bottom", "accessory"],
        ),
        AnchorCase(
            case_id=24,
            anchor_article_id="E006",
            occasion="office",
            gender="women",
            pass_description="Women's kurta at office → bottom+accessory (EITHER occasion, ethnic anchor)",
            expected_slots=["bottom", "accessory"],
        ),
        # ── Men's ethnic 25–29 ──────────────────────────────────────────────
        AnchorCase(
            case_id=25,
            anchor_article_id="M001",
            occasion="festive_puja",
            gender="men",
            pass_description="Men's kurta at festive_puja → bottom filled; dupatta hard-rejected",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=26,
            anchor_article_id="M001",
            occasion="festive_puja",
            gender="men",
            pass_description="Duplicate dupatta gate — confirm from second angle; no dupatta ever for men",
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=27,
            anchor_article_id="M005",
            occasion="wedding_guest",
            gender="men",
            pass_description="Sherwani at wedding_guest → bottom+footwear filled; no dupatta",
            expected_slots=["bottom", "footwear"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=28,
            anchor_article_id="M001",
            occasion="sangeet",
            gender="men",
            pass_description="Men's kurta at sangeet → bottom filled; no dupatta",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        AnchorCase(
            case_id=29,
            anchor_article_id="M001",
            occasion="traditional_ethnic",
            gender="men",
            pass_description="Men's kurta at traditional_ethnic → bottom filled; no dupatta",
            expected_slots=["bottom"],
            forbidden_keywords_in_complements=["dupatta"],
        ),
        # ── Budget constraint anchor 30 ─────────────────────────────────────
        AnchorCase(
            case_id=30,
            anchor_article_id="E006",
            occasion="sangeet",
            gender="women",
            pass_description="Women's kurta at sangeet with budget ₹5000 → total must be <= 5000",
            budget_inr=5000.0,
            expected_slots=["bottom", "accessory", "footwear"],
            check_budget=True,
        ),
        # ── Occasion-coherence anchors 31–35 ───────────────────────────────
        AnchorCase(
            case_id=31,
            anchor_article_id="W001",
            occasion="sangeet",
            gender="men",
            pass_description="Western shirt as seed at sangeet (ethnic_only) → no crash; western complements rejected by gate",
            crash_only=True,
            is_hard_gate=True,
        ),
        AnchorCase(
            case_id=32,
            anchor_article_id="E002",
            occasion="haldi_mehendi",
            gender="women",
            pass_description="Lehenga at haldi_mehendi → accessory+footwear; no top/bottom (one_piece still applies)",
            expected_slots=["accessory", "footwear"],
            forbidden_slots=["top", "bottom"],
        ),
        AnchorCase(
            case_id=33,
            anchor_article_id="E001",
            occasion="party_evening",
            gender="women",
            pass_description="Anarkali at party_evening → still one_piece; accessory+footwear; no top/bottom",
            expected_slots=["accessory", "footwear"],
            forbidden_slots=["top", "bottom"],
        ),
        AnchorCase(
            case_id=34,
            anchor_article_id="E006",
            occasion="smart_casual",
            gender="women",
            pass_description="Women's kurta at smart_casual (EITHER) → bottom+accessory",
            expected_slots=["bottom", "accessory"],
        ),
        AnchorCase(
            case_id=35,
            anchor_article_id="W004",
            occasion="wedding_guest",
            gender="men",
            pass_description="Blazer (outerwear) at wedding_guest (ETHNIC_HEAVY) → top+bottom; western formal OK for men",
            expected_slots=["top", "bottom"],
        ),
        # ── Anchor 36 — Hard dupatta gender gate ────────────────────────────
        AnchorCase(
            case_id=36,
            anchor_article_id="M001",
            occasion="festive_puja",
            gender="men",
            pass_description="HARD GATE: M007 poison dupatta in pool; must NOT appear in any men's complement",
            forbidden_keywords_in_complements=["dupatta"],
            is_hard_gate=True,
        ),
    ]


# ── Eval runner ──────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    """Outcome for a single anchor case."""

    case: AnchorCase
    hard_pass: bool
    soft_pass: bool
    notes: list[str]


def _complement_text(comp: dict[str, Any]) -> str:
    """Combine prod_name + product_type for keyword checks."""
    return (
        (comp.get("prod_name") or "") + " " + (comp.get("product_type") or "")
    ).lower()


def _run_case(
    case: AnchorCase,
    catalogue_df: pd.DataFrame,
    retriever: MiniRetriever,
) -> CaseResult:
    """Execute compose_outfit for one case and evaluate against all assertions."""
    notes: list[str] = []
    hard_pass = True
    soft_pass = True

    try:
        result = compose_outfit(
            catalogue_df,
            retriever,  # type: ignore[arg-type]  # duck-typed; no real HybridRetriever needed
            seed_article_id=case.anchor_article_id,
            occasion_slug=case.occasion,
            gender=case.gender,
            budget_inr=case.budget_inr,
        )
    except Exception as exc:
        notes.append(f"CRASH: {exc}")
        return CaseResult(case=case, hard_pass=False, soft_pass=False, notes=notes)

    if case.crash_only:
        # Case 31 — only assert no crash; complements may be empty due to ethnic_only gate
        return CaseResult(case=case, hard_pass=True, soft_pass=True, notes=["crash_only - OK"])

    complements: list[dict[str, Any]] = result.get("complements") or []
    filled_slots: set[str] = {c.get("_slot", "") for c in complements}

    # ── Hard gate: forbidden_slots ──────────────────────────────────────────
    if case.is_hard_gate:
        for slot in case.forbidden_slots:
            if slot in filled_slots:
                hard_pass = False
                notes.append(f"HARD FAIL: forbidden slot '{slot}' was filled")

    # ── Hard gate: forbidden_keywords_in_complements ────────────────────────
    if case.is_hard_gate:
        for comp in complements:
            text = _complement_text(comp)
            for kw in case.forbidden_keywords_in_complements:
                if kw.lower() in text:
                    hard_pass = False
                    notes.append(
                        f"HARD FAIL: forbidden keyword '{kw}' in complement '{comp.get('prod_name')}'"
                    )

    # ── Soft check: expected_slots present ─────────────────────────────────
    for slot in case.expected_slots:
        if slot not in filled_slots:
            soft_pass = False
            notes.append(f"WARN: expected slot '{slot}' not filled (empty_slots={result['empty_slots']})")

    # ── Soft check: required_keywords_in_slot ──────────────────────────────
    for slot, keywords in case.required_keywords_in_slot.items():
        slot_item = next((c for c in complements if c.get("_slot") == slot), None)
        if slot_item is None:
            soft_pass = False
            notes.append(f"WARN: required slot '{slot}' not filled at all")
        else:
            text = _complement_text(slot_item)
            if not any(kw.lower() in text for kw in keywords):
                soft_pass = False
                notes.append(
                    f"WARN: slot '{slot}' item '{slot_item.get('prod_name')}' "
                    f"missing required keywords {keywords}"
                )

    # ── Hard gate: budget constraint ────────────────────────────────────────
    if case.check_budget and case.budget_inr is not None:
        total = result.get("budget_total_inr") or 0.0
        if total > case.budget_inr:
            hard_pass = False
            notes.append(f"HARD FAIL: budget_total_inr={total} > budget_inr={case.budget_inr}")

    return CaseResult(case=case, hard_pass=hard_pass, soft_pass=soft_pass, notes=notes)


def _run_eval() -> None:
    pool = _POOL_DICTS
    catalogue_df = _build_catalogue_df(pool)

    # Build retriever item pool: convert each item to the shape compose_outfit's
    # _find_best_candidate expects (article_id, prod_name, product_type, colour, etc.)
    retriever_pool: list[dict[str, Any]] = []
    for item in pool:
        retriever_pool.append({
            "article_id": item["article_id"],
            "prod_name": item["prod_name"],
            "product_type": item["product_type"],
            "colour": item["colour"],
            "index_group_name": item["index_group_name"],
            "detail_desc": item.get("detail_desc", ""),
            "price_inr": item.get("price_inr", 0.0),
            "pdp_handle": item.get("pdp_handle", ""),
            "image_url": None,
        })

    retriever = MiniRetriever(retriever_pool)
    cases = _build_cases()

    results: list[CaseResult] = []
    for case in cases:
        r = _run_case(case, catalogue_df, retriever)
        results.append(r)

    # ── Print table ─────────────────────────────────────────────────────────
    LINE = "-" * 72
    print(f"\nCoherence eval - {len(cases)} anchors")
    print(LINE)
    print(f" {'#':>2}  {'Anchor':<28} {'Occasion':<18} {'Gender':<7} {'Result':<7} Notes")
    print(LINE)

    hard_total = sum(1 for r in results if r.case.is_hard_gate)
    hard_passed = 0
    soft_total = len(results)
    soft_passed = 0

    for r in results:
        anchor_name = _anchor_display_name(r.case.anchor_article_id)
        hard_label = "PASS" if r.hard_pass else "FAIL"
        note_str = "; ".join(r.notes) if r.notes else ""
        # Truncate note for table width
        if len(note_str) > 50:
            note_str = note_str[:47] + "..."
        print(
            f" {r.case.case_id:>2}  {anchor_name:<28} {r.case.occasion:<18} "
            f"{r.case.gender:<7} {hard_label:<7} {note_str}"
        )
        if r.case.is_hard_gate and r.hard_pass:
            hard_passed += 1
        if r.soft_pass:
            soft_passed += 1

    print(LINE)

    hard_pct = 100.0 * hard_passed / hard_total if hard_total else 100.0
    soft_pct = 100.0 * soft_passed / soft_total if soft_total else 100.0
    overall_pass = hard_pct >= 90.0

    print(f"Hard gates:  {hard_passed}/{hard_total}  ({hard_pct:.0f}%)")
    print(f"Soft checks: {soft_passed}/{soft_total}  ({soft_pct:.0f}%)")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}  (hard gates >= 90%)\n")

    sys.exit(0 if overall_pass else 1)


def _anchor_display_name(article_id: str) -> str:
    """Resolve a short display name from article_id for the report table."""
    lookup = {item["article_id"]: item["prod_name"] for item in _POOL_DICTS}
    name = lookup.get(article_id, article_id)
    # Truncate to 28 chars for table alignment
    return name[:28] if len(name) > 28 else name


# ── Unit tests ───────────────────────────────────────────────────────────────

def test_mini_retriever_keyword_match() -> None:
    """MiniRetriever returns items matching any query word and respects top_k."""
    pool = [
        {"article_id": "X1", "prod_name": "Blue Kurta", "product_type": "Kurta"},
        {"article_id": "X2", "prod_name": "Red Jeans", "product_type": "Jeans"},
        {"article_id": "X3", "prod_name": "White Shirt", "product_type": "Shirt"},
    ]
    r = MiniRetriever(pool)
    hits = r.search("kurta jeans", top_k=5)
    ids = [h["article_id"] for h in hits]
    assert "X1" in ids, "Kurta should match 'kurta'"
    assert "X2" in ids, "Jeans should match 'jeans'"
    assert "X3" not in ids, "Shirt should not match 'kurta jeans'"

    hits_k1 = r.search("kurta jeans", top_k=1)
    assert len(hits_k1) <= 1


def test_catalogue_df_has_facets() -> None:
    """Every row in the catalogue DataFrame must have a valid facets dict."""
    df = _build_catalogue_df(_POOL_DICTS)
    for _, row in df.iterrows():
        assert isinstance(row["facets"], dict), f"Row {row['article_id']} missing facets dict"
        assert "colour_group_name" in row["facets"]
        assert "product_type_name" in row["facets"]


def test_no_dupatta_for_men() -> None:
    """Hard gate: dupatta must never appear in men's complements even when in pool."""
    pool = _POOL_DICTS
    df = _build_catalogue_df(pool)
    retriever_pool = [
        {
            "article_id": item["article_id"],
            "prod_name": item["prod_name"],
            "product_type": item["product_type"],
            "colour": item["colour"],
            "index_group_name": item["index_group_name"],
            "detail_desc": item.get("detail_desc", ""),
            "price_inr": item.get("price_inr", 0.0),
            "pdp_handle": item.get("pdp_handle", ""),
            "image_url": None,
        }
        for item in pool
    ]
    retriever = MiniRetriever(retriever_pool)
    result = compose_outfit(
        df,
        retriever,  # type: ignore[arg-type]
        seed_article_id="M001",
        occasion_slug="festive_puja",
        gender="men",
    )
    complements = result.get("complements") or []
    for comp in complements:
        text = _complement_text(comp)
        assert "dupatta" not in text, f"Dupatta appeared in men's complement: {comp.get('prod_name')}"


if __name__ == "__main__":
    _run_eval()
