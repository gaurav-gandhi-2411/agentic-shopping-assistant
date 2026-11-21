"""Judge prompt builder, label parser and the 20-item calibration set.

One call judges all candidates of one query. The judge sees a shuffled, numbered list of
title / price / category / colour / short description. It never sees the store name, the system
rank, the catalogue gender column or which variant produced the list. The permutation is a pure
function of (query_id, seed) so labels map back to article ids and every judge family gets the
SAME order (the Claude packets reuse these prompts).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evals.retrieval.common import HERE, sha
from evals.retrieval.scrub import scrub_stores

RUBRIC_PATH = HERE / "rubric.md"
SNIPPET_CHARS = 150
# Output cap: 20 labels + reason is ~120 visible tokens; gpt-oss/qwen reasoning tokens count
# against max_tokens (prod reranker hit its 150 cap for exactly that reason), so allow ample slack.
JUDGE_MAX_TOKENS = 1500
CALIBRATION_MIN_EXACT = 18
CALIBRATION_SEED = 7


@dataclass(frozen=True)
class JudgePrompt:
    """One judge call: prompt text plus the position -> article_id permutation."""

    query_id: str
    system: str
    user: str
    order: list[str]  # order[i] is the article_id shown as number i+1
    prompt_hash: str


def load_rubric() -> str:
    """The rubric doubles as the (byte-stable) system prompt."""
    return RUBRIC_PATH.read_text(encoding="utf-8").strip()


def shuffled_order(query_id: str, ids: Sequence[str], seed: int) -> list[str]:
    """Deterministic per-(query, seed) shuffle of candidate ids."""
    out = list(ids)
    random.Random(f"{seed}:{query_id}").shuffle(out)
    return out


def _fmt_candidate(n: int, c: Mapping[str, Any]) -> str:
    price = c.get("price_inr")
    price_s = f"₹{price:.0f}" if isinstance(price, (int, float)) else "price n/a"
    # Store names are scrubbed from title and description (they carry the brand); the
    # description is scrubbed BEFORE truncation so a cut-off brand fragment cannot survive.
    desc = " ".join((c.get("detail_desc") or c.get("snippet") or "").split())
    desc = scrub_stores(desc)[:SNIPPET_CHARS]
    parts = [
        f"{n}. {scrub_stores(c.get('display_name') or c.get('title') or '')}",
        f"price {price_s}",
        f"category {c.get('product_type') or 'n/a'}",
        f"colour {c.get('colour') or 'n/a'}",
    ]
    return " | ".join(parts) + (f" | {desc}" if desc else "")


def _fmt_constraints(constraints: Mapping[str, Any]) -> str:
    c = constraints
    bits = [
        f"gender: {c.get('gender') or 'not stated'}",
        f"colour: {c.get('colour') or 'not stated'}",
        f"category: {c.get('category') or 'not stated'}",
        f"max price: {('₹%d' % c['max_price']) if c.get('max_price') else 'not stated'}",
    ]
    return "; ".join(bits)


def build_judge_prompt(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    seed: int,
) -> JudgePrompt:
    """Build the single judge prompt for one query (all candidates at once)."""
    by_id = {str(c["article_id"]): c for c in candidates}
    order = shuffled_order(case["id"], list(by_id), seed)
    lines = "\n".join(_fmt_candidate(i, by_id[a]) for i, a in enumerate(order, 1))
    user = (
        f'Shopper query: "{case["text"]}"\n'
        f"Expected constraints (for reference): {_fmt_constraints(case.get('constraints', {}))}\n\n"
        f"Candidates:\n{lines}\n\n"
        f"Label all {len(order)} candidates. JSON only."
    )
    system = load_rubric()
    return JudgePrompt(case["id"], system, user, order, sha(system + "\n" + user))


def parse_labels(text: str, n: int) -> dict[int, int] | None:
    """Parse `{"labels": {"1": 2, ...}}`; None unless all 1..n are present with values 0/1/2."""
    blobs = [text.strip()]
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        blobs.append(m.group())
    for blob in blobs:
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        raw = data.get("labels") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            continue
        try:
            labels = {int(k): int(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            return None
        if set(labels) == set(range(1, n + 1)) and all(v in (0, 1, 2) for v in labels.values()):
            return labels
        return None
    return None


def labels_to_ids(labels: Mapping[int, int], order: Sequence[str]) -> dict[str, int]:
    """Map position-numbered labels back to article ids through the stored permutation."""
    return {order[pos - 1]: lab for pos, lab in labels.items()}


# --------------------------------------------------------------------------------------
# Calibration: 4 synthetic queries x 5 hand-labelled candidates = 20 unambiguous items.
# Built by construction from the rubric (wrong gender/category -> 0, exact match -> 2, a single
# colour miss or a <=25% budget overrun -> 1). Public product wording only.
# --------------------------------------------------------------------------------------
def _cand(aid: str, title: str, price: int, cat: str, colour: str, desc: str) -> dict[str, Any]:
    return {
        "article_id": aid,
        "display_name": title,
        "price_inr": float(price),
        "product_type": cat,
        "colour": colour,
        "detail_desc": desc,
    }


_CAL_RAW: list[tuple[dict[str, Any], list[tuple[dict[str, Any], int]]]] = [
    (
        {
            "id": "cal1",
            "text": "men's black kurta set for a wedding under ₹3000",
            "constraints": {
                "gender": "men",
                "colour": "black",
                "category": "kurta set",
                "max_price": 3000,
            },
        },
        [
            (
                _cand(
                    "c1a",
                    "Men's Black Cotton Silk Kurta Pyjama Set",
                    2499,
                    "Kurta set",
                    "Black",
                    "Festive straight kurta with matching pyjama for men.",
                ),
                2,
            ),
            (
                _cand(
                    "c1b",
                    "Women's Red Banarasi Silk Saree with Blouse",
                    5999,
                    "Saree",
                    "Red",
                    "Woven silk saree with unstitched blouse piece.",
                ),
                0,
            ),
            (
                _cand(
                    "c1c",
                    "Men's Maroon Embroidered Kurta Pyjama Set",
                    2299,
                    "Kurta set",
                    "Maroon",
                    "Embroidered festive kurta set for men.",
                ),
                1,
            ),
            (
                _cand(
                    "c1d",
                    "Men's Navy Slim Fit Cotton Casual Shirt",
                    1299,
                    "Shirt",
                    "Navy",
                    "Everyday casual shirt with button-down collar.",
                ),
                0,
            ),
            (
                _cand(
                    "c1e",
                    "Men's Black Kurta Set with Nehru Jacket",
                    2799,
                    "Kurta set",
                    "Black",
                    "Three-piece festive set with a matching jacket for men.",
                ),
                2,
            ),
        ],
    ),
    (
        {
            "id": "cal2",
            "text": "women's yellow anarkali for haldi under ₹2000",
            "constraints": {
                "gender": "women",
                "colour": "yellow",
                "category": "anarkali",
                "max_price": 2000,
            },
        },
        [
            (
                _cand(
                    "c2a",
                    "Women's Yellow Cotton Anarkali Kurta Set",
                    1799,
                    "Anarkali",
                    "Yellow",
                    "Flared anarkali kurta with dupatta.",
                ),
                2,
            ),
            (
                _cand(
                    "c2b",
                    "Women's Yellow Floral Anarkali Kurta Set",
                    2399,
                    "Anarkali",
                    "Yellow",
                    "Printed flared anarkali kurta with dupatta.",
                ),
                1,
            ),
            (
                _cand(
                    "c2c",
                    "Men's Yellow Cotton Kurta",
                    999,
                    "Kurta",
                    "Yellow",
                    "Straight cotton kurta for men.",
                ),
                0,
            ),
            (
                _cand(
                    "c2d",
                    "Women's Black Leather Ankle Boots",
                    3499,
                    "Boots",
                    "Black",
                    "Block heel ankle boots.",
                ),
                0,
            ),
            (
                _cand(
                    "c2e",
                    "Women's Green Georgette Anarkali Kurta Set",
                    1899,
                    "Anarkali",
                    "Green",
                    "Flared georgette anarkali kurta with dupatta.",
                ),
                1,
            ),
        ],
    ),
    (
        {
            "id": "cal3",
            "text": "formal shirt for men for office",
            "constraints": {
                "gender": "men",
                "colour": None,
                "category": "formal shirt",
                "max_price": None,
            },
        },
        [
            (
                _cand(
                    "c3a",
                    "Men's White Slim Fit Cotton Formal Shirt",
                    1499,
                    "Shirt",
                    "White",
                    "Full sleeve formal shirt for office wear.",
                ),
                2,
            ),
            (
                _cand(
                    "c3b",
                    "Men's Light Blue Formal Full Sleeve Shirt",
                    1799,
                    "Shirt",
                    "Light Blue",
                    "Regular fit formal shirt, wrinkle resistant.",
                ),
                2,
            ),
            (
                _cand(
                    "c3c",
                    "Women's Pink Satin Evening Gown",
                    4999,
                    "Gown",
                    "Pink",
                    "Floor-length satin gown for parties.",
                ),
                0,
            ),
            (
                _cand(
                    "c3d",
                    "Men's Graphic Print Beach Shorts",
                    899,
                    "Shorts",
                    "Blue",
                    "Quick-dry swim shorts with graphic print.",
                ),
                0,
            ),
            (
                _cand(
                    "c3e",
                    "Men's Printed Casual Short Sleeve Shirt",
                    1199,
                    "Shirt",
                    "Multi",
                    "Loud tropical print holiday shirt, relaxed fit.",
                ),
                1,
            ),
        ],
    ),
    (
        {
            "id": "cal4",
            "text": "gold jhumka earrings for sangeet under ₹1500",
            "constraints": {
                "gender": "women",
                "colour": "gold",
                "category": "jhumka earrings",
                "max_price": 1500,
            },
        },
        [
            (
                _cand(
                    "c4a",
                    "Gold-Plated Jhumka Earrings",
                    899,
                    "Earrings",
                    "Gold",
                    "Traditional gold-plated jhumka drops.",
                ),
                2,
            ),
            (
                _cand(
                    "c4b",
                    "Silver Oxidised Jhumka Earrings",
                    699,
                    "Earrings",
                    "Silver",
                    "Oxidised silver-tone jhumka drops.",
                ),
                1,
            ),
            (
                _cand(
                    "c4c",
                    "Gold-Plated Temple Necklace Set",
                    1299,
                    "Necklace set",
                    "Gold",
                    "Temple-style necklace with matching studs.",
                ),
                0,
            ),
            (
                _cand(
                    "c4d",
                    "Men's Brown Leather Belt",
                    999,
                    "Belt",
                    "Brown",
                    "Genuine leather belt with pin buckle.",
                ),
                0,
            ),
            (
                _cand(
                    "c4e",
                    "Gold-Plated Polki Jhumka Earrings",
                    2999,
                    "Earrings",
                    "Gold",
                    "Heavy polki jhumka earrings.",
                ),
                0,
            ),
        ],
    ),
]


def calibration_cases() -> list[dict[str, Any]]:
    """Calibration cases: {case, candidates, gold: {article_id: label}} (20 items in total)."""
    out = []
    for case, items in _CAL_RAW:
        out.append(
            {
                "case": case,
                "candidates": [c for c, _ in items],
                "gold": {c["article_id"]: g for c, g in items},
            }
        )
    return out


def score_calibration(pred: Mapping[str, int], gold: Mapping[str, int]) -> dict[str, Any]:
    """Exact matches, off-by-two count and pass/fail for one judge (needs >= 18/20 exact and
    no off-by-two). A missing prediction counts as a miss and as off-by-two (fail closed)."""
    exact = sum(1 for k, g in gold.items() if pred.get(k) == g)
    off2 = sum(1 for k, g in gold.items() if k not in pred or abs(pred[k] - g) == 2)
    return {
        "n": len(gold),
        "exact": exact,
        "off_by_two": off2,
        "passed": exact >= CALIBRATION_MIN_EXACT and off2 == 0,
    }
