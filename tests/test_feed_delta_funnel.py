"""scripts/ops/feed_delta_funnel.py: each build filter removes exactly what it should.

Hermetic: a hand-written five-product feed, a pytest tmp_path, no network.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import feed_delta_funnel as fdf  # noqa: E402

BRAND = SimpleNamespace(sizing_system="IN", gender_default="women")


def _product(pid: int, title: str, price: str, available: bool, body: str = "<p>nice</p>") -> dict:
    return {
        "id": pid,
        "title": title,
        "handle": f"p{pid}",
        "product_type": "Kurta",
        "vendor": "V",
        "body_html": body,
        "created_at": "2026-10-01T00:00:00+05:30",
        "variants": [{"price": price, "available": available}],
        "images": [{"src": "https://x.example/i.jpg"}],
    }


FEED = [
    _product(1, "Blue Cotton Kurta", "999.00", True),
    _product(2, "Red Cotton Kurta", "999.00", False),  # sold out
    _product(3, "Green Cotton Kurta", "0.00", True),  # unpriced
    _product(4, "Yellow Cotton Kurta", "799.00", True, body=""),  # no description
    _product(5, "Brass Ganesh Idol Statue", "1500.00", True),  # religious decor
]


def test_funnel_counts_each_filter(tmp_path: Path) -> None:
    rec = fdf.funnel(FEED, "teststore", BRAND, tmp_path)
    assert rec["feed_distinct_ids"] == 5
    assert rec["all_variants_sold_out"] == 1 and rec["no_variants"] == 0
    assert rec["after_normalize_instock_price_gt0"] == 3  # 2 sold out, 3 unpriced dropped
    assert rec["religious_decor_dropped"] == 1
    assert rec["no_description_dropped"] == 1
    assert rec["after_01_build_filters"] == 1
    assert rec["after_phaseA_hygiene_allowed_categories"] <= rec["after_01_build_filters"]


def test_load_pages_dedupes_across_pages(tmp_path: Path) -> None:
    for n, ids in ((1, [1, 2]), (2, [2, 3])):  # id 2 repeats on page 2
        body = json.dumps({"products": [_product(i, f"t{i}", "10", True) for i in ids]})
        (tmp_path / f"s__p{n:04d}.json.gz").write_bytes(gzip.compress(body.encode()))
    assert [p["id"] for p in fdf.load_pages(tmp_path, "s")] == [1, 2, 3]
    assert fdf.load_pages(tmp_path, "other") == []


def test_snapshot_overlap_splits_still_new_and_gone(tmp_path: Path) -> None:
    snap = tmp_path / "products.csv"
    snap.write_text("id\n1\n2\n99\n", encoding="utf-8")  # 1 in stock now, 2 sold out now, 99 gone
    out = fdf.snapshot_overlap(FEED, snap)
    assert out["snapshot_rows"] == 3
    assert out["snapshot_ids_still_in_stock_in_feed"] == 1
    assert out["snapshot_ids_in_feed_but_now_sold_out_or_unpriced"] == 1
    assert out["snapshot_ids_not_in_loaded_feed"] == 1
    assert out["in_stock_feed_ids_not_in_snapshot"] == 2  # ids 4 and 5
