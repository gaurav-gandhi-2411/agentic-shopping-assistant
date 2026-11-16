"""The V3(b2)(b) hygiene step inside scripts/build_unified_index.py (no index is built)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

from scripts.build_unified_index import (  # noqa: E402
    _clean_brand_catalogue,
    _print_cleaning_report,
    _write_hygiene_reports,
)


def _toy_catalogue() -> pd.DataFrame:
    rows = [
        ("1", "Red Silk Kurta", "kurta"),
        ("2", "Brocade Lehenga Box", "lehenga"),  # non-fashion: dropped
        ("3", "Gold Plated Bangles Lehenga", "Bangles"),  # contradiction: quarantined
        ("4", "Kundan Jhumka Earrings", "earrings"),
    ]
    return pd.DataFrame(
        {
            "article_id": [r[0] for r in rows],
            "prod_name": [r[1] for r in rows],
            "product_type_name": [r[2] for r in rows],
            "product_group_name": "N/A",
            "graphical_appearance_name": "N/A",
            "colour_group_name": "Red",
            "department_name": "N/A",
            "index_group_name": "N/A",
            "garment_group_name": "N/A",
            "detail_desc": "desc",
            "price_inr": 999.0,
            "size_system": "IN",
            "pdp_handle": "h",
            "image_url": "u",
            "gender": "women",
            "type_confidence": "high",
            "search_text": "",
            "display_name": "",
            "facets": [{} for _ in rows],
            "store": "toy",
        }
    )


def test_clean_brand_catalogue_applies_hygiene() -> None:
    df, stats = _clean_brand_catalogue("toy", _toy_catalogue())
    assert df["article_id"].tolist() == ["1", "4"]
    assert df["category"].tolist() == ["garment", "jewellery"]
    assert stats["non_fashion_dropped"] == 1
    assert stats["quarantined"] == 1
    assert stats["rows_after_cleaning"] == 2
    assert stats["hygiene"]["dropped"]["article_id"].tolist() == ["2"]
    assert stats["hygiene"]["quarantined"]["article_id"].tolist() == ["3"]


def test_hygiene_reports_are_written_and_printed(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, stats = _clean_brand_catalogue("toy", _toy_catalogue())
    _write_hygiene_reports([stats], tmp_path / "hygiene")
    dropped = pd.read_csv(tmp_path / "hygiene" / "hygiene_dropped.csv", dtype=str)
    quarantined = pd.read_csv(tmp_path / "hygiene" / "hygiene_quarantine.csv", dtype=str)
    assert dropped["article_id"].tolist() == ["2"]
    assert quarantined["article_id"].tolist() == ["3"]
    _print_cleaning_report([stats])
    assert "non_fashion" in capsys.readouterr().out
