"""Unit tests for scripts/fix_gender_labels.py — keyword gender inference + migration."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.fix_gender_labels import fix_brand_gender, infer_gender

# ---------------------------------------------------------------------------
# infer_gender — keyword classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Women's Top", "women"),
        ("Womens Formal Shirt", "women"),
        ("Ladies Kurti Set", "women"),
        ("Elegant Saree", "women"),
        ("Men's Shirt", "men"),
        ("Mens Formal Shirt", "men"),
        ("Boys Cargo Pants", "men"),
        ("Sherwani for Groom", "men"),
    ],
)
def test_infer_gender_matches_expected(text: str, expected: str) -> None:
    assert infer_gender(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Unisex Tee",
        "Black Cotton Blend Casual Shirt",
        "Olive Ribbed Two-Way Neck Camisole",
        "",
    ],
)
def test_infer_gender_returns_none_when_ambiguous(text: str) -> None:
    assert infer_gender(text) is None


def test_infer_gender_none_input() -> None:
    assert infer_gender(None) is None


def test_infer_gender_women_substring_does_not_leak_into_men() -> None:
    """'women' contains the literal substring 'men' — must not misclassify as men."""
    assert infer_gender("Women's Casual Top") == "women"
    assert infer_gender("Women's Casual Top") != "men"


# ---------------------------------------------------------------------------
# fix_brand_gender — migration (dry-run vs --apply)
# ---------------------------------------------------------------------------


def _make_test_catalogue(tmp_path: Path, brand: str) -> Path:
    catalogue_dir = tmp_path / brand
    catalogue_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a3", "a4"],
            "prod_name": [
                "Women's Ribbed Camisole",
                "Men's Cotton Shirt",
                "Olive Ribbed Two-Way Neck Camisole",  # ambiguous — stays unknown
                "Already Labelled Top",
            ],
            "search_text": ["", "", "", ""],
            "gender": ["unknown", "unknown", "unknown", "women"],
        }
    )
    path = catalogue_dir / "catalogue.parquet"
    df.to_parquet(path, index=False)
    return path


def test_dry_run_does_not_write_file(tmp_path: Path) -> None:
    """Default (apply=False) must leave the parquet file byte-for-byte unchanged."""
    path = _make_test_catalogue(tmp_path, "testbrand")
    before_bytes = path.read_bytes()

    fix_brand_gender("testbrand", tmp_path, apply=False)

    after_bytes = path.read_bytes()
    assert before_bytes == after_bytes


def test_dry_run_backfills_in_memory_only(tmp_path: Path) -> None:
    """The returned in-memory frame reflects the backfill even though nothing is written."""
    _make_test_catalogue(tmp_path, "testbrand")

    df = fix_brand_gender("testbrand", tmp_path, apply=False)

    row_map = dict(zip(df["article_id"], df["gender"]))
    assert row_map["a1"] == "women"
    assert row_map["a2"] == "men"
    assert row_map["a3"] == "unknown"  # ambiguous — left untouched
    assert row_map["a4"] == "women"  # already-labelled row untouched


def test_apply_writes_backfilled_gender_to_disk(tmp_path: Path) -> None:
    path = _make_test_catalogue(tmp_path, "testbrand")

    fix_brand_gender("testbrand", tmp_path, apply=True)

    written = pd.read_parquet(path)
    row_map = dict(zip(written["article_id"], written["gender"]))
    assert row_map["a1"] == "women"
    assert row_map["a2"] == "men"
    assert row_map["a3"] == "unknown"


def test_idempotent_second_apply_is_a_no_op(tmp_path: Path) -> None:
    """Running the migration twice must produce the same result (idempotent)."""
    path = _make_test_catalogue(tmp_path, "testbrand")

    fix_brand_gender("testbrand", tmp_path, apply=True)
    first_pass = pd.read_parquet(path)["gender"].tolist()

    fix_brand_gender("testbrand", tmp_path, apply=True)
    second_pass = pd.read_parquet(path)["gender"].tolist()

    assert first_pass == second_pass


def test_refuses_to_touch_unified_catalogue(tmp_path: Path) -> None:
    """Must raise rather than ever operate on the 'unified' rebuild artifact."""
    with pytest.raises(ValueError, match="unified"):
        fix_brand_gender("unified", tmp_path, apply=False)


def test_missing_catalogue_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fix_brand_gender("nonexistent_brand", tmp_path, apply=False)


def test_no_gender_column_skips_gracefully(tmp_path: Path) -> None:
    """A brand catalogue without a gender column (e.g. hm) must be skipped, not crash."""
    brand_dir = tmp_path / "hm_like"
    brand_dir.mkdir()
    df = pd.DataFrame({"article_id": ["a1"], "prod_name": ["Some Item"]})
    path = brand_dir / "catalogue.parquet"
    df.to_parquet(path, index=False)

    result = fix_brand_gender("hm_like", tmp_path, apply=False)
    assert "gender" not in result.columns
