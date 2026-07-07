"""Unit tests for scripts/enrich_attributes.py — sample selection, JSON parsing,
resumable cache, and the merged-frame builder.

The one test that calls the real local Ollama (``call_ollama_extract``) is
marked ``requires_ollama`` and skips by default, matching this repo's
convention (tests/test_llm.py) — no mocking, hits the real model when run
explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.enrich_attributes import (
    _append_cache,
    _load_cache,
    _parse_llm_json,
    build_enriched_frame,
    call_ollama_extract,
    process_chunk,
    select_sample,
)

# ---------------------------------------------------------------------------
# _parse_llm_json — tolerant JSON parsing of the model's reply
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_clean_json_parses(self) -> None:
        raw = '{"season": "summer", "occasion_tag": "none", "style_tag": "none", "fabric": "cotton"}'
        result = _parse_llm_json(raw)
        assert result == {"season": "summer", "occasion_tag": None, "style_tag": None, "fabric": "cotton"}

    def test_markdown_fenced_json_parses(self) -> None:
        raw = '```json\n{"season": "winter", "occasion_tag": "none", "style_tag": "none", "fabric": "none"}\n```'
        result = _parse_llm_json(raw)
        assert result["season"] == "winter"
        assert result["fabric"] is None

    def test_none_literal_string_becomes_none(self) -> None:
        raw = '{"season": "None", "occasion_tag": "null", "style_tag": "", "fabric": "cotton"}'
        result = _parse_llm_json(raw)
        assert result["season"] is None
        assert result["occasion_tag"] is None
        assert result["style_tag"] is None
        assert result["fabric"] == "cotton"

    def test_unparseable_reply_returns_all_none(self) -> None:
        result = _parse_llm_json("I'm not sure, this could be a summer dress maybe?")
        assert all(v is None for v in result.values())

    def test_missing_keys_default_to_none(self) -> None:
        result = _parse_llm_json('{"season": "summer"}')
        assert result["season"] == "summer"
        assert result["occasion_tag"] is None
        assert result["style_tag"] is None
        assert result["fabric"] is None

    def test_case_normalised_to_lowercase(self) -> None:
        result = _parse_llm_json('{"season": "Summer", "occasion_tag": "none", "style_tag": "none", "fabric": "none"}')
        assert result["season"] == "summer"


# ---------------------------------------------------------------------------
# Resumable JSONL cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_load_cache_empty_when_missing(self, tmp_path: Path) -> None:
        assert _load_cache(tmp_path / "missing.jsonl") == {}

    def test_append_then_load_roundtrips(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.jsonl"
        records = [
            {"article_id": "a1", "season": "summer", "source": {"season": "rule"}},
            {"article_id": "a2", "season": None, "source": {"season": "none"}},
        ]
        _append_cache(cache_path, records)
        loaded = _load_cache(cache_path)
        assert set(loaded.keys()) == {"a1", "a2"}
        assert loaded["a1"]["season"] == "summer"

    def test_append_is_incremental_across_calls(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.jsonl"
        _append_cache(cache_path, [{"article_id": "a1", "source": {}}])
        _append_cache(cache_path, [{"article_id": "a2", "source": {}}])
        loaded = _load_cache(cache_path)
        assert set(loaded.keys()) == {"a1", "a2"}


# ---------------------------------------------------------------------------
# select_sample — stratified + probe rows
# ---------------------------------------------------------------------------


def _toy_catalogue() -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows.append(
            {
                "article_id": f"top-{i}",
                "prod_name": f"Plain Top {i}",
                "product_type_name": "top",
                "detail_desc": "A plain everyday top with no special features.",
                "search_text": f"Plain Top {i}. top. .",
                "facets": {"product_type_name": "top"},
            }
        )
    for i in range(10):
        rows.append(
            {
                "article_id": f"dress-{i}",
                "prod_name": f"Boho Summer Dress {i}",
                "product_type_name": "dress",
                "detail_desc": "A lightweight sleeveless bohemian summer dress for casual wear.",
                "search_text": f"Boho Summer Dress {i}. dress. .",
                "facets": {"product_type_name": "dress"},
            }
        )
    return pd.DataFrame(rows)


class TestSelectSample:
    def test_sample_size_respected(self) -> None:
        df = _toy_catalogue()
        sample = select_sample(df, sample_size=15, seed=42)
        assert len(sample) <= 15

    def test_deterministic_for_fixed_seed(self) -> None:
        df = _toy_catalogue()
        s1 = select_sample(df, sample_size=15, seed=42)
        s2 = select_sample(df, sample_size=15, seed=42)
        assert sorted(s1["article_id"]) == sorted(s2["article_id"])

    def test_no_duplicate_article_ids(self) -> None:
        df = _toy_catalogue()
        sample = select_sample(df, sample_size=20, seed=42)
        assert sample["article_id"].is_unique

    def test_includes_a_summer_signal_probe(self) -> None:
        """The boho/summer dress rows should be force-included via the probe keywords."""
        df = _toy_catalogue()
        sample = select_sample(df, sample_size=40, seed=42)
        assert sample["article_id"].str.startswith("dress-").any()


# ---------------------------------------------------------------------------
# process_chunk — rules-only path (llm_client=None), no network
# ---------------------------------------------------------------------------


class TestProcessChunk:
    def test_rules_only_skips_llm_entirely(self) -> None:
        df = _toy_catalogue().head(5)
        records, stats = process_chunk(df, llm_client=None)
        assert len(records) == 5
        assert stats["llm_calls"] == 0
        assert all("source" in r for r in records)

    def test_boho_dress_gets_style_and_season_from_rules(self) -> None:
        df = _toy_catalogue()[lambda d: d["article_id"] == "dress-0"]
        records, _ = process_chunk(df, llm_client=None)
        record = records[0]
        assert record["season"] == "summer"
        assert record["style_tag"] == "boho"
        assert record["source"]["season"] == "rule"


# ---------------------------------------------------------------------------
# build_enriched_frame — merges cache into facets + search_text
# ---------------------------------------------------------------------------


class TestBuildEnrichedFrame:
    def test_merges_facets_and_search_text(self) -> None:
        df = _toy_catalogue()[lambda d: d["article_id"] == "dress-0"].reset_index(drop=True)
        cache = {
            "dress-0": {
                "article_id": "dress-0",
                "season": "summer",
                "occasion_tag": "casual",
                "style_tag": "boho",
                "fabric": None,
                "source": {"season": "rule", "occasion_tag": "rule", "style_tag": "rule", "fabric": "none"},
            }
        }
        out = build_enriched_frame(df, cache)
        assert out["facets"].iloc[0]["season"] == "summer"
        assert out["facets"].iloc[0]["style_tag"] == "boho"
        assert "summer" in out["search_text"].iloc[0]
        assert "boho" in out["search_text"].iloc[0]

    def test_row_missing_from_cache_left_unchanged(self) -> None:
        df = _toy_catalogue()[lambda d: d["article_id"] == "top-0"].reset_index(drop=True)
        original_search_text = df["search_text"].iloc[0]
        out = build_enriched_frame(df, cache={})
        assert out["search_text"].iloc[0] == original_search_text
        assert out["facets"].iloc[0].get("season") is None


# ---------------------------------------------------------------------------
# call_ollama_extract — real local Ollama, no mocking (matches tests/test_llm.py)
# ---------------------------------------------------------------------------


@pytest.mark.requires_ollama
def test_call_ollama_extract_returns_valid_shape() -> None:
    from src.catalogue.loader import load_config
    from src.llm.client import get_llm_client

    llm = get_llm_client(load_config())
    result, latency = call_ollama_extract(
        llm,
        prod_name="Lightweight Linen Summer Shirt",
        product_type_name="shirt",
        detail_desc="A breathable sleeveless linen shirt perfect for summer casual outings.",
    )
    assert set(result.keys()) == {"season", "occasion_tag", "style_tag", "fabric"}
    assert latency > 0
