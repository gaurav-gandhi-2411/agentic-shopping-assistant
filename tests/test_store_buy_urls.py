"""Every store in the live index must have a working Buy URL (Wave 1b V3a).

Before this change 22 of the 42 indexed stores (27,364 items, 24.3%) had no ``STORE_CONFIG``
entry, so ``build_pdp_url`` returned None and their cards rendered no Buy button.  These
tests fail if any indexed store lacks config again.

Fixtures (all offline, no secrets/PII):
  - tests/fixtures/indexed_stores.json      the 42 stores of the live index (slug, items, source)
  - tests/fixtures/store_handle_samples.json  <=3 real pdp_handle values per store
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pytest
import yaml

from src.config.stores import STORE_CONFIG, _get_affiliate_config, build_pdp_url

_FIXTURES = Path(__file__).parent / "fixtures"
_BRANDS = Path(__file__).resolve().parent.parent / "brands"

INDEXED: list[dict[str, Any]] = json.loads(
    (_FIXTURES / "indexed_stores.json").read_text(encoding="utf-8")
)["stores"]
SAMPLES: dict[str, list[dict[str, str]]] = json.loads(
    (_FIXTURES / "store_handle_samples.json").read_text(encoding="utf-8")
)["samples"]
INDEXED_SLUGS = [s["slug"] for s in INDEXED]

# Explicit exceptions to the generic Shopify URL form ``https://<host>/products/<handle>``.
# They still need config and a working URL (see test_non_shopify_urls); only the Shopify
# path-form assertions are skipped.  Both are static Kaggle dumps that leave the public index
# in V3(b).  Any other store lacking config, or not following the Shopify form, fails.
NON_SHOPIFY_FORM: dict[str, str] = {
    "myntra": "static Kaggle dump; handle is a relative path under www.myntra.com",
    "flipkart": "static Kaggle dump; handle is already a full flipkart.com URL",
}


@pytest.fixture(autouse=True)
def _no_affiliate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Plain links only: an ASA_AFFILIATE_* var in the runner must not change expectations."""
    for slug in INDEXED_SLUGS:
        monkeypatch.delenv(f"ASA_AFFILIATE_{slug.upper()}", raising=False)
    _get_affiliate_config.cache_clear()
    yield
    _get_affiliate_config.cache_clear()


def stores_without_buy_url(slugs: list[str]) -> list[str]:
    """Slugs that have no config or whose config cannot build a URL for a sample handle."""
    bad = []
    for slug in slugs:
        handle = (SAMPLES.get(slug) or [{"pdp_handle": "sample-handle"}])[0]["pdp_handle"]
        if not build_pdp_url(slug, {"pdp_handle": handle}):
            bad.append(slug)
    return bad


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------


def test_fixture_shape() -> None:
    assert len(INDEXED_SLUGS) == len(set(INDEXED_SLUGS)) == 42
    assert sum(s["items"] for s in INDEXED) == 112_425
    assert {s["source"] for s in INDEXED} == {"live_shopify", "static_dump"}
    assert set(SAMPLES) == set(INDEXED_SLUGS)
    assert all(1 <= len(v) <= 3 for v in SAMPLES.values())


def test_allow_list_is_only_static_dumps_with_reasons() -> None:
    by_slug = {s["slug"]: s for s in INDEXED}
    for slug, reason in NON_SHOPIFY_FORM.items():
        assert reason.strip()
        assert by_slug[slug]["source"] == "static_dump"
    static = {s["slug"] for s in INDEXED if s["source"] == "static_dump"}
    assert static == set(NON_SHOPIFY_FORM), "a static store is missing from the allow-list"


# ---------------------------------------------------------------------------
# Every indexed store has config and builds a valid URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", INDEXED_SLUGS)
def test_indexed_store_has_config(slug: str) -> None:
    assert slug in STORE_CONFIG, f"indexed store {slug!r} has no STORE_CONFIG entry"
    assert STORE_CONFIG[slug].get("display_name")


def test_no_indexed_store_without_buy_url() -> None:
    assert stores_without_buy_url(INDEXED_SLUGS) == []


def test_check_fails_for_unconfigured_store() -> None:
    """Mutation-style proof: a new store in the fixture without config is reported."""
    assert stores_without_buy_url([*INDEXED_SLUGS, "brand-new-store"]) == ["brand-new-store"]


def test_check_fails_when_a_store_config_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(STORE_CONFIG, "jadeblue")
    assert stores_without_buy_url(INDEXED_SLUGS) == ["jadeblue"]


_SHOPIFY_CASES = [
    (slug, row["pdp_handle"])
    for slug in INDEXED_SLUGS
    if slug not in NON_SHOPIFY_FORM
    for row in SAMPLES[slug]
]


@pytest.mark.parametrize(("slug", "handle"), _SHOPIFY_CASES)
def test_shopify_url_form(slug: str, handle: str) -> None:
    url = build_pdp_url(slug, {"pdp_handle": handle})
    assert url is not None
    parsed = urlparse(url)
    template_host = urlparse(STORE_CONFIG[slug]["pdp_url_template"]).netloc
    assert parsed.scheme == "https"
    assert parsed.netloc == template_host
    assert parsed.path == f"/products/{handle}"  # sample handles are plain slugs
    assert "//" not in url.split("://", 1)[1]
    assert " " not in url
    assert not parsed.query and not parsed.fragment


def test_non_shopify_urls() -> None:
    assert (
        build_pdp_url("myntra", {"pdp_handle": "jeans/brand/name/14030730/buy"})
        == "https://www.myntra.com/jeans/brand/name/14030730/buy"
    )
    full = "https://www.flipkart.com/x-jeans/p/itm4a6?pid=JEA1&lid=LST1"
    assert build_pdp_url("flipkart", {"pdp_handle": full}) == full
    for slug in NON_SHOPIFY_FORM:
        for row in SAMPLES[slug]:
            assert build_pdp_url(slug, row) is not None


# ---------------------------------------------------------------------------
# Handle quoting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("handle", "expected_path"),
    [
        ("plain-slug-123", "plain-slug-123"),
        ("two words", "two%20words"),
        ("tee-❤️", "tee-%E2%9D%A4%EF%B8%8F"),  # real daivik handle shape
        ("blissterry™-joggers", "blissterry%E2%84%A2-joggers"),  # real blissclub shape
        ("100%-cotton", "100%25-cotton"),  # lone % is encoded
        ("already%20encoded", "already%20encoded"),  # valid %XX is not double-encoded
        ("a?b#c", "a%3Fb%23c"),
        ("/leading-slash", "leading-slash"),
        ("  padded  ", "padded"),
    ],
)
def test_handle_quoting(handle: str, expected_path: str) -> None:
    url = build_pdp_url("snitch", {"pdp_handle": handle})
    assert url == f"https://snitch.co.in/products/{expected_path}"
    assert "//" not in url.split("://", 1)[1]
    assert unquote(urlparse(url).path).startswith("/products/")


def test_handle_of_only_slashes_has_no_url() -> None:
    assert build_pdp_url("snitch", {"pdp_handle": "///"}) is None


# ---------------------------------------------------------------------------
# Byte-identical regression for stores configured before V3a
# ---------------------------------------------------------------------------

_PRE_V3A = {
    "myntra": ("Myntra", "https://www.myntra.com/{handle}"),
    "snitch": ("Snitch", "https://snitch.co.in/products/{handle}"),
    "fashor": ("Fashor", "https://fashor.com/products/{handle}"),
    "powerlook": ("Powerlook", "https://powerlook.in/products/{handle}"),
    "virgio": ("Virgio", "https://virgio.com/products/{handle}"),
    "berrylush": ("Berrylush", "https://www.berrylush.com/products/{handle}"),
    "globalrepublic": ("Global Republic", "https://globalrepublic.in/products/{handle}"),
    "libas": ("Libas", "https://libas.in/products/{handle}"),
    "sukkhi": ("Sukkhi", "https://sukkhi.com/products/{handle}"),
    "priyaasi": ("Priyaasi", "https://priyaasi.com/products/{handle}"),
    "voylla": ("Voylla", "https://www.voylla.com/products/{handle}"),
    "ishhaara": ("Ishhaara", "https://ishhaara.com/products/{handle}"),
    "theamethyststore": ("The Amethyst Store", "https://theamethyststore.com/products/{handle}"),
    "southtemplejewellery": (
        "South Temple Jewellery",
        "https://southtemplejewellery.com/products/{handle}",
    ),
    "daivik": ("Daivik", "https://daivik.in/products/{handle}"),
    "mortantra": ("Mortantra", "https://mortantra.com/products/{handle}"),
    "thejewelbox": ("The Jewelbox", "https://thejewelbox.in/products/{handle}"),
    "blissclub": ("BlissClub", "https://blissclub.com/products/{handle}"),
    "silvertraq": ("Silvertraq", "https://silvertraq.com/products/{handle}"),
    "campusshoes": ("Campus Shoes", "https://campusshoes.com/products/{handle}"),
}


@pytest.mark.parametrize("slug", sorted(_PRE_V3A))
def test_pre_v3a_stores_unchanged(slug: str) -> None:
    name, template = _PRE_V3A[slug]
    assert STORE_CONFIG[slug]["display_name"] == name
    assert STORE_CONFIG[slug]["pdp_url_template"] == template
    assert build_pdp_url(slug, {"pdp_handle": "some-handle-1"}) == template.replace(
        "{handle}", "some-handle-1"
    )


def test_flipkart_and_hm_unchanged() -> None:
    assert STORE_CONFIG["flipkart"]["pdp_url_template"] is None
    assert STORE_CONFIG["hm"]["pdp_url_template"] is None
    assert build_pdp_url("hm", {"pdp_handle": "x"}) is None


# ---------------------------------------------------------------------------
# Two sources of truth: brands/<slug>.yaml must agree with STORE_CONFIG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", [s for s in INDEXED_SLUGS if s != "flipkart"])
def test_brand_yaml_template_matches_store_config(slug: str) -> None:
    data = yaml.safe_load((_BRANDS / f"{slug}.yaml").read_text(encoding="utf-8"))
    assert data["pdp_url_template"] == STORE_CONFIG[slug]["pdp_url_template"]


# ---------------------------------------------------------------------------
# Buy button reaches the card (API response shape)
# ---------------------------------------------------------------------------

# snitch was configured before V3a; jadeblue / 5-elements / tjori were not.
_CARD_CASES = [
    ("snitch", "https://snitch.co.in/products/black-tee-1", "Snitch"),
    ("jadeblue", "https://jadeblue.com/products/linen-shirt-9", "JadeBlue"),
    ("5-elements", "https://5-elements.co.in/products/linen-shirt-9", "5 Elements"),
    ("tjori", "https://www.tjori.com/products/linen-shirt-9", "Tjori"),
]


@pytest.mark.parametrize(("store", "url", "display"), _CARD_CASES)
def test_chat_card_carries_buy_url(store: str, url: str, display: str) -> None:
    from api.schemas import ItemSummary

    slug = url.rsplit("/", 1)[1]
    item = {"article_id": "A1", "prod_name": "Tee", "store": store, "pdp_handle": slug}
    payload = ItemSummary.from_agent_item(item).model_dump(mode="json")
    assert payload["pdp_url"] == url
    assert payload["store_display"] == display


@pytest.mark.parametrize(("store", "url", "display"), _CARD_CASES)
def test_catalogue_card_carries_buy_url(store: str, url: str, display: str) -> None:
    from api.routes.catalogue import _row_to_item

    row = {"prod_name": "Tee", "store": store, "pdp_handle": url.rsplit("/", 1)[1]}
    payload = _row_to_item("A1", row).model_dump(mode="json")
    assert payload["pdp_url"] == url
    assert payload["store_display"] == display


def test_cross_store_cart_links_include_previously_unconfigured_store() -> None:
    from src.agents.outfit.cart_links import build_cart_action

    items = [
        {"article_id": "A1", "prod_name": "Tee", "store": "snitch", "pdp_handle": "black-tee-1"},
        {"article_id": "A2", "prod_name": "Shirt", "store": "jadeblue", "pdp_handle": "shirt-9"},
    ]
    result = build_cart_action(items, brand="unified")
    assert result["missing"] == []
    assert [link["buy_url"] for link in result["item_links"]] == [
        "https://snitch.co.in/products/black-tee-1",
        "https://jadeblue.com/products/shirt-9",
    ]


# ---------------------------------------------------------------------------
# Real index (CI only: needs data/processed/unified/catalogue.parquet)
# ---------------------------------------------------------------------------

_CATALOGUE = Path("data/processed/unified/catalogue.parquet")


@pytest.mark.requires_index
def test_real_catalogue_stores_are_indexed_and_configured() -> None:
    import pandas as pd

    if not _CATALOGUE.exists():
        pytest.skip("unified catalogue.parquet not present")
    df = pd.read_parquet(_CATALOGUE, columns=["store", "pdp_handle"])
    stores = set(df["store"].dropna().astype(str))
    assert stores <= set(INDEXED_SLUGS), (
        f"stores in the index but not in tests/fixtures/indexed_stores.json: "
        f"{sorted(stores - set(INDEXED_SLUGS))}"
    )
    assert stores <= set(STORE_CONFIG), f"no config: {sorted(stores - set(STORE_CONFIG))}"
    # every row that has a handle gets a Buy URL
    has_handle = df[df["pdp_handle"].notna() & (df["pdp_handle"].astype(str).str.strip() != "")]
    no_url = [
        (r.store, r.pdp_handle)
        for r in has_handle.itertuples()
        if not build_pdp_url(r.store, {"pdp_handle": r.pdp_handle})
    ]
    assert no_url == [], f"{len(no_url)} rows with a handle but no Buy URL, e.g. {no_url[:5]}"
