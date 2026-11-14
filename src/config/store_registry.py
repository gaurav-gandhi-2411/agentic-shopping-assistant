"""Store registry: the single source of truth for WHICH stores exist and what we may do with them.

``src/config/stores.py`` (``STORE_CONFIG``) answers "how do I render a Buy link for a catalogue
row"; this module answers the Wave 1b V3 questions on top of it:

  - ``public``          may the store's items reach end users?  Exactly the 38 refreshable
                        Shopify stores are public.  Myntra and Flipkart (static Kaggle dumps, no
                        refresh path), bhasinbrothers (feed 404) and priyaasi (feed unverified)
                        stay in the index for local development and evals only.
                        Source: reports/wave1b/v3a-catalogue-inventory.md section 1 and
                        reports/wave1b/v3a/inventory_by_store.csv (V3a, 2026-11-14).
  - ``refresh_enabled`` per-store OPT-OUT switch honoured by the daily refresh job: set it to
                        ``False`` (one-line PR) to stop fetching a store's feed, for example when
                        the store asks us to stop.  Defaults to ``True`` for public stores.
                        An opted-out store keeps its last good rows until they age out of the
                        next build's store set; to remove it from users at once set ``public``
                        to ``False`` as well.
  - ``feed_url_pattern`` the public product feed template (Shopify ``/products.json``).

``tests/test_store_registry.py`` fails if this table and ``STORE_CONFIG`` disagree, or if a store
of the live index (``tests/fixtures/indexed_stores.json``) is missing here.

Feed etiquette for the (future) refresh job lives here too so it is reviewed in one place:
at most ``FEED_MAX_REQUESTS_PER_SECOND`` per domain, a User-Agent that names StyleMaitri and a
contact address, robots.txt honoured, HTTP 429 honoured (``Retry-After``), opt-out via
``refresh_enabled``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

SourceType = Literal["shopify", "static_dump"]

# Public query-time filter escape hatch for local development and evals. Default (unset): only
# ``public`` stores reach users. Read at call time so tests and one-off scripts can flip it.
INCLUDE_NON_PUBLIC_ENV = "INCLUDE_NON_PUBLIC_STORES"

# Feed etiquette (see module docstring). The job itself is V3(c); these are the agreed limits.
FEED_MAX_REQUESTS_PER_SECOND = 1.0
FEED_PAGE_SIZE = 250
FEED_USER_AGENT_PRODUCT = "StyleMaitriFeedFetcher/1.0"
_SHOPIFY_FEED_PATTERN = (
    "https://{domain}/products.json?limit=" + str(FEED_PAGE_SIZE) + "&page={page}"
)


def feed_user_agent(contact: str) -> str:
    """User-Agent for feed fetches: names StyleMaitri and carries a contact address.

    ``contact`` is an e-mail address or URL a store owner can write to. It is a required
    argument (no default) so the job cannot run anonymously.
    """
    contact = contact.strip()
    if not contact:
        raise ValueError("a contact address is required in the feed User-Agent")
    return f"{FEED_USER_AGENT_PRODUCT} (+{contact})"


@dataclass(frozen=True)
class StoreRecord:
    """One store. ``domain`` is the host that answers product pages and ``/products.json``."""

    slug: str
    display_name: str
    domain: str | None
    source_type: SourceType
    public: bool
    refresh_enabled: bool = True

    @property
    def feed_url_pattern(self) -> str | None:
        """``https://<domain>/products.json?limit=250&page={page}`` for Shopify stores, else None."""
        if self.source_type != "shopify" or not self.domain:
            return None
        return _SHOPIFY_FEED_PATTERN.replace("{domain}", self.domain)

    def feed_url(self, page: int) -> str | None:
        """The feed URL for ``page`` (1-based), or None when the store has no feed."""
        pattern = self.feed_url_pattern
        return pattern.format(page=page) if pattern else None


def _shopify(slug: str, name: str, domain: str, *, public: bool = True) -> StoreRecord:
    return StoreRecord(
        slug=slug,
        display_name=name,
        domain=domain,
        source_type="shopify",
        public=public,
        refresh_enabled=public,  # a non-public store is never refreshed
    )


def _static(slug: str, name: str, domain: str | None) -> StoreRecord:
    return StoreRecord(
        slug=slug,
        display_name=name,
        domain=domain,
        source_type="static_dump",
        public=False,
        refresh_enabled=False,  # no feed: nothing to refresh
    )


_RECORDS: tuple[StoreRecord, ...] = (
    # --- the 38 public, refreshable Shopify stores ---
    _shopify("libas", "Libas", "libas.in"),
    _shopify("theamethyststore", "The Amethyst Store", "theamethyststore.com"),
    _shopify("ishhaara", "Ishhaara", "ishhaara.com"),
    _shopify("jadeblue", "JadeBlue", "jadeblue.com"),
    _shopify("snitch", "Snitch", "snitch.co.in"),
    _shopify("mortantra", "Mortantra", "mortantra.com"),
    _shopify("globalrepublic", "Global Republic", "globalrepublic.in"),
    _shopify("vastramay", "Vastramay", "vastramay.com"),
    _shopify("voylla", "Voylla", "www.voylla.com"),
    _shopify("jompers", "Jompers", "jompers.com"),
    _shopify("tjori", "Tjori", "www.tjori.com"),
    _shopify("kisah", "Kisah", "kisah.in"),
    _shopify("fashor", "Fashor", "fashor.com"),
    _shopify("campusshoes", "Campus Shoes", "campusshoes.com"),
    _shopify("mohanlalsons", "Mohanlal Sons", "www.mohanlalsons.com"),
    _shopify("virgio", "Virgio", "virgio.com"),
    _shopify("5-elements", "5 Elements", "5-elements.co.in"),
    _shopify("rathore", "Raghavendra Rathore Jodhpur", "www.rathore.com"),
    _shopify("daivik", "Daivik", "daivik.in"),
    _shopify("sukkhi", "Sukkhi", "sukkhi.com"),
    _shopify("powerlook", "Powerlook", "powerlook.in"),
    _shopify("kalpraag", "Kalpraag", "kalpraag.com"),
    _shopify("southtemplejewellery", "South Temple Jewellery", "southtemplejewellery.com"),
    _shopify("houseofvian", "House Of Vian", "www.houseofvian.com"),
    _shopify("kalapuri", "Kalapuri", "kalapuri.com"),
    _shopify("fizzygoblet", "Fizzy Goblet", "www.fizzygoblet.com"),
    _shopify("silvertraq", "Silvertraq", "silvertraq.com"),
    _shopify("coralhaze", "Coral Haze", "coralhaze.com"),
    _shopify("irasoles", "Ira Soles", "irasoles.com"),
    _shopify("taurjuttis", "Taur Juttis", "taurjuttis.com"),
    _shopify("blissclub", "BlissClub", "blissclub.com"),
    _shopify("korakari", "Korakari", "www.korakari.com"),
    _shopify("kraftojodhpur", "Krafto Jodhpur", "www.kraftojodhpur.com"),
    _shopify("thejewelbox", "The Jewelbox", "thejewelbox.in"),
    _shopify("vhaan", "Vhaan", "vhaan.in"),
    _shopify("chappers", "Chappers", "chappers.in"),
    _shopify("juttichooindia", "Juttichoo", "juttichooindia.com"),
    _shopify("shopkop", "ShopKop", "shopkop.in"),
    # --- indexed but NOT public (local development and evals only) ---
    _static("myntra", "Myntra", "www.myntra.com"),  # static Kaggle dump, unreachable for checks
    _static("flipkart", "Flipkart", None),  # static Kaggle dump; pdp_handle is a full URL
    _shopify("bhasinbrothers", "Bhasin Brothers", "bhasinbrothers.com", public=False),  # feed 404
    _shopify("priyaasi", "Priyaasi", "priyaasi.com", public=False),  # feed unverified
    # --- in the repo but not in the live index ---
    _static("hm", "H&M", None),  # archival Kaggle data, dormant
    _shopify("berrylush", "Berrylush", "www.berrylush.com", public=False),  # password-walled
)

STORE_REGISTRY: dict[str, StoreRecord] = {r.slug: r for r in _RECORDS}
if len(STORE_REGISTRY) != len(_RECORDS):  # pragma: no cover - import-time guard against typos
    raise RuntimeError("duplicate slug in store registry")


def include_non_public() -> bool:
    """True when ``INCLUDE_NON_PUBLIC_STORES`` is truthy (local development and evals)."""
    return os.environ.get(INCLUDE_NON_PUBLIC_ENV, "").strip().lower() in ("1", "true", "yes")


def get_store(slug: str | None) -> StoreRecord | None:
    """Registry record for ``slug`` (case-insensitive), or None when unknown."""
    return STORE_REGISTRY.get(slug.lower()) if slug else None


def public_stores() -> frozenset[str]:
    """Slugs of the stores whose items may reach end users."""
    return frozenset(s for s, r in STORE_REGISTRY.items() if r.public)


def refreshable_stores() -> frozenset[str]:
    """Slugs the refresh job fetches: public, Shopify, and not opted out."""
    return frozenset(
        s
        for s, r in STORE_REGISTRY.items()
        if r.public and r.refresh_enabled and r.source_type == "shopify"
    )


def get_non_public_stores() -> frozenset[str]:
    """Slugs that must NOT reach users (empty when ``INCLUDE_NON_PUBLIC_STORES`` is set).

    A store absent from the registry is treated as non-public: an unknown store must fail
    closed, so callers that see a slug outside the registry use :func:`is_public`.
    """
    if include_non_public():
        return frozenset()
    return frozenset(s for s, r in STORE_REGISTRY.items() if not r.public)


def is_public(slug: str | None) -> bool:
    """True when items of ``slug`` may reach users. Unknown or empty slug -> False (fail closed),
    except under ``INCLUDE_NON_PUBLIC_STORES`` where everything passes."""
    if include_non_public():
        return True
    rec = get_store(slug)
    return bool(rec and rec.public)
