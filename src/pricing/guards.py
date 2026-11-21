"""Pure helpers for the live price check: config, URL guard (SSRF) and product-JSON parsing.

Nothing here does I/O. ``product_url`` is the ONLY place a request URL is built: the host comes
from the store registry (https only), the handle must be a plain slug, and a store that is not
public, not Shopify-backed or opted out of refresh yields ``None`` (never fetched).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from src.config.store_registry import get_store

USER_AGENT = (
    "StyleMaitri/1.0 (+https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant)"
)
ROBOTS_TOKEN = "StyleMaitri"
ENV_FLAG = "ASA_LIVE_PRICE_CHECK"  # 1/0 overrides config.yaml live_price_check.enabled

# Shopify handles are lowercase slugs; anything outside this alphabet (``/ ? # % : @ \`` spaces,
# unicode) is rejected rather than escaped, so a hostile handle cannot steer the request.
_HANDLE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,199}$", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$")
MAX_BODY_BYTES = 1_000_000


@dataclass(frozen=True)
class LiveConfig:
    """Tunables; defaults mirror the spec. ``enabled`` is OFF unless config turns it on."""

    enabled: bool = False
    budget_s: float = 1.5
    cache_ttl_s: float = 3600.0
    max_cards: int = 8
    min_interval_s: float = 1.0  # average spacing per domain
    burst: int = 2  # robots.txt + one product page in the same budget
    breaker_failures: int = 3
    breaker_open_s: float = 300.0
    max_retry_after_s: float = 3600.0
    cache_max_entries: int = 4096

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> LiveConfig:
        """Build from ``config.yaml`` ``live_price_check`` plus the ``ASA_LIVE_PRICE_CHECK`` env."""
        section = (config or {}).get("live_price_check") or {}
        base = cls()
        kwargs: dict[str, Any] = {}
        for name in ("budget_s", "cache_ttl_s", "max_cards", "breaker_failures", "breaker_open_s"):
            if name in section:
                kwargs[name] = type(getattr(base, name))(section[name])
        enabled = bool(section.get("enabled", base.enabled))
        env = os.environ.get(ENV_FLAG, "").strip().lower()
        if env in ("0", "false", "no"):
            enabled = False
        elif env in ("1", "true", "yes"):
            enabled = True
        return cls(enabled=enabled, **kwargs)


@dataclass(frozen=True)
class LiveResult:
    """What the store says right now. ``price_inr`` is None when it could not be read."""

    price_inr: float | None
    available: bool


def product_url(store: str, handle: str | None) -> str | None:
    """The only URL this package will ever request for a card, or None when it must not be fetched.

    None for: unknown / non-public / non-Shopify / opted-out stores, a missing domain, and any
    handle that is not a plain slug. The host always comes from the registry, never the handle.
    """
    rec = get_store(store)
    if rec is None or not rec.public or not rec.refresh_enabled or rec.source_type != "shopify":
        return None
    domain = (rec.domain or "").lower()
    if not _DOMAIN_RE.match(domain) or not handle or not isinstance(handle, str):
        return None
    handle = handle.strip()
    if not _HANDLE_RE.match(handle) or ".." in handle:
        return None
    return f"https://{domain}/products/{handle}.js"


def parse_product_js(body: bytes) -> LiveResult:
    """Parse Shopify ``/products/<handle>.js`` JSON. Raises ValueError when malformed.

    ``price`` fields there are integer paise; the indexed price is ``variants[0].price`` in rupees,
    so compare like with like. Unavailable only when the store positively says every variant is
    sold out (missing stock fields are treated as available: fail open).
    """
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("product JSON is not an object")
    variants = data.get("variants")
    if variants is not None and not isinstance(variants, list):
        raise ValueError("variants is not a list")
    variants = [v for v in (variants or []) if isinstance(v, dict)]
    raw_price = variants[0].get("price") if variants else data.get("price")
    price: float | None = None
    if raw_price is not None and not isinstance(raw_price, bool):
        try:
            price = float(raw_price) / 100.0
        except (TypeError, ValueError) as exc:
            raise ValueError("unreadable price") from exc
        if price <= 0:
            price = None
    flags = [v["available"] for v in variants if isinstance(v.get("available"), bool)]
    if flags:
        available = any(flags)
    else:
        available = data.get("available") is not False
    return LiveResult(price_inr=price, available=available)
