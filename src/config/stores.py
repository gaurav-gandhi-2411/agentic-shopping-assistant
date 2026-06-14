"""Cross-store configuration for the unified multi-brand index.

Each entry covers one store slug that may appear in the unified catalogue's
``store`` column.  The config drives:

  - ``display_name``: human-readable store name shown on item cards and buy
    buttons in the frontend (e.g. "H&M", "Myntra").
  - ``pdp_url_template``: Python-format string with a single ``{handle}``
    placeholder used to expand a catalogue ``pdp_handle`` value into a
    fully-qualified product page URL.  Set to ``None`` for stores where
    ``pdp_handle`` already contains the full URL (currently Flipkart).

Deep-link expansion logic (``build_pdp_url``)
----------------------------------------------
The rules below reflect the data that was ingested into the unified
catalogue as of the Wave-5 build (2026-06-14):

  +-----------+---------------------------------------------------+
  | store     | pdp_handle format in catalogue                    |
  +===========+===================================================+
  | hm        | NULL — H&M items have no live PDP in the dataset  |
  | myntra    | relative path, e.g. "tops/brand/.../12345678/buy" |
  | flipkart  | full URL including query params                    |
  | snitch    | slug, e.g. "my-product-name-4xyz123"               |
  | fashor    | slug                                               |
  | powerlook | slug                                               |
  | virgio    | slug                                               |
  +-----------+---------------------------------------------------+

Phase F — affiliate tag config (ENV-VAR-driven)
-----------------------------------------------
Affiliate parameters are configured per-store via environment variables of
the form ``ASA_AFFILIATE_<STORE>`` where ``<STORE>`` is the uppercased store
slug (e.g. ``ASA_AFFILIATE_FLIPKART``, ``ASA_AFFILIATE_SNITCH``).

The variable value must follow the format ``<mode>:<value>``:

  - ``param:<querystring>`` — raw query-string fragment appended to the plain
    PDP URL.  ``?`` is used when the URL has no existing query string; ``&``
    is used when query params are already present (Flipkart URLs always have
    ``?pid=…`` so ``&`` is always chosen there).  Example::

        ASA_AFFILIATE_SNITCH=param:affid=asa001
        ASA_AFFILIATE_FLIPKART=param:affid=asa001&utm_source=asa

  - ``wrap:<template>`` — the plain PDP URL is URL-encoded and substituted
    into the ``{url}`` placeholder in the template.  Useful for link-redirect
    affiliate networks.  Example::

        ASA_AFFILIATE_MYNTRA=wrap:https://linksredirect.com/?pub=FAKE123&source=asa&url={url}

  - Unset / empty / unrecognised mode / ``wrap`` template missing ``{url}``
    → ``None`` (plain link; current default behaviour).

ENV is read once per process (``@functools.lru_cache`` on the parser).
To pick up new values in tests, call ``_get_affiliate_config.cache_clear()``
before each test case.

Default is PLAIN deep-links — this module's output is byte-identical to
pre-Phase-F behaviour when no ``ASA_AFFILIATE_*`` variables are set.

DO NOT add real affiliate IDs/secrets to this file or to any committed file.
"""
from __future__ import annotations

import functools
import logging
import os
import urllib.parse
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-store config table
# ---------------------------------------------------------------------------

STORE_CONFIG: dict[str, dict[str, Any]] = {
    "hm": {
        "display_name": "H&M",
        # Dormant — hm is excluded from the unified index (archival Kaggle data, no live
        # PDP/image; requeue for partner-API phase).  Entry kept as dead-code documentation
        # so build_pdp_url returns None for any stale hm rows rather than crashing.
        "pdp_url_template": None,
    },
    "myntra": {
        "display_name": "Myntra",
        # pdp_handle is a relative path like "tops/brand/…/17048614/buy"
        # Prefix with the Myntra base URL to form the full PDP link.
        "pdp_url_template": "https://www.myntra.com/{handle}",
    },
    "flipkart": {
        "display_name": "Flipkart",
        # pdp_handle for Flipkart is already a fully-qualified URL (including pid/lid
        # query params).  Setting template to None signals build_pdp_url to use the
        # handle verbatim rather than expanding it.
        "pdp_url_template": None,
    },
    "snitch": {
        "display_name": "Snitch",
        "pdp_url_template": "https://snitch.co.in/products/{handle}",
    },
    "fashor": {
        "display_name": "Fashor",
        "pdp_url_template": "https://fashor.com/products/{handle}",
    },
    "powerlook": {
        "display_name": "Powerlook",
        "pdp_url_template": "https://powerlook.in/products/{handle}",
    },
    "virgio": {
        "display_name": "Virgio",
        "pdp_url_template": "https://virgio.com/products/{handle}",
    },
}

# ---------------------------------------------------------------------------
# Affiliate config helpers
# ---------------------------------------------------------------------------

# Sentinel representing "no affiliate config" so lru_cache can distinguish
# between a cached None result and a cache miss.
_UNSET = object()


@functools.lru_cache(maxsize=32)
def _get_affiliate_config(store_slug: str) -> tuple[str, str] | None:
    """Read and parse the ``ASA_AFFILIATE_<STORE>`` env var for one store.

    Returns a ``(mode, value)`` tuple on success, or ``None`` when:
      - the env var is unset or empty,
      - the mode is not ``"param"`` or ``"wrap"``,
      - mode is ``"wrap"`` but the template does not contain ``{url}``.

    Results are cached for the lifetime of the process (env is read once).
    Call ``_get_affiliate_config.cache_clear()`` between test cases.
    """
    env_key = f"ASA_AFFILIATE_{store_slug.upper()}"
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return None

    # Split on the FIRST colon only so the value may itself contain colons.
    parts = raw.split(":", 1)
    if len(parts) != 2:
        logger.warning(
            "ASA_AFFILIATE env var has no colon separator — ignoring",
            extra={"env_key": env_key, "raw": raw},
        )
        return None

    mode, value = parts[0].strip().lower(), parts[1]

    if mode not in ("param", "wrap"):
        logger.warning(
            "ASA_AFFILIATE env var has unrecognised mode — ignoring",
            extra={"env_key": env_key, "mode": mode},
        )
        return None

    if mode == "wrap" and "{url}" not in value:
        logger.warning(
            "ASA_AFFILIATE wrap template missing {url} placeholder — ignoring",
            extra={"env_key": env_key, "value": value},
        )
        return None

    return (mode, value)


def _apply_affiliate(plain_url: str, store_slug: str) -> str:
    """Return ``plain_url`` with the store's affiliate config applied, if any.

    Falls back to ``plain_url`` unchanged when no valid config is present
    (default — output is byte-identical to pre-Phase-F behaviour).
    """
    config = _get_affiliate_config(store_slug)
    if config is None:
        return plain_url

    mode, value = config
    if mode == "param":
        sep = "&" if "?" in plain_url else "?"
        return f"{plain_url}{sep}{value}"

    # mode == "wrap"
    encoded = urllib.parse.quote(plain_url, safe="")
    return value.replace("{url}", encoded)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_store_display_name(store: str | None) -> str | None:
    """Return the human-readable display name for a store slug.

    Returns ``None`` when the slug is unknown so callers can fall back to the
    raw slug rather than showing an empty string.
    """
    if not store:
        return None
    entry = STORE_CONFIG.get(store.lower())
    return entry["display_name"] if entry else None


def build_pdp_url(store: str | None, row: dict[str, Any]) -> str | None:
    """Build a fully-qualified PDP deep-link for one catalogue row.

    Parameters
    ----------
    store:
        Store slug (e.g. ``"myntra"``, ``"snitch"``).  May be ``None`` for
        legacy per-brand rows that pre-date the unified index.
    row:
        Dict containing at minimum ``"pdp_handle"`` (may be ``None``).

    Returns
    -------
    str | None
        Full URL, or ``None`` when the store has no live PDP data or the
        row's ``pdp_handle`` is absent/null.

    Notes
    -----
    - For Flipkart, ``pdp_handle`` already contains the full URL; it is
      returned verbatim without template expansion.
    - For Myntra, ``pdp_handle`` is a relative path; it is prefixed with the
      Myntra base URL.
    - For Shopify-backed stores (snitch/fashor/powerlook/virgio), the handle
      is a product slug inserted into ``/products/{handle}``.
    - For H&M and any unknown store, ``None`` is returned because no PDP
      template is defined.
    - Affiliate tags (Phase F): set ``ASA_AFFILIATE_<STORE>`` env var to
      ``param:<querystring>`` or ``wrap:<template_with_{url}>`` to attach a
      tag.  Default is plain links (no env var = no tag).
    """
    if not store:
        return None

    handle = row.get("pdp_handle")
    if not handle:
        return None
    handle = str(handle).strip()
    if not handle:
        return None

    cfg = STORE_CONFIG.get(store.lower())
    if cfg is None:
        return None

    template = cfg.get("pdp_url_template")

    if template is None:
        # Flipkart: handle is already a full URL.
        if handle.startswith("http"):
            return _apply_affiliate(handle, store.lower())
        # Unknown store with no template and non-URL handle — no link available.
        return None

    # Standard case: expand {handle} in the template.
    url = template.replace("{handle}", handle)

    # Phase F: apply per-store affiliate config from ASA_AFFILIATE_<STORE> env var.
    return _apply_affiliate(url, store.lower())
