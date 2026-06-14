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

Extension point for Phase F — affiliate tags
--------------------------------------------
Phase F will attach store-specific affiliate parameters to each deep-link.
To add a tag for a store:

  1. Add an ``affiliate_param`` key to the store's entry in ``STORE_CONFIG``
     below (e.g. ``affiliate_param: "aff_id=ASA001"``).
  2. Implement the appending logic inside ``build_pdp_url``.  Do NOT add
     affiliate logic here yet — wait for the owner to confirm the parameters
     and consent/disclosure approach for each store.

DO NOT add affiliate parameters in this file without owner sign-off on
disclosure obligations for each marketplace.
"""
from __future__ import annotations

from typing import Any

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
        # Phase F: affiliate_param = None  # e.g. "utm_source=asa&utm_medium=app"
    },
    "myntra": {
        "display_name": "Myntra",
        # pdp_handle is a relative path like "tops/brand/…/17048614/buy"
        # Prefix with the Myntra base URL to form the full PDP link.
        "pdp_url_template": "https://www.myntra.com/{handle}",
        # Phase F: affiliate_param = None  # e.g. "af_channel=agentic_stylist"
    },
    "flipkart": {
        "display_name": "Flipkart",
        # pdp_handle for Flipkart is already a fully-qualified URL (including pid/lid
        # query params).  Setting template to None signals build_pdp_url to use the
        # handle verbatim rather than expanding it.
        "pdp_url_template": None,
        # Phase F: affiliate_param = None  # e.g. Flipkart affiliate code in query param
    },
    "snitch": {
        "display_name": "Snitch",
        "pdp_url_template": "https://snitch.co.in/products/{handle}",
        # Phase F: affiliate_param = None
    },
    "fashor": {
        "display_name": "Fashor",
        "pdp_url_template": "https://fashor.com/products/{handle}",
        # Phase F: affiliate_param = None
    },
    "powerlook": {
        "display_name": "Powerlook",
        "pdp_url_template": "https://powerlook.in/products/{handle}",
        # Phase F: affiliate_param = None
    },
    "virgio": {
        "display_name": "Virgio",
        "pdp_url_template": "https://virgio.com/products/{handle}",
        # Phase F: affiliate_param = None
    },
}

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

    Phase F extension point
    -----------------------
    After the template URL is built, this is the correct place to append
    per-store affiliate parameters.  Add the parameter string from
    ``STORE_CONFIG[store]["affiliate_param"]`` as a query-string suffix.
    Do NOT implement until the owner has confirmed disclosure obligations.
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
            return handle
        # Unknown store with no template and non-URL handle — no link available.
        return None

    # Standard case: expand {handle} in the template.
    url = template.replace("{handle}", handle)

    # Phase F: append affiliate param here when cfg.get("affiliate_param") is set.
    # Example (do NOT uncomment without owner sign-off):
    #   param = cfg.get("affiliate_param")
    #   if param:
    #       sep = "&" if "?" in url else "?"
    #       url = f"{url}{sep}{param}"

    return url
