"""Read-time rendering rules for saved-look snapshots (V3(d)(ii)).

A snapshot is a frozen copy of a board taken when it was saved and is never rewritten. At read
time, items from a store that is not public (Myntra, Flipkart, bhasinbrothers, priyaasi, and any
store missing from the registry: fail closed) are marked ``unavailable`` and lose every outbound
link, so the page can render them as "no longer available" with no Buy button.
"""

from __future__ import annotations

import copy
from typing import Any

from src.config.store_registry import is_public

# Fields that can carry an outbound link (or the handle one is built from) on a snapshot item.
_LINK_FIELDS = ("buy_url", "pdp_url", "pdp_handle")


def _item_store(item: dict[str, Any], brand: str | None) -> str | None:
    """Store slug of a snapshot item; legacy single-brand saves fall back to the look's brand."""
    store = item.get("store")
    if store:
        return str(store)
    return brand if brand and brand != "unified" else None


def render_snapshot(snapshot: dict[str, Any], brand: str | None) -> dict[str, Any]:
    """Return a deep copy of ``snapshot`` with non-public items made unavailable and link-free.

    - each such item gets ``unavailable: true`` and its buy_url / pdp_url / pdp_handle nulled;
    - their entries are removed from ``item_links``;
    - ``cart_url`` is dropped when any item was made unavailable (a whole-look cart link would
      still contain the hidden item);
    - public-store items are left byte-identical. The input is never mutated.
    """
    out = copy.deepcopy(snapshot)
    items = out.get("items")
    if not isinstance(items, list):
        return out
    effective_brand = brand or (out.get("brand") if isinstance(out.get("brand"), str) else None)
    hidden: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if is_public(_item_store(item, effective_brand)):
            continue
        item["unavailable"] = True
        for field in _LINK_FIELDS:
            if field in item:
                item[field] = None
        hidden.add(str(item.get("article_id", "")))
    if hidden:
        links = out.get("item_links")
        if isinstance(links, list):
            out["item_links"] = [
                lk
                for lk in links
                if not (isinstance(lk, dict) and str(lk.get("article_id")) in hidden)
            ]
        out["cart_url"] = None
    return out
