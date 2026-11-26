"""Incremental index builder core: fetched feeds + previous state -> new state + report.

Pure orchestration. The two expensive or I/O parts are injected: ``prepare`` (feed products -> clean
catalogue rows; the real one replays the repo's build filters) and an ``Embedder`` (text and image
vectors). Tests use fakes; nothing here touches the network. Rules, in order of application:
docs/ops/incremental-build-rules.md (committed before this code).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.refresh.gender_hygiene import AMBIGUOUS, CORRECTED, resolve_gender
from src.refresh.state import IndexState, StateRow

OK = "ok"
IMAGE_EMBED_CAP_PER_RUN = 2000  # CLIP on a CPU runner; the cold start is spread over runs
SHRINK_GUARD_RATIO = 0.5  # an "ok" store below this share of its previous rows is carried over
SHRINK_GUARD_MIN_PREVIOUS = 20  # ... but only when it had at least this many rows


class Embedder(Protocol):
    """Text and image embedding. ``embed_images`` returns None for an image it could not embed."""

    def embed_text(self, texts: list[str]) -> np.ndarray: ...

    def embed_images(self, urls: list[str]) -> list[np.ndarray | None]: ...


# prepare(slug, shopify_products) -> clean catalogue rows (article_id, prod_name, gender,
# search_text, image_url, ...) for the products that survive the build filters.
Prepare = Callable[[str, list[dict]], list[dict]]


@dataclass
class FeedStore:
    """One store's fetch outcome as the builder sees it (``products`` empty unless ok)."""

    slug: str
    status: str
    products: list[dict] = field(default_factory=list)


@dataclass
class BuildResult:
    """New state plus the report (JSON-safe dict) and the gender correction log."""

    state: IndexState
    report: dict[str, Any]


def _price(p: dict) -> float | None:
    variants = p.get("variants") or []
    try:
        v = float(variants[0].get("price")) if variants else None
    except (TypeError, ValueError):
        return None
    return v if v and v > 0 else None


def _tags(p: dict) -> list[str]:
    tags = p.get("tags") or []
    if isinstance(tags, str):  # /products.json returns a list; the admin API a comma string
        tags = tags.split(",")
    return sorted(str(t).strip() for t in tags if str(t).strip())


def item_view(p: dict) -> dict[str, Any]:
    """The hashed fields of a Shopify product, plus whether it is purchasable (in stock, priced)."""
    variants = p.get("variants") or []
    images = p.get("images") or []
    price = _price(p)
    return {
        "id": str(p.get("id", "")),
        "title": str(p.get("title", "")).strip(),
        "price": price,
        "available": any(v.get("available") for v in variants),
        "image_url": (images[0].get("src") if images else None),
        "tags": _tags(p),
        "product_type": str(p.get("product_type", "")).strip(),
    }


def content_hash(item: dict[str, Any]) -> str:
    """sha256 of [title, price, available, image_url, sorted tags] (rules doc section 1)."""
    payload = [item["title"], item["price"], item["available"], item["image_url"], item["tags"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _jsonable(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, np.generic):
        return _jsonable(v.item())
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _clean_row(row: dict) -> dict:
    return {str(k): _jsonable(v) for k, v in row.items()}


def _empty_store_report(slug: str, status: str) -> dict[str, Any]:
    return {
        "slug": slug,
        "status": status,
        "fetched": 0,
        "kept": 0,
        "unchanged": 0,
        "new": 0,
        "changed": 0,
        "dropped": 0,
        "carried_over": 0,
        "carry_reason": "",
        "gender_corrected": 0,
        "gender_ambiguous": 0,
        "texts_embedded": 0,
        "images_embedded": 0,
        "images_deferred": 0,
        "images_failed": 0,
    }


def build_incremental(
    feeds: list[FeedStore],
    prev: IndexState,
    prepare: Prepare,
    embedder: Embedder,
    *,
    image_cap: int = IMAGE_EMBED_CAP_PER_RUN,
    clock: Callable[[], float] = time.monotonic,
) -> BuildResult:
    """Produce the new index state from ``feeds`` and the previous state ``prev``.

    Stores are processed in slug order, the image cap is global and the image work list is ordered
    by (slug, article id), so the same inputs always embed the same items.
    """
    t0 = clock()
    feed_by_slug = {f.slug: f for f in feeds}
    prev_by_store = prev.by_store()
    slugs = sorted(set(feed_by_slug) | set(prev_by_store))
    out_rows: list[StateRow] = []
    status: dict[str, str] = {}
    stores: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    cap_left = image_cap
    cap_hit = False

    for slug in slugs:
        feed = feed_by_slug.get(slug)
        st_status = feed.status if feed else "not_attempted"
        rep = _empty_store_report(slug, st_status)
        prev_rows = prev_by_store.get(slug, {})
        stores.append(rep)
        status[slug] = st_status

        def carry(reason: str) -> None:
            rep["carried_over"] = len(prev_rows)
            rep["kept"] = len(prev_rows)
            rep["carry_reason"] = reason
            out_rows.extend(prev_rows[k] for k in sorted(prev_rows))

        if feed is None or feed.status != OK:
            carry(st_status)
            continue

        items = {}
        for p in feed.products:
            it = item_view(p)
            items.setdefault(it["id"], it)
        rep["fetched"] = len(items)
        candidates = {i: it for i, it in items.items() if it["available"] and it["price"]}
        hashes = {i: content_hash(it) for i, it in candidates.items()}
        unchanged = {
            i for i, h in hashes.items() if i in prev_rows and prev_rows[i].content_hash == h
        }
        todo = [
            p
            for p in feed.products
            if str(p.get("id", "")) in candidates and str(p.get("id", "")) not in unchanged
        ]
        prepared = {str(r["article_id"]): _clean_row(r) for r in prepare(slug, todo)}
        kept_ids = unchanged | set(prepared)
        if len(prev_rows) >= SHRINK_GUARD_MIN_PREVIOUS and (
            len(kept_ids) < SHRINK_GUARD_RATIO * len(prev_rows)
        ):
            carry("shrink_guard")
            continue

        # Gender correction on new/changed rows only (stored rows were corrected when built).
        for aid in sorted(prepared):
            row, it = prepared[aid], candidates[aid]
            d = resolve_gender(
                row.get("prod_name"), row.get("gender"), it["product_type"], it["tags"]
            )
            if d.outcome == CORRECTED:
                row["gender"] = d.after
                rep["gender_corrected"] += 1
                corrections.append(
                    {
                        "article_id": aid,
                        "store": slug,
                        "name": row.get("prod_name"),
                        "before": d.before,
                        "after": d.after,
                        "reason": d.reason,
                    }
                )
            elif d.outcome == AMBIGUOUS:
                rep["gender_ambiguous"] += 1

        new_ids = sorted(set(prepared) - set(prev_rows))
        changed_ids = sorted(set(prepared) & set(prev_rows))
        rep.update(
            unchanged=len(unchanged),
            new=len(new_ids),
            changed=len(changed_ids),
            kept=len(kept_ids),
            dropped=len(set(prev_rows) - kept_ids),
        )

        # Text vectors for new/changed rows (uncapped, local model).
        ids_to_embed = sorted(prepared)
        text_vecs: dict[str, np.ndarray] = {}
        if ids_to_embed:
            mat = embedder.embed_text(
                [str(prepared[a].get("search_text") or "") for a in ids_to_embed]
            )
            text_vecs = dict(zip(ids_to_embed, mat, strict=True))
            rep["texts_embedded"] = len(ids_to_embed)

        # Assemble rows; decide which still need an image vector.
        store_rows: dict[str, StateRow] = {}
        for aid in kept_ids:
            if aid in unchanged:
                store_rows[aid] = prev_rows[aid]
                continue
            row, old = prepared[aid], prev_rows.get(aid)
            same_image = old is not None and old.row.get("image_url") == row.get("image_url")
            store_rows[aid] = StateRow(
                store=slug,
                article_id=aid,
                content_hash=hashes[aid],
                row=row,
                text_vec=np.asarray(text_vecs[aid], dtype=np.float32),
                image_vec=old.image_vec if same_image and old is not None else None,
                image_pending=bool(old.image_pending) if same_image and old is not None else True,
            )
        work = sorted(
            a
            for a, r in store_rows.items()
            if r.image_pending and r.image_vec is None and r.row.get("image_url")
        )
        for a, r in store_rows.items():  # no image URL -> nothing to embed, never pending
            if not r.row.get("image_url"):
                r.image_pending = False
        take, deferred = work[:cap_left], work[cap_left:]
        if deferred:
            cap_hit = True
        if take:
            vecs = embedder.embed_images([str(store_rows[a].row["image_url"]) for a in take])
            cap_left -= len(take)
            for a, v in zip(take, vecs, strict=True):
                r = store_rows[a]
                if v is None:
                    rep["images_failed"] += 1
                else:
                    r.image_vec, r.image_pending = np.asarray(v, dtype=np.float32), False
                    rep["images_embedded"] += 1
        rep["images_deferred"] = len(deferred)
        out_rows.extend(store_rows[k] for k in sorted(store_rows))

    keys = (
        "fetched",
        "kept",
        "unchanged",
        "new",
        "changed",
        "dropped",
        "carried_over",
        "gender_corrected",
        "gender_ambiguous",
        "texts_embedded",
        "images_embedded",
        "images_deferred",
        "images_failed",
    )
    report: dict[str, Any] = {
        "runtime_s": round(clock() - t0, 3),
        "image_cap": image_cap,
        "cap_hit": cap_hit,
        "totals": {k: sum(s[k] for s in stores) for k in keys},
        "carried_over_stores": {s["slug"]: s["carry_reason"] for s in stores if s["carry_reason"]},
        "stores": stores,
        "gender_corrections": corrections,
    }
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return BuildResult(IndexState(out_rows, status, built_at), report)


def to_markdown(report: dict[str, Any]) -> str:
    """Human-readable report: per-store table, totals, carry-overs, first gender corrections."""
    cols = (
        "fetched",
        "kept",
        "new",
        "changed",
        "dropped",
        "carried_over",
        "gender_corrected",
        "images_embedded",
        "images_deferred",
    )
    lines = [
        "# Incremental build report",
        "",
        f"- runtime_s: {report['runtime_s']}",
        f"- image cap: {report['image_cap']} (cap hit: {report['cap_hit']})",
        "- totals: " + ", ".join(f"{k}={v}" for k, v in report["totals"].items()),
        "",
        "| store | status | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for s in report["stores"]:
        lines.append(
            f"| {s['slug']} | {s['status']} | " + " | ".join(str(s[c]) for c in cols) + " |"
        )
    if report["carried_over_stores"]:
        lines += ["", "## Carried over untouched", ""]
        lines += [f"- {k}: {v}" for k, v in report["carried_over_stores"].items()]
    lines += ["", f"## Gender corrections ({len(report['gender_corrections'])}; first 20)", ""]
    for c in report["gender_corrections"][:20]:
        lines.append(
            f"- {c['store']} {c['article_id']}: {c['before']} -> {c['after']} "
            f"({c['reason']}) {c['name']}"
        )
    return "\n".join(lines) + "\n"


def index_arrays(state: IndexState) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    """State -> (catalogue frame, text matrix, ids with an image vector, image matrix).

    Row order matches the text matrix; the image matrix is aligned with the returned id list.
    """
    cat = pd.DataFrame([r.row for r in state.rows])
    text = (
        np.stack([r.text_vec for r in state.rows]) if state.rows else np.zeros((0, 0), np.float32)
    )
    with_img = [r for r in state.rows if r.image_vec is not None]
    img = np.stack([r.image_vec for r in with_img]) if with_img else np.zeros((0, 0), np.float32)
    return cat, text, [r.article_id for r in with_img], img
