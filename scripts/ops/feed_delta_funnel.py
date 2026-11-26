"""Feed-vs-index funnel for one store (R3 item 6a): which build filter removes how many items.

Takes the raw pages saved by ``fetch_pass.py --raw-dir`` and replays, with the repo's OWN code, the
filters that stand between a store's ``/products.json`` and the live index:

  1. distinct product ids in the feed                      (what the fetch counts report)
  2. ``download_shopify.normalize``: drop products whose variants are ALL unavailable (sold out)
     and products whose first-variant price is missing or <= 0
  3. ``01_build_retrieval``: ``adapt_feed`` + ``drop_religious_decor_items`` + drop rows with no
     description
  4. ``build_unified_index._clean_brand_catalogue`` steps: mojibake, saree reclassification,
     fabric-bolt drop, colour backfill, hygiene (non-fashion drop, quarantine) and the store's
     ``allowed_categories``

and optionally compares the in-stock ids with an older snapshot CSV (``data/raw/shopify/...``) to
split "new since the snapshot" from "gone since the snapshot". Counts only; no payload is written.

Usage:
    python scripts/ops/feed_delta_funnel.py --slug voylla --raw-dir DIR [--snapshot-csv PATH]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import download_shopify as ds  # noqa: E402

from src.catalogue.adapter import adapt_feed  # noqa: E402
from src.catalogue.cleaning import (  # noqa: E402
    backfill_colours,
    clean_mojibake_columns,
    drop_religious_decor_items,
    drop_true_fabric_material,
    reclassify_finished_sarees,
    recompute_derived_columns,
)
from src.catalogue.hygiene import apply_hygiene, restrict_to_allowed_categories  # noqa: E402


def load_pages(raw_dir: Path, slug: str) -> list[dict]:
    """Distinct products (by id, first occurrence) from ``<slug>__pNNNN.json.gz`` page files."""
    seen: set[object] = set()
    out: list[dict] = []
    for f in sorted(raw_dir.glob(f"{slug}__p*.json.gz")):
        for p in json.loads(gzip.decompress(f.read_bytes()))["products"]:
            if p.get("id") not in seen:
                seen.add(p.get("id"))
                out.append(p)
    return out


def funnel(products: list[dict], slug: str, brand: object, workdir: Path) -> dict[str, int]:
    """Per-filter counts for ``products`` (see module docstring). ``workdir`` holds one temp CSV.

    The CSV round trip mirrors the real build (``download_shopify`` writes a CSV, the build reads
    it back), so e.g. an empty description becomes NaN exactly as it does in production.
    """
    rec: dict[str, int] = {"feed_distinct_ids": len(products)}
    rec["no_variants"] = sum(1 for p in products if not (p.get("variants") or []))
    rec["all_variants_sold_out"] = sum(
        1
        for p in products
        if (p.get("variants") or []) and not any(v.get("available") for v in p["variants"])
    )
    with contextlib.redirect_stdout(io.StringIO()):  # normalize() prints a drop notice
        norm = ds.normalize(products, "")
    rec["after_normalize_instock_price_gt0"] = len(norm)
    tmp = workdir / f"_funnel_{slug}.csv"
    norm.to_csv(tmp, index=False, encoding="utf-8")
    ad = adapt_feed(pd.read_csv(tmp), brand)
    ad, rec["religious_decor_dropped"] = drop_religious_decor_items(ad)
    rec["no_description_dropped"] = int(ad["detail_desc"].isna().sum())
    ad = ad.dropna(subset=["detail_desc"]).drop_duplicates("article_id").reset_index(drop=True)
    rec["after_01_build_filters"] = len(ad)
    ad["store"] = slug
    ad, _ = clean_mojibake_columns(ad)
    ad, _ = reclassify_finished_sarees(ad)
    ad, _ = drop_true_fabric_material(ad)
    ad, _ = backfill_colours(ad)
    ad = recompute_derived_columns(ad)
    ad, _ = apply_hygiene(ad)
    ad, _ = restrict_to_allowed_categories(ad)
    rec["after_phaseA_hygiene_allowed_categories"] = len(ad)
    return rec


def snapshot_overlap(products: list[dict], snapshot_csv: Path) -> dict[str, object]:
    """Split the in-stock feed ids vs an older snapshot CSV: still there / new / gone."""
    snap = pd.read_csv(snapshot_csv, usecols=["id"])
    snap_ids = set(snap["id"].astype(str))
    with contextlib.redirect_stdout(io.StringIO()):
        norm = ds.normalize(products, "")
    feed_instock = set(norm["id"].astype(str))
    feed_all = {str(p["id"]) for p in products}
    taken = dt.datetime.fromtimestamp(snapshot_csv.stat().st_mtime, tz=dt.timezone.utc)
    created = {str(p["id"]): str(p.get("created_at") or "") for p in products}
    new_ids = feed_instock - snap_ids
    new_after = len(
        [i for i in new_ids if created[i] and dt.datetime.fromisoformat(created[i]) > taken]
    )
    return {
        "snapshot_rows": len(snap),
        "snapshot_date_mtime": taken.strftime("%Y-%m-%d"),
        "snapshot_ids_still_in_stock_in_feed": len(snap_ids & feed_instock),
        "snapshot_ids_in_feed_but_now_sold_out_or_unpriced": len((snap_ids & feed_all) - feed_instock),
        # "gone" only holds when EVERY page of the feed was loaded; with partial pages it is just
        # "not in the loaded pages".
        "snapshot_ids_not_in_loaded_feed": len(snap_ids - feed_all),
        "in_stock_feed_ids_not_in_snapshot": len(new_ids),
        "of_those_created_after_snapshot": new_after,
    }


def main() -> int:
    """CLI entry: print the funnel (and snapshot overlap) as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--snapshot-csv", type=Path, default=None)
    ap.add_argument("--workdir", type=Path, default=None, help="temp CSV dir (default: raw dir)")
    args = ap.parse_args()
    import yaml  # noqa: PLC0415 - only the CLI needs it

    cfg = yaml.safe_load((ROOT / "brands" / f"{args.slug}.yaml").read_text(encoding="utf-8"))
    brand = SimpleNamespace(
        sizing_system=cfg.get("sizing_system", "IN"), gender_default=cfg.get("gender_default", "women")
    )
    products = load_pages(args.raw_dir, args.slug)
    if not products:
        print(f"no saved pages for {args.slug} in {args.raw_dir}", file=sys.stderr)
        return 2
    out: dict[str, object] = dict(funnel(products, args.slug, brand, args.workdir or args.raw_dir))
    if args.snapshot_csv:
        out["snapshot"] = snapshot_overlap(products, args.snapshot_csv)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
