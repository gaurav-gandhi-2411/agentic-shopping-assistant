"""Incremental index build from a fetch pass: previous state + fetched feeds -> new state + report.

Reads the pages saved by ``fetch_pass.py --raw-dir`` and its ``--out`` summary (per-store status),
the previous state file (optional on the first run), and writes the new state file plus a JSON and
a markdown report. Rules: docs/ops/incremental-build-rules.md. Core logic: src/refresh/incremental.py.
NOT wired into prod: it writes no index, no bucket and calls no cloud API.

Dry-run command (6e, run by the orchestrator via the incremental-build-dry-run workflow):
    python scripts/ops/fetch_pass.py --out fetch.json --raw-dir raw --only voylla,libas
    python scripts/ops/incremental_build.py --fetch-summary fetch.json --raw-dir raw \\
        --state-out out/index-state.npz --report-json out/report.json --report-md out/report.md
    # with the previous state: add --state-in prev/index-state.npz

``--embedder fake`` (default) uses deterministic hash vectors and no model or network, for plumbing
checks. ``--embedder real`` loads MiniLM (text) and CLIP (images, downloaded from the store CDN).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feed_delta_funnel import load_pages  # noqa: E402

from src.refresh import incremental as inc  # noqa: E402
from src.refresh import state as st  # noqa: E402

MAX_IMAGE_BYTES = 10_000_000  # a hostile or broken CDN response cannot exhaust runner memory
IMAGE_TIMEOUT_S = 15.0


class FakeEmbedder:
    """Deterministic hash-seeded vectors; no model, no network (plumbing and dry checks)."""

    @staticmethod
    def _vec(key: str, dim: int) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec("t" + t, st.TEXT_DIM) for t in texts])

    def embed_images(self, urls: list[str]) -> list[np.ndarray | None]:
        return [self._vec("i" + u, st.IMAGE_DIM) for u in urls]


class RealEmbedder:
    """MiniLM text + CLIP image embedding, loaded lazily. Images are fetched over HTTPS."""

    def __init__(self) -> None:
        import yaml  # noqa: PLC0415 - heavy deps only when this embedder is chosen
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        from src.retrieval.clip_encoder import get_clip_encoder  # noqa: PLC0415

        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        self._text = SentenceTransformer(cfg["retrieval"]["dense_model"], device="cpu")
        self._clip = get_clip_encoder(cfg["clip"]["model"])

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._text.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_images(self, urls: list[str]) -> list[np.ndarray | None]:
        from PIL import Image  # noqa: PLC0415

        out: list[np.ndarray | None] = []
        for url in urls:
            try:
                if not url.startswith("https://"):
                    raise ValueError("non-https image url")
                req = urllib.request.Request(
                    url, headers={"User-Agent": "StyleMaitriFeedFetcher/1.0"}
                )
                with urllib.request.urlopen(req, timeout=IMAGE_TIMEOUT_S) as resp:  # noqa: S310
                    data = resp.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    raise ValueError("image too large")
                out.append(self._clip.encode_image(Image.open(io.BytesIO(data))))
            except Exception:  # noqa: BLE001 - any failure leaves the row pending, retried next run
                out.append(None)
        return out


def make_prepare(brands: dict[str, object]):  # noqa: ANN201
    """Real ``prepare``: replay the build filters on a store's products (same path as the funnel).

    normalize -> adapt_feed -> religious decor / no description / dedupe ->
    ``build_unified_index._clean_brand_catalogue`` (mojibake, saree, fabric, colour, hygiene,
    allowed categories). The CSV round trip mirrors the real build.
    """

    def prepare(slug: str, products: list[dict]) -> list[dict]:
        import contextlib  # noqa: PLC0415

        import download_shopify as ds  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from build_unified_index import _clean_brand_catalogue  # noqa: PLC0415

        from src.catalogue.adapter import adapt_feed  # noqa: PLC0415
        from src.catalogue.cleaning import drop_religious_decor_items  # noqa: PLC0415

        if not products:
            return []
        with contextlib.redirect_stdout(io.StringIO()):
            norm = ds.normalize(products, "")
        if norm.empty:
            return []
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / f"{slug}.csv"
            norm.to_csv(csv, index=False, encoding="utf-8")
            ad = adapt_feed(pd.read_csv(csv), brands[slug])
        ad, _ = drop_religious_decor_items(ad)
        ad = ad.dropna(subset=["detail_desc"]).drop_duplicates("article_id").reset_index(drop=True)
        ad["store"] = slug
        ad, _ = _clean_brand_catalogue(slug, ad)
        return ad.to_dict("records")

    return prepare


def load_brands(slugs: list[str]) -> dict[str, object]:
    """Brand config stand-in per slug from brands/<slug>.yaml (sizing system, gender default)."""
    import yaml  # noqa: PLC0415

    out: dict[str, object] = {}
    for slug in slugs:
        cfg = yaml.safe_load((ROOT / "brands" / f"{slug}.yaml").read_text(encoding="utf-8"))
        out[slug] = SimpleNamespace(
            sizing_system=cfg.get("sizing_system", "IN"),
            gender_default=cfg.get("gender_default", "women"),
        )
    return out


def load_feeds(summary: dict, raw_dir: Path) -> list[inc.FeedStore]:
    """FeedStores from a fetch_pass summary and its raw pages.

    An ``ok`` store with no saved pages is reported as ``no_pages`` (carried over): failing closed
    beats rebuilding a store from nothing.
    """
    feeds: list[inc.FeedStore] = []
    for s in summary["stores"]:
        if s["status"] != inc.OK:
            feeds.append(inc.FeedStore(s["slug"], s["status"]))
            continue
        products = load_pages(raw_dir, s["slug"])
        feeds.append(inc.FeedStore(s["slug"], inc.OK if products else "no_pages", products))
    return feeds


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Exit 0 on success, 2 when the previous state is unreadable (fail closed)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch-summary", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--state-in", type=Path, default=None)
    ap.add_argument("--state-out", required=True, type=Path)
    ap.add_argument("--report-json", required=True, type=Path)
    ap.add_argument("--report-md", required=True, type=Path)
    ap.add_argument("--image-cap", type=int, default=inc.IMAGE_EMBED_CAP_PER_RUN)
    ap.add_argument("--embedder", choices=("fake", "real"), default="fake")
    args = ap.parse_args(argv)

    try:
        prev = st.read_state(args.state_in) if args.state_in else st.IndexState()
    except st.StateError as exc:
        print(f"refusing to build: {exc}", file=sys.stderr)
        return 2
    summary = json.loads(args.fetch_summary.read_text(encoding="utf-8"))
    feeds = load_feeds(summary, args.raw_dir)
    brands = load_brands([f.slug for f in feeds if f.status == inc.OK])
    embedder = RealEmbedder() if args.embedder == "real" else FakeEmbedder()
    res = inc.build_incremental(
        feeds, prev, make_prepare(brands), embedder, image_cap=args.image_cap
    )
    st.write_state(res.state, args.state_out)
    args.report_json.write_text(json.dumps(res.report, indent=1), encoding="utf-8")
    args.report_md.write_text(inc.to_markdown(res.report), encoding="utf-8")
    print(json.dumps(res.report["totals"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
