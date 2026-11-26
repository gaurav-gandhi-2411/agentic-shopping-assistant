"""scripts/ops/incremental_build.py end to end on a tiny fixture: real build filters, fake vectors.

Hermetic: pages and summary are written to tmp_path; the fake embedder needs no model or network.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("faiss")  # build_unified_index (the real filter replay) imports faiss
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import incremental_build as ib  # noqa: E402

from src.refresh import state as st  # noqa: E402

SLUG = "voylla"


def _p(pid: int, title: str, avail: bool = True, price: str = "799.00") -> dict:
    return {
        "id": pid,
        "title": title,
        "handle": f"p{pid}",
        "product_type": "Necklace",
        "vendor": "V",
        "tags": ["jewellery"],
        "body_html": f"<p>{title} in gold tone</p>",
        "variants": [{"price": price, "available": avail}],
        "images": [{"src": f"https://cdn.example/{pid}.jpg"}],
    }


FEED = [
    _p(1, "Gold Tone Necklace"),
    _p(2, "Pearl Drop Necklace"),
    _p(3, "Sold Out Necklace", False),
]


def _setup(tmp_path: Path, products: list[dict], status: str = "ok") -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for f in raw.glob("*.gz"):
        f.unlink()
    (raw / f"{SLUG}__p0001.json.gz").write_bytes(
        gzip.compress(json.dumps({"products": products}).encode())
    )
    summary = tmp_path / "fetch.json"
    summary.write_text(json.dumps({"stores": [{"slug": SLUG, "status": status}]}))
    return raw, summary


def _run(tmp_path: Path, raw: Path, summary: Path, name: str, state_in: Path | None) -> dict:
    out = tmp_path / name
    args = [
        "--fetch-summary",
        str(summary),
        "--raw-dir",
        str(raw),
        "--state-out",
        str(out / "s.npz"),
        "--report-json",
        str(out / "r.json"),
        "--report-md",
        str(out / "r.md"),
    ]
    out.mkdir()
    if state_in:
        args += ["--state-in", str(state_in)]
    assert ib.main(args) == 0
    assert (out / "r.md").read_text().startswith("# Incremental build report")
    return json.loads((out / "r.json").read_text())


def test_cold_run_then_incremental_run_then_carry_over(tmp_path):
    raw, summary = _setup(tmp_path, FEED)
    r1 = _run(tmp_path, raw, summary, "run1", None)
    s1 = next(s for s in r1["stores"] if s["slug"] == SLUG)
    assert (s1["fetched"], s1["kept"], s1["new"], s1["images_embedded"]) == (3, 2, 2, 2)

    # same feed again from the saved state: nothing to embed
    r2 = _run(tmp_path, raw, summary, "run2", tmp_path / "run1" / "s.npz")
    s2 = next(s for s in r2["stores"] if s["slug"] == SLUG)
    assert (s2["unchanged"], s2["new"], s2["texts_embedded"], s2["images_embedded"]) == (2, 0, 0, 0)

    # one product repriced, one removed
    raw, summary = _setup(tmp_path, [_p(1, "Gold Tone Necklace", price="999.00")])
    r3 = _run(tmp_path, raw, summary, "run3", tmp_path / "run2" / "s.npz")
    s3 = next(s for s in r3["stores"] if s["slug"] == SLUG)
    assert (s3["changed"], s3["dropped"], s3["images_embedded"]) == (1, 1, 0)

    # store rate limited: previous rows untouched and reported
    raw, summary = _setup(tmp_path, [], status="rate_limited")
    r4 = _run(tmp_path, raw, summary, "run4", tmp_path / "run3" / "s.npz")
    assert r4["carried_over_stores"] == {SLUG: "rate_limited"}
    assert len(st.read_state(tmp_path / "run4" / "s.npz").rows) == 1


def test_unreadable_previous_state_is_a_hard_refusal(tmp_path, capsys):
    raw, summary = _setup(tmp_path, FEED)
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"junk")
    out = tmp_path / "o"
    out.mkdir()
    rc = ib.main(
        [
            "--fetch-summary",
            str(summary),
            "--raw-dir",
            str(raw),
            "--state-in",
            str(bad),
            "--state-out",
            str(out / "s.npz"),
            "--report-json",
            str(out / "r.json"),
            "--report-md",
            str(out / "r.md"),
        ]
    )
    assert rc == 2 and not (out / "s.npz").exists()


def test_ok_store_without_pages_is_carried_not_rebuilt(tmp_path):
    raw, summary = _setup(tmp_path, FEED)
    r1 = _run(tmp_path, raw, summary, "run1", None)
    assert r1["totals"]["kept"] == 2
    for f in raw.glob("*.gz"):
        f.unlink()
    r2 = _run(tmp_path, raw, summary, "run2", tmp_path / "run1" / "s.npz")
    assert r2["carried_over_stores"] == {SLUG: "no_pages"}
    assert r2["totals"]["kept"] == 2
