"""src/refresh/incremental.py: every rule branch with fake embedders. Hermetic, no network."""

from __future__ import annotations

import hashlib

import numpy as np

from src.refresh import incremental as inc
from src.refresh import state as st


def _vec(key: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
    return np.random.default_rng(seed).random(dim, dtype=np.float32)


class FakeEmbedder:
    """Deterministic vectors; records every call; ``fail`` urls come back as None."""

    def __init__(self, fail: frozenset[str] = frozenset()) -> None:
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[str]] = []
        self.fail = fail

    def embed_text(self, texts: list[str]) -> np.ndarray:
        self.text_calls.append(list(texts))
        return np.stack([_vec("t" + t, st.TEXT_DIM) for t in texts])

    def embed_images(self, urls: list[str]) -> list[np.ndarray | None]:
        self.image_calls.append(list(urls))
        return [None if u in self.fail else _vec("i" + u, st.IMAGE_DIM) for u in urls]


def prod(
    pid: int,
    title: str = "Blue Kurta",
    price: str = "999",
    avail: bool = True,
    img: str | None = "auto",
    tags: tuple[str, ...] = (),
    ptype: str = "Kurta",
) -> dict:
    image = f"https://img/{pid}.jpg" if img == "auto" else img
    return {
        "id": pid,
        "title": title,
        "product_type": ptype,
        "tags": list(tags),
        "variants": [{"price": price, "available": avail}],
        "images": [{"src": image}] if image else [],
        "_label": "women",
    }


def prepare(slug: str, products: list[dict]) -> list[dict]:
    """Stand-in for the build-filter replay: keeps everything it is given."""
    return [
        {
            "article_id": str(p["id"]),
            "prod_name": p["title"],
            "gender": p["_label"],
            "search_text": p["title"],
            "image_url": (p["images"][0]["src"] if p["images"] else None),
            "store": slug,
        }
        for p in products
    ]


def run(feeds, prev=None, emb=None, **kw):
    emb = emb or FakeEmbedder()
    res = inc.build_incremental(feeds, prev or st.IndexState(), prepare, emb, **kw)
    return res, emb


def feed(slug: str, products: list[dict], status: str = "ok") -> inc.FeedStore:
    return inc.FeedStore(slug, status, products)


def store_rep(res, slug: str) -> dict:
    return next(s for s in res.report["stores"] if s["slug"] == slug)


def test_cold_start_embeds_everything():
    res, emb = run([feed("a", [prod(1), prod(2)])])
    r = store_rep(res, "a")
    assert (r["fetched"], r["kept"], r["new"], r["changed"], r["dropped"]) == (2, 2, 2, 0, 0)
    assert (r["texts_embedded"], r["images_embedded"], r["images_deferred"]) == (2, 2, 0)
    assert len(emb.image_calls[0]) == 2
    assert all(x.image_vec is not None and not x.image_pending for x in res.state.rows)


def test_unchanged_items_reuse_vectors_without_embedding():
    res1, _ = run([feed("a", [prod(1), prod(2)])])
    res2, emb2 = run([feed("a", [prod(1), prod(2)])], prev=res1.state)
    r = store_rep(res2, "a")
    assert (r["unchanged"], r["new"], r["changed"]) == (2, 0, 0)
    assert emb2.text_calls == [] and emb2.image_calls == []
    for a, b in zip(res1.state.rows, res2.state.rows, strict=True):
        assert np.array_equal(a.text_vec, b.text_vec) and np.array_equal(a.image_vec, b.image_vec)


def test_each_hashed_field_triggers_change():
    base = prod(1, tags=("x",))
    res1, _ = run([feed("a", [base])])
    variants = {
        "title": prod(1, title="Red Kurta", tags=("x",)),
        "price": prod(1, price="1200", tags=("x",)),
        "image": prod(1, img="https://img/other.jpg", tags=("x",)),
        "tags": prod(1, tags=("x", "y")),
    }
    for name, p in variants.items():
        res2, _ = run([feed("a", [p])], prev=res1.state)
        assert store_rep(res2, "a")["changed"] == 1, name
    # description / vendor are not hashed (documented limitation)
    p = dict(base, body_html="new words", vendor="V2")
    res3, _ = run([feed("a", [p])], prev=res1.state)
    assert store_rep(res3, "a")["unchanged"] == 1


def test_changed_price_reembeds_text_but_reuses_image_vector():
    res1, _ = run([feed("a", [prod(1)])])
    res2, emb2 = run([feed("a", [prod(1, price="1500")])], prev=res1.state)
    assert len(emb2.text_calls) == 1 and emb2.image_calls == []
    assert np.array_equal(res2.state.rows[0].image_vec, res1.state.rows[0].image_vec)


def test_changed_image_url_reembeds_image():
    res1, _ = run([feed("a", [prod(1)])])
    res2, emb2 = run([feed("a", [prod(1, img="https://img/new.jpg")])], prev=res1.state)
    assert emb2.image_calls == [["https://img/new.jpg"]]
    assert not np.array_equal(res2.state.rows[0].image_vec, res1.state.rows[0].image_vec)


def test_missing_and_sold_out_items_are_dropped():
    res1, _ = run([feed("a", [prod(1), prod(2), prod(3)])])
    res2, _ = run([feed("a", [prod(1), prod(3, avail=False), prod(4, price="0")])], prev=res1.state)
    r = store_rep(res2, "a")
    assert [x.article_id for x in res2.state.rows] == ["1"]
    assert (r["fetched"], r["kept"], r["dropped"], r["new"]) == (3, 1, 2, 0)


def test_image_cap_defers_in_order_and_next_run_finishes():
    feeds = [feed("a", [prod(i) for i in (1, 2, 3)]), feed("b", [prod(4), prod(5)])]
    res1, emb1 = run(feeds, image_cap=2)
    assert res1.report["cap_hit"] is True
    assert emb1.image_calls == [["https://img/1.jpg", "https://img/2.jpg"]]  # (slug, id) order
    assert (store_rep(res1, "a")["images_embedded"], store_rep(res1, "a")["images_deferred"]) == (
        2,
        1,
    )
    assert store_rep(res1, "b")["images_deferred"] == 2
    assert res1.report["totals"]["kept"] == 5  # deferred rows are kept, never dropped
    pending = [r.article_id for r in res1.state.rows if r.image_pending]
    assert pending == ["3", "4", "5"] and all(
        r.image_vec is None for r in res1.state.rows if r.image_pending
    )
    # next run: hashes unchanged, pending rows still get their image vectors
    res2, emb2 = run(feeds, prev=res1.state, image_cap=2)
    assert sum(len(c) for c in emb2.image_calls) == 2 and emb2.text_calls == []
    assert res2.report["cap_hit"] is True
    res3, _ = run(feeds, prev=res2.state, image_cap=2)
    assert res3.report["cap_hit"] is False
    assert all(r.image_vec is not None and not r.image_pending for r in res3.state.rows)


def test_cap_exactly_met_is_not_a_hit():
    res, _ = run([feed("a", [prod(1), prod(2)])], image_cap=2)
    assert res.report["cap_hit"] is False


def test_image_failure_stays_pending_and_is_retried():
    res1, _ = run(
        [feed("a", [prod(1), prod(2)])], emb=FakeEmbedder(fail=frozenset({"https://img/2.jpg"}))
    )
    r = store_rep(res1, "a")
    assert (r["images_embedded"], r["images_failed"], r["kept"]) == (1, 1, 2)
    res2, emb2 = run([feed("a", [prod(1), prod(2)])], prev=res1.state)
    assert emb2.image_calls == [["https://img/2.jpg"]]
    assert all(x.image_vec is not None for x in res2.state.rows)


def test_item_without_image_is_kept_not_pending():
    res, emb = run([feed("a", [prod(1, img=None)])])
    row = res.state.rows[0]
    assert row.image_vec is None and not row.image_pending and emb.image_calls == []


def test_non_ok_store_is_carried_over_untouched_and_reported():
    res1, _ = run([feed("a", [prod(1), prod(2)]), feed("b", [prod(3)])])
    for status in ("rate_limited", "capped", "challenge"):
        res2, emb2 = run([feed("a", [], status), feed("b", [prod(3)])], prev=res1.state)
        a_rows = [r for r in res2.state.rows if r.store == "a"]
        assert [r.article_id for r in a_rows] == ["1", "2"]
        assert all(
            x is y
            for x, y in zip(a_rows, [r for r in res1.state.rows if r.store == "a"], strict=True)
        )
        r = store_rep(res2, "a")
        assert (r["status"], r["carried_over"], r["carry_reason"]) == (status, 2, status)
        assert res2.report["carried_over_stores"] == {"a": status}
        assert res2.state.store_status["a"] == status
        assert emb2.text_calls == []


def test_store_missing_from_fetch_is_carried_over_as_not_attempted():
    res1, _ = run([feed("a", [prod(1)]), feed("b", [prod(3)])])
    res2, _ = run([feed("b", [prod(3)])], prev=res1.state)
    assert {r.store for r in res2.state.rows} == {"a", "b"}
    assert res2.report["carried_over_stores"] == {"a": "not_attempted"}


def test_non_ok_store_without_previous_rows_contributes_nothing():
    res, _ = run([feed("a", [], "rate_limited")])
    assert res.state.rows == [] and store_rep(res, "a")["carried_over"] == 0


def test_shrink_guard_carries_over_a_suspiciously_small_ok_feed():
    full = [prod(i) for i in range(1, 41)]
    res1, _ = run([feed("a", full)])
    res2, emb2 = run([feed("a", full[:5])], prev=res1.state)  # 5 < 50% of 40
    r = store_rep(res2, "a")
    assert (r["carry_reason"], r["kept"], r["carried_over"]) == ("shrink_guard", 40, 40)
    assert emb2.text_calls == []
    res3, _ = run([feed("a", full[:25])], prev=res1.state)  # 25 >= 50%: genuine shrink applied
    assert store_rep(res3, "a")["dropped"] == 15
    # small stores are exempt (below SHRINK_GUARD_MIN_PREVIOUS)
    res4, _ = run([feed("s", [prod(1), prod(2), prod(3)])])
    res5, _ = run([feed("s", [prod(1)])], prev=res4.state)
    assert store_rep(res5, "s")["dropped"] == 2


def test_gender_correction_is_applied_logged_and_counted_per_store():
    products = [
        prod(1, title="Men Linen Shirt", ptype="Shirt"),  # label women: corrected
        prod(2, title="Men Linen Shirt", ptype="Womens Shirt"),  # type supports label: ambiguous
        prod(3, title="Women Kurta"),  # label women: no conflict
    ]
    res, _ = run([feed("a", products), feed("b", [prod(9, title="Boys Tee", ptype="Tee")])])
    rows = {r.article_id: r.row["gender"] for r in res.state.rows}
    assert rows == {"1": "men", "2": "women", "3": "women", "9": "men"}
    assert (store_rep(res, "a")["gender_corrected"], store_rep(res, "a")["gender_ambiguous"]) == (
        1,
        1,
    )
    assert store_rep(res, "b")["gender_corrected"] == 1
    log = res.report["gender_corrections"]
    assert [(c["store"], c["article_id"], c["before"], c["after"]) for c in log] == [
        ("a", "1", "women", "men"),
        ("b", "9", "women", "men"),
    ]
    # a second run does not log them again (stored rows are corrected and unchanged)
    res2, _ = run(
        [feed("a", products), feed("b", [prod(9, title="Boys Tee", ptype="Tee")])], prev=res.state
    )
    assert res2.report["gender_corrections"] == []


def test_deterministic_and_state_survives_a_file_round_trip(tmp_path):
    feeds = [feed("b", [prod(2), prod(1)]), feed("a", [prod(3)])]
    r1, _ = run(feeds)
    r2, _ = run(feeds)
    assert [(x.store, x.article_id) for x in r1.state.rows] == [("a", "3"), ("b", "1"), ("b", "2")]
    assert all(
        np.array_equal(x.text_vec, y.text_vec)
        for x, y in zip(r1.state.rows, r2.state.rows, strict=True)
    )
    p = tmp_path / "index-state.npz"
    st.write_state(r1.state, p)
    r3, emb3 = run(feeds, prev=st.read_state(p))
    assert r3.report["totals"]["unchanged"] == 3 and emb3.text_calls == []


def test_report_markdown_and_index_arrays():
    res, _ = run(
        [
            feed("a", [prod(1, title="Men Shirt", ptype="Shirt"), prod(2, img=None)]),
            feed("b", [], "rate_limited"),
        ]
    )
    md = inc.to_markdown(res.report)
    assert "| a | ok |" in md and "| b | rate_limited |" in md and "Men Shirt" in md
    cat, text, img_ids, img = inc.index_arrays(res.state)
    assert len(cat) == 2 and text.shape == (2, st.TEXT_DIM)
    assert img_ids == ["1"] and img.shape == (1, st.IMAGE_DIM)
    assert inc.index_arrays(st.IndexState())[0].empty
