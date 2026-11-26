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
    """Deterministic vectors; records every call (MiniLM text and CLIP text)."""

    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.clip_calls: list[list[str]] = []

    def embed_text(self, texts: list[str]) -> np.ndarray:
        self.text_calls.append(list(texts))
        return np.stack([_vec("t" + t, st.TEXT_DIM) for t in texts])

    def embed_clip_text(self, texts: list[str]) -> np.ndarray:
        self.clip_calls.append(list(texts))
        return np.stack([_vec("c" + t, st.CLIP_DIM) for t in texts])


def prod(
    pid: int,
    title: str = "Blue Kurta",
    price: str = "999",
    avail: bool = True,
    img: str | None = "auto",
    tags: tuple[str, ...] = (),
    ptype: str = "Kurta",
    desc: str = "",
) -> dict:
    image = f"https://img/{pid}.jpg" if img == "auto" else img
    return {
        "id": pid,
        "title": title,
        "product_type": ptype,
        "body_html": desc,
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
            "search_text": p["title"] + ". " + p["product_type"] + ". " + p["body_html"],
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
    assert (r["texts_embedded"], r["vectors_reused"]) == (2, 0)
    assert len(emb.text_calls[0]) == 2 and emb.clip_calls == emb.text_calls
    assert all(x.clip_vec.shape == (st.CLIP_DIM,) for x in res.state.rows)
    assert all(x.clip_vec.dtype == np.float32 for x in res.state.rows)


def test_unchanged_items_reuse_vectors_without_embedding():
    res1, _ = run([feed("a", [prod(1), prod(2)])])
    res2, emb2 = run([feed("a", [prod(1), prod(2)])], prev=res1.state)
    r = store_rep(res2, "a")
    assert (r["unchanged"], r["new"], r["changed"]) == (2, 0, 0)
    assert emb2.text_calls == [] and emb2.clip_calls == []
    for a, b in zip(res1.state.rows, res2.state.rows, strict=True):
        assert np.array_equal(a.text_vec, b.text_vec) and np.array_equal(a.clip_vec, b.clip_vec)


def test_each_hashed_field_triggers_change():
    base = prod(1, tags=("x",))
    res1, _ = run([feed("a", [base])])
    variants = {
        "title": prod(1, title="Red Kurta", tags=("x",)),
        "price": prod(1, price="1200", tags=("x",)),
        "product_type": prod(1, ptype="Tunic", tags=("x",)),
        "description": prod(1, desc="new words", tags=("x",)),
    }
    for name, p in variants.items():
        res2, _ = run([feed("a", [p])], prev=res1.state)
        assert store_rep(res2, "a")["changed"] == 1, name
    # image url, tags and vendor do not feed search_text and are not hashed (documented)
    for name, p in {
        "image": prod(1, img="https://img/other.jpg", tags=("x",)),
        "tags": prod(1, tags=("x", "y")),
        "vendor": dict(base, vendor="V2"),
    }.items():
        res3, _ = run([feed("a", [p])], prev=res1.state)
        assert store_rep(res3, "a")["unchanged"] == 1, name


def test_price_only_change_reuses_both_vectors():
    res1, _ = run([feed("a", [prod(1)])])
    res2, emb2 = run([feed("a", [prod(1, price="1500")])], prev=res1.state)
    r = store_rep(res2, "a")
    assert (r["changed"], r["texts_embedded"], r["vectors_reused"]) == (1, 0, 1)
    assert emb2.text_calls == [] and emb2.clip_calls == []
    assert np.array_equal(res2.state.rows[0].text_vec, res1.state.rows[0].text_vec)
    assert np.array_equal(res2.state.rows[0].clip_vec, res1.state.rows[0].clip_vec)
    assert res2.state.rows[0].content_hash != res1.state.rows[0].content_hash


def test_changed_search_text_reembeds_both_models():
    res1, _ = run([feed("a", [prod(1), prod(2)])])
    res2, emb2 = run([feed("a", [prod(1, title="Red Kurta"), prod(2, price="1")])], prev=res1.state)
    r = store_rep(res2, "a")
    assert (r["changed"], r["texts_embedded"], r["vectors_reused"]) == (2, 1, 1)
    assert emb2.text_calls == [["Red Kurta. Kurta. "]] and emb2.clip_calls == emb2.text_calls
    new, old = res2.state.rows[0], res1.state.rows[0]
    assert not np.array_equal(new.text_vec, old.text_vec)
    assert not np.array_equal(new.clip_vec, old.clip_vec)


def test_missing_and_sold_out_items_are_dropped():
    res1, _ = run([feed("a", [prod(1), prod(2), prod(3)])])
    res2, _ = run([feed("a", [prod(1), prod(3, avail=False), prod(4, price="0")])], prev=res1.state)
    r = store_rep(res2, "a")
    assert [x.article_id for x in res2.state.rows] == ["1"]
    assert (r["fetched"], r["kept"], r["dropped"], r["new"]) == (3, 1, 2, 0)


def test_embedding_is_by_sorted_article_id_with_one_call_per_model_per_store():
    feeds = [feed("b", [prod(5), prod(4)]), feed("a", [prod(3), prod(1), prod(2)])]
    _, emb = run(feeds)
    assert [len(c) for c in emb.text_calls] == [3, 2]  # store a then b, one call each
    assert emb.text_calls[0] == [
        "Blue Kurta. Kurta. ",
        "Blue Kurta. Kurta. ",
        "Blue Kurta. Kurta. ",
    ]
    assert emb.clip_calls == emb.text_calls


def test_item_without_image_is_kept_and_embedded_from_text():
    res, emb = run([feed("a", [prod(1, img=None)])])
    row = res.state.rows[0]
    assert row.clip_vec.shape == (st.CLIP_DIM,) and len(emb.clip_calls) == 1


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


def test_hyphenated_and_denylisted_names_are_left_alone_and_counted_per_store():
    a = [
        prod(1, title="Man-Made Silk Saree"),  # label women: hyphenated, unchanged
        prod(2, title="Man Repeller Tee"),  # denylisted: unchanged, not counted
        prod(3, title="Men's Kurta", ptype="Kurta"),  # genuine: corrected
    ]
    b = [prod(9, title="Boy-friend Fit Jeans")]
    res, _ = run([feed("a", a), feed("b", b)])
    rows = {r.article_id: r.row["gender"] for r in res.state.rows}
    assert rows == {"1": "women", "2": "women", "3": "men", "9": "women"}
    ra, rb = store_rep(res, "a"), store_rep(res, "b")
    assert (ra["gender_corrected"], ra["gender_ambiguous"], ra["gender_ambiguous_hyphenated"]) == (
        1,
        1,
        1,
    )
    assert rb["gender_ambiguous_hyphenated"] == 1
    assert res.report["totals"]["gender_ambiguous_hyphenated"] == 2
    assert "gender_ambiguous_hyphenated" in inc.to_markdown(res.report)


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
    cat, text, clip = inc.index_arrays(res.state)
    assert len(cat) == 2 and text.shape == (2, st.TEXT_DIM) and clip.shape == (2, st.CLIP_DIM)
    assert inc.index_arrays(st.IndexState())[0].empty
