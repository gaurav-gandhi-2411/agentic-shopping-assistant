"""src/refresh/state.py and gender_hygiene.py: format guarantees and each rule branch. Hermetic."""

from __future__ import annotations

import numpy as np
import pytest

from src.refresh import gender_hygiene as gh
from src.refresh import state as st


def _row(aid: str, store: str = "s1", image: bool = True, pending: bool = False) -> st.StateRow:
    rng = np.random.default_rng(42)
    return st.StateRow(
        store=store,
        article_id=aid,
        content_hash="h" + aid,
        row={"article_id": aid, "n": 1.5},
        text_vec=rng.random(st.TEXT_DIM, dtype=np.float32),
        image_vec=rng.random(st.IMAGE_DIM, dtype=np.float32) if image else None,
        image_pending=pending,
    )


def test_state_round_trip(tmp_path):
    s = st.IndexState(
        rows=[_row("1"), _row("2", image=False, pending=True)],
        store_status={"s1": "ok", "s2": "rate_limited"},
        built_at="2026-11-26T00:00:00Z",
    )
    p = tmp_path / "index-state.npz"
    st.write_state(s, p)
    back = st.read_state(p)
    assert [r.article_id for r in back.rows] == ["1", "2"]
    assert back.rows[0].row == {"article_id": "1", "n": 1.5}
    assert np.array_equal(back.rows[0].image_vec, s.rows[0].image_vec)
    assert back.rows[1].image_vec is None and back.rows[1].image_pending
    assert back.store_status == s.store_status and back.built_at == s.built_at


def test_empty_state_round_trip(tmp_path):
    p = tmp_path / "e.npz"
    st.write_state(st.IndexState(), p)
    assert st.read_state(p).rows == []


def _rewrite(path, **override):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data.update(override)
    np.savez_compressed(path, **data)


def test_checksum_mismatch_refused(tmp_path):
    p = tmp_path / "s.npz"
    st.write_state(st.IndexState(rows=[_row("1")]), p)
    _rewrite(p, content_hash=np.array(["tampered"]))
    with pytest.raises(st.StateError, match="checksum"):
        st.read_state(p)


def test_unknown_schema_version_refused(tmp_path):
    import json

    p = tmp_path / "s.npz"
    st.write_state(st.IndexState(rows=[_row("1")]), p)
    with np.load(p, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
    header["schema_version"] = 99
    _rewrite(p, header=np.array(json.dumps(header)))
    with pytest.raises(st.StateError, match="schema_version"):
        st.read_state(p)


def test_missing_and_garbage_file_refused(tmp_path):
    with pytest.raises(st.StateError):
        st.read_state(tmp_path / "nope.npz")
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip")
    with pytest.raises(st.StateError):
        st.read_state(bad)


# --- gender rule: one test per branch -------------------------------------------------------


def test_leading_gender_words():
    assert gh.leading_gender("Men's Slim Shirt") == "men"
    assert gh.leading_gender("Womens Kurta") == "women"
    assert gh.leading_gender("Boys Tee") == "men"
    assert gh.leading_gender("Girls Frock") == "women"
    assert gh.leading_gender("Roman Sandals") is None
    assert gh.leading_gender("Boyfriend Jeans") is None
    assert gh.leading_gender("Blue Women Top") is None  # only the FIRST word counts
    assert gh.leading_gender("") is None and gh.leading_gender(None) is None


def test_no_conflict_when_label_matches_or_not_gendered():
    assert gh.resolve_gender("Men Shirt", "men", "", []).outcome == gh.NO_CONFLICT
    assert gh.resolve_gender("Men Shirt", "unknown", "", []).outcome == gh.NO_CONFLICT
    assert gh.resolve_gender("Men Shirt", "mixed", "", []).outcome == gh.NO_CONFLICT
    assert gh.resolve_gender("Blue Shirt", "women", "", []).outcome == gh.NO_CONFLICT


def test_corrected_when_no_other_evidence():
    d = gh.resolve_gender("Women Kurta Set", "men", "Kurta Set", ["cotton"])
    assert (d.outcome, d.before, d.after) == (gh.CORRECTED, "men", "women")
    assert d.reason == "leading_word_no_other_evidence"
    assert gh.resolve_gender("Boys Tee", "women", "", []).after == "men"


def test_unisex_marker_leaves_label():
    for args in (
        ("Men Unisex Tee", "women", "", []),
        ("Men Tee", "women", "Unisex", []),
        ("Men Tee", "women", "", ["Unisex"]),
    ):
        d = gh.resolve_gender(*args)
        assert (d.outcome, d.after, d.reason) == (gh.AMBIGUOUS, "women", "unisex_marker")


def test_product_type_decides_first():
    agree = gh.resolve_gender("Women Top", "men", "Womens Tops", [])
    assert (agree.outcome, agree.after, agree.reason) == (
        gh.CORRECTED,
        "women",
        "product_type_agrees_with_name",
    )
    supports = gh.resolve_gender("Women Top", "men", "Mens Tops", [])
    assert (supports.outcome, supports.after, supports.reason) == (
        gh.AMBIGUOUS,
        "men",
        "product_type_supports_label",
    )
    both = gh.resolve_gender("Women Top", "men", "Men and Women Tops", [])
    assert (both.outcome, both.reason) == (gh.AMBIGUOUS, "product_type_both_genders")
    # product_type evidence outranks contradicting tags
    d = gh.resolve_gender("Women Top", "men", "Womens Tops", ["mens"])
    assert d.outcome == gh.CORRECTED


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("Man-Made Silk Saree", "women"),
        ("Woman-Owned Brand Tee", "men"),
        ("Men-in-Black Tee", "women"),
        ("Boy-friend Fit Jeans", "women"),
    ],
)
def test_hyphenated_leading_word_never_corrects(name, label):
    d = gh.resolve_gender(name, label, "", [])
    assert (d.outcome, d.after, d.reason) == (gh.AMBIGUOUS, label, gh.HYPHENATED_REASON)
    assert gh.leading_gender(name) is None


@pytest.mark.parametrize(
    ("name", "label"), [("Man Repeller Tee", "women"), ("Ladies Man Tee", "men")]
)
def test_denylisted_phrase_never_corrects(name, label):
    d = gh.resolve_gender(name, label, "", [])
    assert (d.outcome, d.after) == (gh.NO_CONFLICT, label)


@pytest.mark.parametrize(
    ("name", "label", "after"),
    [
        ("Men's Kurta", "women", "men"),
        ("Men’s Kurta", "women", "men"),
        ("Women Kurta", "men", "women"),
        ("Boys Shorts", "women", "men"),
    ],
)
def test_genuine_leading_gender_still_corrected(name, label, after):
    d = gh.resolve_gender(name, label, "", [])
    assert (d.outcome, d.after) == (gh.CORRECTED, after)


def test_hyphenated_without_conflict_is_not_counted():
    # label already agrees with the hyphen piece: nothing the old rule would have changed
    assert gh.resolve_gender("Man-Made Tee", "men", "", []).outcome == gh.NO_CONFLICT
    assert gh.resolve_gender("T-Shirt Dress", "women", "", []).outcome == gh.NO_CONFLICT


def test_tags_decide_when_type_silent():
    agree = gh.resolve_gender("Men Shirt", "women", "Shirt", ["Men", "formal"])
    assert (agree.outcome, agree.after, agree.reason) == (
        gh.CORRECTED,
        "men",
        "tags_agrees_with_name",
    )
    supports = gh.resolve_gender("Men Shirt", "women", "Shirt", ["women", "formal"])
    assert (supports.outcome, supports.after, supports.reason) == (
        gh.AMBIGUOUS,
        "women",
        "tags_supports_label",
    )
    both = gh.resolve_gender("Men Shirt", "women", "Shirt", ["men", "women"])
    assert (both.outcome, both.reason) == (gh.AMBIGUOUS, "tags_both_genders")
