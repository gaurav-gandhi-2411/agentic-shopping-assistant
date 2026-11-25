"""F1 (sets demoted) and F2 (gender-interleaved) factor definitions."""

from __future__ import annotations

from evals.retrieval import factors as fx
from evals.retrieval.common import CANDIDATE_DIR, load_candidates, load_cases
from src.agents.outfit.slots import is_multi_piece_set


def _is_set(it: dict) -> bool:
    return is_multi_piece_set(it.get("product_type") or "", it.get("prod_name") or "")


def test_f1_flag_is_deterministic_from_occasion_and_garment() -> None:
    flagged = {c["id"] for c in load_cases() if fx.is_set_demotion_query(c)}
    assert flagged == {"q01", "q02", "q07", "q13", "q16", "q27", "q28", "q43"}
    by_id = {c["id"]: c for c in load_cases()}
    assert not fx.is_set_demotion_query(by_id["q03"])  # "palazzo kurta set": a set is wanted
    assert not fx.is_set_demotion_query(by_id["q15"])  # generic "outfit"
    assert not fx.is_set_demotion_query(by_id["q05"])  # reception is not an ethnic-wear occasion


def test_f2_flag_is_the_twelve_gender_neutral_fixtures() -> None:
    ids = [c["id"] for c in load_cases() if fx.is_gender_unspecified(c)]
    assert len(ids) == 12 and "q15" in ids and "q02" not in ids


def test_demote_sets_drops_sets_when_enough_singles_exist() -> None:
    items = load_candidates("q02", CANDIDATE_DIR)["items"]  # 12 singles + 8 sets
    after = fx.demote_sets(items)
    assert len(after) == 12 and not any(_is_set(it) for it in after)
    assert [it["article_id"] for it in after] == [
        it["article_id"] for it in items if not _is_set(it)
    ]  # stable: singles keep their relative retrieval order


def test_demote_sets_on_underflow_is_a_stable_partition_not_a_removal() -> None:
    items = load_candidates("q14", CANDIDATE_DIR)["items"]  # 4 singles + 16 sets (< 5 singles)
    after = fx.demote_sets(items)
    singles = [it["article_id"] for it in items if not _is_set(it)]
    sets = [it["article_id"] for it in items if _is_set(it)]
    assert len(singles) < 5 and len(after) == len(items) == 20
    assert [it["article_id"] for it in after] == singles + sets


def test_pool_loss_counts_removed_and_removed_relevant_items() -> None:
    before = [{"article_id": a} for a in ("a", "b", "c", "d")]
    after = [{"article_id": "a"}, {"article_id": "c"}]
    labels = {"a": 2, "b": 2, "c": 0, "d": 0}
    assert fx.pool_loss(before, after, labels) == {
        "n_before": 4,
        "n_after": 2,
        "n_removed": 2,
        "n_removed_relevant": 1,
    }


def _it(aid: str, gender: str) -> dict:
    return {"article_id": aid, "gender": gender}


def test_interleave_alternates_and_keeps_neutral_items_in_place() -> None:
    pool = [
        _it("w1", "women"),
        _it("w2", "women"),
        _it("m1", "men"),
        _it("u1", "unknown"),
        _it("m2", "men"),
        _it("w3", "women"),
    ]
    out = [it["article_id"] for it in fx.interleave_gender(pool)]
    assert out == ["w1", "m1", "w2", "u1", "m2", "w3"]


def test_interleave_starts_with_the_gender_of_the_best_ranked_item() -> None:
    pool = [_it("m1", "men"), _it("m2", "men"), _it("w1", "women"), _it("m3", "men")]
    out = [it["article_id"] for it in fx.interleave_gender(pool)]
    assert out == ["m1", "w1", "m2", "m3"]  # men first; women run out; remaining men follow


def test_interleave_is_a_permutation_and_a_noop_for_a_single_gender() -> None:
    mixed = load_candidates("q15", CANDIDATE_DIR)["items"]
    out = fx.interleave_gender(mixed)
    assert sorted(i["article_id"] for i in out) == sorted(i["article_id"] for i in mixed)
    genders = [i["gender"] for i in out if i["gender"] in ("women", "men")]
    n_pairs = 2 * min(genders.count("women"), genders.count("men"))  # alternation region
    assert all(a != b for a, b in zip(genders[:n_pairs], genders[1:n_pairs], strict=False))
    only_w = [_it("a", "women"), _it("b", "women")]
    assert fx.interleave_gender(only_w) == only_w
    assert fx.interleave_gender([]) == []
