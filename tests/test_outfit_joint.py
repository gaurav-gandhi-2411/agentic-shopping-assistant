"""Joint seed + slot selector: optimality vs brute force, determinism, budget, timeout, ties."""
from __future__ import annotations

import itertools
import random
import time

import pytest

from src.agents.outfit.joint import (
    AnchorPlan,
    JointTimeout,
    Option,
    SlotChoices,
    select_joint,
)


def _opt(aid: str, value: float, price: float, key: tuple[str, str] | None = None) -> Option:
    return Option({"article_id": aid}, value, price, key or (aid, ""))


def _brute(plans: list[AnchorPlan], budget: float) -> tuple[tuple[int, int, float], tuple[str, ...]]:
    """Exhaustive reference: best (required, cards, value) then smallest id tuple."""
    best: tuple[tuple[int, int, float], tuple[str, ...]] | None = None
    for plan in plans:
        a = plan.anchor
        if a.price > budget:
            continue
        axes = [[None, *s.options] for s in plan.slots]
        for combo in itertools.product(*axes):
            chosen = [c for c in combo if c is not None]
            items = [a, *chosen]
            if sum(o.price for o in items) > budget:
                continue
            if len({o.article_id for o in items}) != len(items):
                continue
            if len({o.key for o in items}) != len(items):
                continue
            req = sum(1 for s, c in zip(plan.slots, combo, strict=True) if c and s.required)
            rank = (req, len(items), round(sum(o.value for o in items), 9))
            ids = (a.article_id, *(c.article_id if c else "" for c in combo))
            if best is None or rank > best[0] or (rank == best[0] and ids < best[1]):
                best = (rank, ids)
    assert best is not None
    return best


def _random_plans(rng: random.Random) -> list[AnchorPlan]:
    plans = []
    for p in range(rng.randint(1, 3)):
        slots = []
        for s in range(rng.randint(1, 3)):
            opts = tuple(
                _opt(
                    f"p{p}s{s}o{o}",
                    rng.choice([0.25, 0.5, 0.75, 1.0]),  # coarse values force real ties
                    float(rng.randrange(100, 900, 50)),
                    key=(f"name{rng.randint(0, 3)}", "red"),  # collisions exercise the dedup key
                )
                for o in range(rng.randint(0, 4))
            )
            slots.append(SlotChoices(f"slot{s}", rng.random() < 0.6, opts))
        plans.append(
            AnchorPlan(
                _opt(f"a{p}", rng.choice([0.5, 1.0]), float(rng.randrange(200, 1200, 50)), ("a", str(p))),
                tuple(slots),
            )
        )
    return plans


def _solution_ids(sol, plans: list[AnchorPlan]) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    return (
        plans[sol.plan_index].anchor.article_id,
        *(p.article_id if p else "" for p in sol.picks),
    )


@pytest.mark.parametrize("seed", range(60))
def test_matches_brute_force_on_random_instances(seed: int) -> None:
    rng = random.Random(seed)
    plans = _random_plans(rng)
    budget = float(rng.randrange(400, 3000, 100))
    feasible = any(p.anchor.price <= budget for p in plans)
    sol = select_joint(plans, budget)
    if not feasible:
        assert sol is None
        return
    assert sol is not None
    want_rank, want_ids = _brute(plans, budget)
    assert (sol.required_filled, sol.cards, round(sol.value, 9)) == want_rank
    assert _solution_ids(sol, plans) == want_ids


def test_budget_is_never_exceeded() -> None:
    for seed in range(40):
        rng = random.Random(1000 + seed)
        plans = _random_plans(rng)
        budget = float(rng.randrange(400, 3000, 100))
        sol = select_joint(plans, budget)
        if sol is not None:
            assert sol.total_price <= budget


def test_required_slots_beat_a_more_valuable_optional_item() -> None:
    plan = AnchorPlan(
        _opt("a", 1.0, 1000),
        (
            SlotChoices("optional", False, (_opt("opt", 1.0, 900),)),
            SlotChoices("required", True, (_opt("req", 0.1, 800),)),
        ),
    )
    sol = select_joint([plan], 2000)
    assert sol is not None
    assert [p.article_id if p else None for p in sol.picks] == [None, "req"]
    assert sol.required_filled == 1


def test_fractional_prices_summing_exactly_to_the_budget_are_accepted() -> None:
    # 0.1 + 0.2 == 0.30000000000000004 in binary floats: must still fit a budget of exactly 0.3
    plan = AnchorPlan(_opt("a", 1.0, 0.1), (SlotChoices("s", True, (_opt("x", 1.0, 0.2),)),))
    sol = select_joint([plan], 0.3)
    assert sol is not None and sol.cards == 2
    big = AnchorPlan(
        _opt("a", 1.0, 1000.10), (SlotChoices("s", True, (_opt("x", 1.0, 1499.90),)),)
    )
    sol = select_joint([big], 2500.0)
    assert sol is not None and sol.cards == 2 and sol.total_price == 2500.0


def test_one_paisa_over_the_budget_is_rejected() -> None:
    plan = AnchorPlan(_opt("a", 1.0, 0.1), (SlotChoices("s", True, (_opt("x", 1.0, 0.21),)),))
    sol = select_joint([plan], 0.3)
    assert sol is not None and sol.cards == 1  # the 0.21 item would total 0.31
    over = AnchorPlan(_opt("a", 1.0, 2500.01), ())
    assert select_joint([over], 2500.0) is None


def test_one_required_slot_beats_more_optional_cards() -> None:
    """Two cheap optional items would make more cards, but the required slot comes first."""
    plan = AnchorPlan(
        _opt("a", 1.0, 100),
        (
            SlotChoices("opt1", False, (_opt("o1", 1.0, 100),)),
            SlotChoices("opt2", False, (_opt("o2", 1.0, 100),)),
            SlotChoices("req", True, (_opt("r", 0.1, 250),)),
        ),
    )
    sol = select_joint([plan], 400)
    assert sol is not None
    assert [p.article_id if p else None for p in sol.picks] == [None, None, "r"]
    assert (sol.required_filled, sol.cards) == (1, 2)


def test_greedy_trap_is_solved_jointly() -> None:
    """The first slot's best item (1599) leaves nothing for the second; the joint pick does."""
    plan = AnchorPlan(
        _opt("shirt", 1.0, 700),
        (
            SlotChoices("bottom", True, (_opt("pricey", 1.0, 1599), _opt("cheap", 0.8, 599))),
            SlotChoices("footwear", True, (_opt("shoes", 1.0, 899),)),
        ),
    )
    sol = select_joint([plan], 2500)
    assert sol is not None and sol.cards == 3
    assert [p.article_id if p else None for p in sol.picks] == ["cheap", "shoes"]
    assert sol.total_price == 700 + 599 + 899


def test_deterministic_across_repeated_runs_and_input_order() -> None:
    rng = random.Random(7)
    plans = _random_plans(rng)
    first = select_joint(plans, 2000)
    assert first is not None
    for _ in range(5):
        again = select_joint(plans, 2000)
        assert again is not None and _solution_ids(again, plans) == _solution_ids(first, plans)
    # options of one slot given in reverse order: same winner
    flipped = [
        AnchorPlan(
            p.anchor, tuple(SlotChoices(s.name, s.required, s.options[::-1]) for s in p.slots)
        )
        for p in plans
    ]
    other = select_joint(flipped, 2000)
    assert other is not None and _solution_ids(other, flipped) == _solution_ids(first, plans)


def test_ties_break_to_the_smaller_article_id() -> None:
    plan = AnchorPlan(
        _opt("a", 1.0, 100),
        (SlotChoices("s", True, (_opt("zz", 0.5, 100), _opt("bb", 0.5, 100), _opt("mm", 0.5, 100))),),
    )
    sol = select_joint([plan], 1000)
    assert sol is not None and sol.picks[0] is not None and sol.picks[0].article_id == "bb"


def test_same_article_or_dedup_key_is_never_used_twice() -> None:
    shared = _opt("x", 1.0, 100, ("kurta", "red"))
    plan = AnchorPlan(
        _opt("a", 1.0, 100, ("kurta", "red")),  # anchor shares its key with `shared`
        (
            SlotChoices("s1", True, (shared, _opt("y", 0.4, 100, ("other", "blue")))),
            SlotChoices("s2", True, (_opt("x", 1.0, 100, ("zzz", "red")),)),  # same id as shared
        ),
    )
    sol = select_joint([plan], 1000)
    assert sol is not None
    ids = [p.article_id for p in sol.picks if p]
    assert len(ids) == len(set(ids))
    assert "x" not in ids or ids.count("x") == 1
    assert ("kurta", "red") not in [p.key for p in sol.picks if p]


def test_no_anchor_within_budget_returns_none() -> None:
    plan = AnchorPlan(_opt("a", 1.0, 5000), ())
    assert select_joint([plan], 1000) is None
    assert select_joint([], 1000) is None


def test_timeout_raises_instead_of_returning_a_partial_answer() -> None:
    # 4 slots x 12 distinct options each: far more than _CHECK_EVERY nodes.
    slots = tuple(
        SlotChoices(
            f"s{s}", True, tuple(_opt(f"s{s}o{o}", 1.0 - o * 0.001, 10.0, (f"n{s}{o}", "")) for o in range(12))
        )
        for s in range(4)
    )
    plans = [AnchorPlan(_opt(f"a{p}", 1.0, 10.0, ("a", str(p))), slots) for p in range(30)]
    with pytest.raises(JointTimeout):
        select_joint(plans, 10_000.0, deadline=time.monotonic() - 1.0)


def test_bounded_search_prunes_a_large_instance_quickly() -> None:
    slots = tuple(
        SlotChoices(
            f"s{s}", True, tuple(_opt(f"s{s}o{o}", 1.0 - o * 0.01, 50.0, (f"n{s}{o}", "")) for o in range(20))
        )
        for s in range(4)
    )
    plans = [AnchorPlan(_opt("a", 1.0, 100.0), slots)]
    t0 = time.monotonic()
    sol = select_joint(plans, 10_000.0, deadline=time.monotonic() + 5.0)
    assert sol is not None and sol.cards == 5
    assert sol.nodes < 21**4 // 10  # pruning, not a full 194k-node enumeration
    assert time.monotonic() - t0 < 5.0
