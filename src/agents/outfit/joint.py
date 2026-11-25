"""Joint seed + slot selection under a total budget (pure search, no retrieval, no LLM).

The greedy composer picks the anchor first and then fills slots one at a time, so an early pick
can spend money a later required slot needs (a Rs 1,599 bottom leaves nothing for footwear). This
module chooses the anchor and every slot item TOGETHER from bounded candidate lists.

Objective, compared lexicographically (higher wins):
    1. number of required slots filled,
    2. number of cards (anchor + filled slots),
    3. sum of item values (each in (0, 1], supplied by the caller).
Ties on all three are broken by the tuple of article ids (anchor first, then each slot's pick in
slot order, "" for an empty slot), smaller wins, so the result is deterministic.

Hard constraints: total price <= budget, no article twice, no repeated dedup key (the composer
passes (normalised product name, colour)).

Complexity: per anchor plan, a depth-first branch-and-bound over ``len(slots)`` levels with up to
``N`` options per slot (plus "skip"), worst case O((N + 1) ** slots) nodes; the composer uses
N <= 20 and <= 4 slots, i.e. <= 21 ** 4 = 194,481 nodes per plan worst case. Options are visited
best-value-first and a node is cut when even the optimistic completion (every remaining slot
filled with its best-valued option, ignoring price) cannot beat the incumbent, so typical plans
visit a few hundred nodes. A wall-clock deadline is checked every ``_CHECK_EVERY`` nodes and
raises :class:`JointTimeout`; the caller falls back to the greedy path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

_CHECK_EVERY: int = 256
_VALUE_DECIMALS: int = 9  # float sums are rounded before comparison so ties are real ties


class JointTimeout(Exception):  # noqa: N818 - reads as a control-flow signal, not an error class
    """The wall-clock budget ran out before the search finished."""


@dataclass(frozen=True)
class Option:
    """One selectable item: the item dict, its value in (0, 1], its price and its dedup key."""

    item: dict
    value: float
    price: float
    key: tuple[str, str] = ("", "")

    @property
    def article_id(self) -> str:
        return str(self.item.get("article_id", ""))


@dataclass(frozen=True)
class SlotChoices:
    """A slot and its candidate options (already gated by the caller)."""

    name: str
    required: bool
    options: tuple[Option, ...]


@dataclass(frozen=True)
class AnchorPlan:
    """One anchor candidate with the slots (and slot options) that go with it."""

    anchor: Option
    slots: tuple[SlotChoices, ...]


@dataclass(frozen=True)
class JointSolution:
    """The winning combination: which plan, and the pick (or None) for each of its slots."""

    plan_index: int
    picks: tuple[Option | None, ...]
    required_filled: int
    cards: int
    value: float
    total_price: float
    nodes: int


def _rank(required_filled: int, cards: int, value: float) -> tuple[int, int, float]:
    return (required_filled, cards, round(value, _VALUE_DECIMALS))


def select_joint(
    plans: list[AnchorPlan],
    budget: float,
    *,
    deadline: float | None = None,
) -> JointSolution | None:
    """Best (anchor, slot items) over all ``plans`` within ``budget``; None if no anchor fits.

    ``deadline`` is a ``time.monotonic()`` timestamp; exceeding it raises :class:`JointTimeout`.
    """
    best: JointSolution | None = None
    best_ids: tuple[str, ...] = ()
    nodes = 0

    for plan_index, plan in enumerate(plans):
        if plan.anchor.price > budget:
            continue
        slots = [
            SlotChoices(
                s.name,
                s.required,
                tuple(
                    sorted(
                        (o for o in s.options if o.price + plan.anchor.price <= budget),
                        key=lambda o: (-o.value, o.article_id),
                    )
                ),
            )
            for s in plan.slots
        ]
        n = len(slots)
        # Optimistic suffix bounds (price ignored): how much the slots from i on could still add.
        suf_req = [0] * (n + 1)
        suf_cards = [0] * (n + 1)
        suf_val = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            has = bool(slots[i].options)
            suf_req[i] = suf_req[i + 1] + (1 if has and slots[i].required else 0)
            suf_cards[i] = suf_cards[i + 1] + (1 if has else 0)
            suf_val[i] = suf_val[i + 1] + (slots[i].options[0].value if has else 0.0)

        picks: list[Option | None] = [None] * n
        a_opt = plan.anchor

        def walk(
            i: int,
            spent: float,
            used_ids: frozenset[str],
            used_keys: frozenset[tuple[str, str]],
            req: int,
            cards: int,
            value: float,
            *,
            _plan_index: int = plan_index,
            _slots: list[SlotChoices] = slots,
            _picks: list[Option | None] = picks,
            _suf_req: list[int] = suf_req,
            _suf_cards: list[int] = suf_cards,
            _suf_val: list[float] = suf_val,
            _a_opt: Option = a_opt,
            _n: int = n,
        ) -> None:
            nonlocal best, best_ids, nodes
            nodes += 1
            if deadline is not None and nodes % _CHECK_EVERY == 0 and time.monotonic() > deadline:
                raise JointTimeout(f"joint search exceeded its wall budget after {nodes} nodes")
            if best is not None:
                bound = _rank(req + _suf_req[i], cards + _suf_cards[i], value + _suf_val[i])
                if bound < _rank(best.required_filled, best.cards, best.value):
                    return
            if i == _n:
                ids = (_a_opt.article_id, *(p.article_id if p else "" for p in _picks))
                cand = _rank(req, cards, value)
                if best is None:
                    better = True
                else:
                    cur = _rank(best.required_filled, best.cards, best.value)
                    better = cand > cur or (cand == cur and ids < best_ids)
                if better:
                    best_ids = ids
                    best = JointSolution(
                        plan_index=_plan_index,
                        picks=tuple(_picks),
                        required_filled=req,
                        cards=cards,
                        value=value,
                        total_price=spent,
                        nodes=0,
                    )
                return
            slot = _slots[i]
            for opt in slot.options:
                if spent + opt.price > budget:
                    continue
                if opt.article_id in used_ids or opt.key in used_keys:
                    continue
                _picks[i] = opt
                walk(
                    i + 1,
                    spent + opt.price,
                    used_ids | {opt.article_id},
                    used_keys | {opt.key},
                    req + (1 if slot.required else 0),
                    cards + 1,
                    value + opt.value,
                )
            _picks[i] = None
            walk(i + 1, spent, used_ids, used_keys, req, cards, value)

        walk(
            0,
            a_opt.price,
            frozenset({a_opt.article_id}),
            frozenset({a_opt.key}),
            0,
            1,
            a_opt.value,
        )

    if best is None:
        return None
    return JointSolution(
        plan_index=best.plan_index,
        picks=best.picks,
        required_filled=best.required_filled,
        cards=best.cards,
        value=best.value,
        total_price=best.total_price,
        nodes=nodes,
    )
