"""Grounding by construction for LLM reply text (V5).

The LLM refers to catalogue items only by id (``[item:ID]``, ``[price:ID]``); this module renders
names and prices from the catalogue record, then removes any sentence that still carries a price /
size / stock / discount / delivery claim the shown records do not back, or that references an id
outside the shown set. Validation fails closed: on any error, or when nothing survives, the free
text is replaced by a template sentence built from the records.

    ground_reply(text, shown_items, user_text=..., fallback=...) -> GroundResult

Pure and deterministic (no I/O, no LLM). Kill switch: env ``GROUNDED_RENDERING`` set to
``0``/``false``/``off``/``no`` disables every caller; it is read on each call.
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.agents.claim_checker import _extract_claims, _mentions, _sentences, check
from src.agents.grounding import grounded_fallback_reply

logger = logging.getLogger(__name__)

KILL_SWITCH_ENV = "GROUNDED_RENDERING"
_OFF_VALUES = frozenset({"0", "false", "off", "no"})

# [item:123] / [price:123]; the id charset is deliberately loose so a malformed or invented id is
# still recognised as a token (and then rejected as unknown) instead of leaking into the text.
_TOKEN_RE = re.compile(r"\[\s*(?P<kind>[A-Za-z_]+)\s*:\s*(?P<id>[^\]\s]{1,80})\s*\]")
_STRAY_RE = re.compile(r"\[\s*(?:item|price)\s*:", re.IGNORECASE)
_NON_PRICE_KINDS = frozenset({"size", "stock", "percent", "discount", "delivery"})

RESPOND_ID_RULES = """\
REFERRING TO ITEMS (this overrides any instruction above about naming items or quoting prices):
- Each item below has an id. Refer to an item ONLY by writing [item:ID] with its id, for example \
"the [item:123] is a great pick". Do not type an item's name yourself.
- To mention an item's price write [price:ID], for example "[item:123] at [price:123]". Never \
type a price, size, stock level, discount, offer or delivery detail yourself; the app fills in \
names and prices from the catalogue.
- Only use ids from the list below."""


def grounded_rendering_enabled() -> bool:
    """False when the operator set GROUNDED_RENDERING to 0/false/off/no (default: enabled)."""
    return os.environ.get(KILL_SWITCH_ENV, "1").strip().lower() not in _OFF_VALUES


def add_id_rules(prompt: str) -> str:
    """Insert the id-reference rules just before the prompt's closing instruction."""
    marker = "Write your response now."
    if marker in prompt:
        return prompt.replace(marker, RESPOND_ID_RULES + "\n\n" + marker)
    return prompt + "\n\n" + RESPOND_ID_RULES


@dataclass(frozen=True)
class GroundResult:
    """Outcome of grounding one reply."""

    text: str
    flags: list[str] = field(default_factory=list)
    replaced: bool = False  # the whole text was swapped for the template sentence
    failed_closed: bool = False  # ... because validation itself raised


def _record_name(item: dict[str, Any]) -> str:
    return str(item.get("display_name") or item.get("prod_name") or "").strip()


def _record_price(item: dict[str, Any]) -> str | None:
    price = item.get("price_inr")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
        return None
    return f"₹{price:,.0f}"


def _render_sentence(sentence: str, by_id: dict[str, dict[str, Any]], flags: list[str]) -> str | None:
    """Substitute tokens in one sentence; None when the sentence must be dropped."""
    bad = False

    def sub(m: re.Match[str]) -> str:
        nonlocal bad
        kind = m.group("kind").lower()
        item = by_id.get(m.group("id"))
        if item is None:
            flags.append(f"unknown_id:{m.group('id')}")
            bad = True
            return ""
        if kind == "item":
            rendered = _record_name(item)
        elif kind == "price":
            rendered = _record_price(item) or ""
        else:
            flags.append(f"unsupported_token:{kind}")
            bad = True
            return ""
        if not rendered:
            flags.append(f"no_record_value:{kind}:{m.group('id')}")
            bad = True
        return rendered

    out = _TOKEN_RE.sub(sub, sentence)
    if not bad and _STRAY_RE.search(out):
        flags.append("stray_token")
        bad = True
    return None if bad else out


def ground_reply(
    text: str,
    shown_items: Sequence[dict[str, Any]],
    *,
    user_text: str = "",
    extra_amounts: Iterable[float] = (),
    fallback: str | None = None,
) -> GroundResult:
    """Render id tokens from the records, strip unbacked claims, fail closed.

    Args:
        text: raw LLM text, possibly containing ``[item:ID]`` / ``[price:ID]`` tokens.
        shown_items: the item records the reply may talk about (ids = ``article_id``).
        user_text: the user's message; only used by the checker's budget / denial echo exemptions.
        extra_amounts: code-computed money figures rendered alongside the reply.
        fallback: template sentence used when nothing survives or validation raises; defaults to
            ``grounded_fallback_reply(shown_items)``.
    """
    if not text or not text.strip():
        return GroundResult(text)
    flags: list[str] = []
    try:
        by_id = {str(it.get("article_id")): it for it in shown_items if it.get("article_id") is not None}
        spans = _sentences(text)
        rendered: list[str | None] = [
            _render_sentence(text[s:e], by_id, flags) for s, e in spans
        ]
        keep = [r is not None for r in rendered]

        kept_text = "".join(r for r in rendered if r is not None)
        offsets: list[tuple[int, int, int]] = []  # (start, end, span index) in kept_text
        pos = 0
        for i, r in enumerate(rendered):
            if r is not None:
                offsets.append((pos, pos + len(r), i))
                pos += len(r)
        for v in check(kept_text, shown_items, user_text=user_text, extra_amounts=extra_amounts):
            flags.append(f"claim:{v.kind}")
            for start, end, i in offsets:
                if start <= v.start < end or start < v.end <= end:
                    keep[i] = False

        _drop_unattributed(rendered, keep, shown_items, user_text, extra_amounts, flags)

        if all(keep):
            # nothing dropped: the original text with only the tokens substituted
            return GroundResult(_TOKEN_RE.sub(lambda m: _render_token(m, by_id), text), flags)
        survivors = "".join(r for r, k in zip(rendered, keep) if k and r is not None).strip()
        if survivors:
            return GroundResult(survivors, flags)
        return GroundResult(
            fallback if fallback is not None else grounded_fallback_reply(list(shown_items)),
            [*flags, "all_dropped"],
            replaced=True,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY validator error
        logger.warning("[grounding] validator error, serving template: %r", exc)
        return GroundResult(
            fallback if fallback is not None else grounded_fallback_reply(list(shown_items)),
            [*flags, "validator_error"],
            replaced=True,
            failed_closed=True,
        )


def _drop_unattributed(
    rendered: list[str | None],
    keep: list[bool],
    items: Sequence[dict[str, Any]],
    user_text: str,
    extra_amounts: Iterable[float],
    flags: list[str],
) -> None:
    """Tighten sentences that make a claim without naming an item ("It is in stock in size M.").

    check() judges a sentence that names no item against the WHOLE shown set, so one item backing
    the claim would excuse it for all. Here such a sentence is judged against the items of the
    nearest preceding sentence that named some (the usual "It ..." follow-up); with no such
    sentence it must hold for EVERY shown item on the non-price kinds (size / stock / discount /
    delivery). Mutates ``keep`` in place.
    """
    extras = list(extra_amounts)
    prev_scope: list[dict[str, Any]] = []
    for i, sent in enumerate(rendered):
        if sent is None or not keep[i]:
            continue
        groups, _ = _mentions(sent, list(items))
        named = [it for _, its in groups for it in its]
        if named:
            prev_scope = named
            continue
        kinds = {c.kind for c in _extract_claims(sent)}
        if not kinds:
            continue
        if prev_scope:
            bad = [v for v in check(sent, prev_scope, user_text=user_text, extra_amounts=extras)]
        else:
            bad = [
                v for it in items
                for v in check(sent, [it], user_text=user_text, extra_amounts=extras)
                if v.kind in _NON_PRICE_KINDS
            ]
        if bad:
            keep[i] = False
            flags.append(f"unattributed_claim:{bad[0].kind}")


def _render_token(m: re.Match[str], by_id: dict[str, dict[str, Any]]) -> str:
    """Render one already-validated token (kind/id known good by _render_sentence)."""
    item = by_id[m.group("id")]
    return _record_name(item) if m.group("kind").lower() == "item" else (_record_price(item) or "")
