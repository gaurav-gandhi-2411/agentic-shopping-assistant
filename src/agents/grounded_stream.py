"""Incremental grounding for a streamed reply (V5 amendment 1).

    g = StreamGrounder(shown_items, user_text=...)
    for chunk in model_stream:
        for piece in g.feed(chunk):
            send(piece)
    for piece in g.finish():
        send(piece)

Tokens pass through as they arrive (never past the last whitespace, so a word or tag split across
chunks is never half-emitted). A closed ``[item:ID]`` with a known id is rendered to the record's
name and streaming continues. From the first claim-like trigger of a sentence (a non-item ``[``,
a digit or currency symbol, claim vocabulary, or a hit of the claim checker) the rest of the
sentence is HELD until the sentence ends, then judged with the batch rules and emitted whole or
not at all. See docs/grounding-v5-acceptance.md, AMENDMENT 1, for the exact guarantee: the
claim-free lead-in before a sentence's first trigger can reach the user even if the sentence is
then removed.

Pure and deterministic (no I/O besides one structured log line, no LLM).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

from src.agents.claim_checker import _extract_claims, _mentions, _sentences, check
from src.agents.grounded_render import (
    _NON_PRICE_KINDS,
    _TOKEN_RE,
    GroundResult,
    _record_name,
    _render_sentence,
)
from src.agents.grounding import grounded_fallback_reply

logger = logging.getLogger(__name__)

# Claim-like triggers. A superset of the words the claim checker keys on, deliberately broad: a
# false trigger only delays text to the end of its sentence. Anchored at word starts (no trailing
# \b on stems) so a prefix such as "unavail" or "deliver" fires before the word is complete.
_TRIGGER_RE = re.compile(
    r"\[|[\d%₹$€£]|"
    r"\b(?:stock|sold|sizes?|sale|offers?|off\b|discount|price|cost|deliver|ship|pick[\s\-]?up|"
    r"collect|free|express|save|saving|half|left\b|(?:un)?avail|medium|large|small|xl|xs|xxl|fit|"
    r"runs?\b|true\b|returns?|cod\b|bogo|coupon|clearance|markdown|deal|bargain|promo|cash|flat|"
    r"limited|hurry|restock|ready|dispatch|arriv|mrp|tight|petite|rs\b|inr|usd|rupee|bucks)",
    re.IGNORECASE,
)
_UNCLOSED_EVENT = "grounded_unclosed_span"


class _StreamStateError(RuntimeError):
    """The incremental view diverged from what was already emitted (fail closed)."""


class StreamGrounder:
    """Feed raw model chunks, get back the text that is safe to send now."""

    def __init__(
        self,
        shown_items: Sequence[dict[str, Any]],
        *,
        user_text: str = "",
        extra_amounts: Iterable[float] = (),
        fallback: str | None = None,
        hold_sentences: bool = False,
    ) -> None:
        # hold_sentences=True emits each sentence only once it has been judged: no lead-in
        # fragments (guarantee G2 disappears) at the cost of sentence-sized frames.
        self._hold_sentences = hold_sentences
        self._items = list(shown_items)
        self._by_id = {
            str(it.get("article_id")): it for it in self._items if it.get("article_id") is not None
        }
        self._user_text = user_text
        self._extras = list(extra_amounts)
        self._fallback = fallback
        self._raw = ""  # raw text of the sentence in progress (and any not yet split off)
        self._emitted_view = ""  # view-coordinates text of the current sentence already consumed
        self._prev_scope: list[dict[str, Any]] = []
        self._sent: list[str] = []  # everything handed to the caller, in order
        self._kept_any = False
        self._dropped_any = False
        self._seen_text = False
        self._dead = False
        self.flags: list[str] = []
        self.replaced = False
        self.failed_closed = False
        self.fragments = 0  # sentences removed after a claim-free lead-in was already emitted
        self.unclosed_spans = 0

    # ------------------------------------------------------------------ public API

    @property
    def text(self) -> str:
        """Everything emitted so far; equals what the user saw and what is persisted."""
        return "".join(self._sent)

    def result(self) -> GroundResult:
        """Summary in the batch result shape (text, flags, replaced, failed_closed)."""
        return GroundResult(self.text, list(self.flags), self.replaced, self.failed_closed)

    def feed(self, chunk: str) -> list[str]:
        """Add a raw chunk; return the pieces now safe to emit."""
        if self._dead or not chunk:
            return []
        if chunk.strip():
            self._seen_text = True
        try:
            self._raw += chunk
            return self._advance(final=False)
        except Exception as exc:  # noqa: BLE001 - fail closed on ANY validator error
            self._fail(exc)
            return []

    def finish(self) -> list[str]:
        """End of stream: judge the last sentence, log unclosed spans, add the template if needed."""
        out: list[str] = []
        if not self._dead:
            try:
                self._log_unclosed()
                out += self._advance(final=True)
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)
        if self._seen_text and not self._kept_any:
            fallback = (
                self._fallback
                if self._fallback is not None
                else grounded_fallback_reply(self._items)
            )
            self.flags.append("all_dropped")
            self.replaced = True
            lead = "\n\n" if self.text.strip() else ""  # keep an already-sent fragment apart
            out += self._send(lead + fallback, raw_consumed=False)
        return out

    # ------------------------------------------------------------------ internals

    def _fail(self, exc: Exception) -> None:
        logger.warning("[grounding] stream validator error, emitting nothing more: %r", exc)
        self._dead = True
        self.failed_closed = True
        self.flags.append("validator_error")

    def _send(self, text: str, *, raw_consumed: bool = True) -> list[str]:
        """Record and return a piece; leading whitespace is dropped once a sentence was removed."""
        if raw_consumed and self._dropped_any and not self.text.strip():
            text = text.lstrip()
        if not text:
            return []
        self._sent.append(text)
        return [text]

    def _advance(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while self._raw.strip():
            spans = _sentences(self._raw)
            if len(spans) >= 2 or (final and spans):
                end = spans[0][1]
                sentence, self._raw = self._raw[:end], self._raw[end:]
                out += self._complete(sentence)
                continue
            break
        if not final:
            out += self._emit_partial()
        return out

    def _view(self, raw: str) -> str:
        """The sentence so far with closed, known ``[item:ID]`` spans rendered to names."""

        def sub(m: re.Match[str]) -> str:
            item = self._by_id.get(m.group("id"))
            is_item = item is not None and m.group("kind").lower() == "item"
            name = _record_name(item) if item is not None and is_item else ""
            return name or m.group(0)

        return _TOKEN_RE.sub(sub, raw)

    def _emit_partial(self) -> list[str]:
        if self._hold_sentences:
            return []
        view = self._view(self._raw)
        done = len(self._emitted_view)
        if view[:done] != self._emitted_view:
            raise _StreamStateError("view prefix changed after emission")
        starts = [m.start() for m in [_TRIGGER_RE.search(view)] if m]
        starts += [c.start for c in _extract_claims(view)[:1]]
        limit = min(starts) if starts else len(view)
        runs = list(re.finditer(r"\s+", view[:limit] if starts else view))
        # Emit up to the end of the last COMPLETE word: never the word in progress, and never the
        # whitespace after it (that whitespace may belong to the next sentence once this one is
        # cut off, so it travels with the next piece instead).
        safe_end = runs[-1].start() if runs else 0
        if safe_end <= done:
            return []
        self._emitted_view = view[:safe_end]
        return self._send(view[done:safe_end])

    def _complete(self, sentence: str) -> list[str]:
        """Judge a finished sentence; emit its un-emitted remainder, or nothing."""
        rendered = self._judge(sentence)
        emitted = self._emitted_view
        self._emitted_view = ""
        if rendered is None:
            self._dropped_any = True
            if emitted.strip():
                self.fragments += 1
            return []
        if not rendered.startswith(emitted):
            raise _StreamStateError("rendered sentence does not extend the emitted lead-in")
        self._kept_any = True
        return self._send(rendered[len(emitted) :])

    def _judge(self, sentence: str) -> str | None:
        """Same rules as ground_reply, applied to one sentence; None = drop it."""
        rendered = _render_sentence(sentence, self._by_id, self.flags)
        if rendered is None:
            return None
        violations = check(
            rendered, self._items, user_text=self._user_text, extra_amounts=self._extras
        )
        if violations:
            self.flags += [f"claim:{v.kind}" for v in violations]
            return None
        groups, _ = _mentions(rendered, self._items)
        named = [it for _, its in groups for it in its]
        if named:
            self._prev_scope = named
            return rendered
        if not {c.kind for c in _extract_claims(rendered)}:
            return rendered
        if self._prev_scope:
            bad = check(
                rendered, self._prev_scope, user_text=self._user_text, extra_amounts=self._extras
            )
        else:
            bad = [
                v
                for it in self._items
                for v in check(
                    rendered, [it], user_text=self._user_text, extra_amounts=self._extras
                )
                if v.kind in _NON_PRICE_KINDS
            ]
        if bad:
            self.flags.append(f"unattributed_claim:{bad[0].kind}")
            return None
        return rendered

    def _log_unclosed(self) -> None:
        """Count '[' openers after the last ']' of the unfinished tail; log, never the text."""
        tail = self._raw
        count = tail[tail.rfind("]") + 1 :].count("[")
        if count:
            self.unclosed_spans = count
            logger.warning(json.dumps({"event": _UNCLOSED_EVENT, "count": count}))
