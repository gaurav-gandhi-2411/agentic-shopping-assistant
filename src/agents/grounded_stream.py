"""Incremental grounding for a streamed reply (V5 amendments 1 to 3).

    g = StreamGrounder(shown_items, user_text=...)
    for chunk in model_stream:
        for piece in g.feed(chunk):
            send(piece)
    for piece in g.finish():
        send(piece)

Tokens pass through as they arrive (never past the last whitespace, so a word or tag split across
chunks is never half-emitted). A closed ``[item:ID]`` with a known id is rendered to the record's
name and streaming continues. From the first claim-like trigger of a sentence (a non-item ``[``,
a digit, number word or currency symbol, claim vocabulary, or a hit of the claim checker) the rest
of the sentence is HELD until the sentence ends, then judged with the batch rules and emitted whole
or not at all. See docs/grounding-v5-acceptance.md, AMENDMENT 1, for the exact guarantee: the
claim-free lead-in before a sentence's first trigger can reach the user even if the sentence is
then removed. AMENDMENT 3: a sentence in progress is capped at 800 characters (the rest of the
stream is discarded, fail closed) and a validator error always ends in the template sentence.

Pure and deterministic (no I/O besides structured log lines, no LLM). The constructor only stores
its arguments, so building one cannot raise.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

from src.agents.claim_checker import (
    _NW_ALT,
    _extract_claims,
    _mentions,
    _sentences,
    check,
)
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
# Amendment 3 adds number words, percent / per cent, buy, quarter / third and fraction characters.
_TRIGGER_RE = re.compile(
    r"\[|[\d%₹$€£½¼¾⅓⅔⅛]|"
    r"\b(?:stock|sold|sizes?|sale|offers?|off\b|discount|price|cost|deliver|ship|pick[\s\-]?up|"
    r"collect|free|express|save|saving|half|quarter|third|left\b|(?:un)?avail|medium|large|small|"
    r"xl|xs|xxl|fit|runs?\b|true\b|returns?|cod\b|bogo|coupon|clearance|markdown|deal|bargain|"
    r"promo|cash|flat|limited|hurry|restock|ready|dispatch|arriv|mrp|tight|petite|rs\b|inr|usd|"
    r"rupee|bucks|percent|per\b|buy)"
    rf"|\b(?:{_NW_ALT})\b",
    re.IGNORECASE,
)
_UNCLOSED_EVENT = "grounded_unclosed_span"
# Hard cap on the sentence in progress. Every feed rescans it, so cost grows quadratically:
# measured with 7-character chunks, 800 characters cost 141 ms in total (worst feed 2.4 ms) and
# 4000 characters cost 3688 ms (worst feed 17 ms). Real sentences are well under 400 characters.
MAX_OPEN_SENTENCE_CHARS = 800


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
        # Only stores arguments (no work that can raise); lookups are built lazily inside the
        # guarded paths so a bad item cannot break the turn outside fail-closed handling.
        # hold_sentences=True emits each sentence only once it has been judged: no lead-in
        # fragments (guarantee G2 disappears) at the cost of sentence-sized frames.
        self._hold_sentences = hold_sentences
        self._shown = shown_items
        self._items: list[dict[str, Any]] = []
        self._by_id_cache: dict[str, dict[str, Any]] | None = None
        self._user_text = user_text
        self._extras = extra_amounts
        self._extras_list: list[float] = []
        self._fallback = fallback
        self._raw = ""  # raw text of the sentence in progress (and any not yet split off)
        self._emitted_view = ""  # view-coordinates text of the current sentence already consumed
        self._prev_scope: list[dict[str, Any]] = []
        self._sent: list[str] = []  # everything handed to the caller, in order
        self._outbox: list[str] = []  # pieces not yet returned by feed()/finish()
        self._kept_any = False
        self._dropped_any = False
        self._seen_text = False
        self._dead = False
        self._discarding = False
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
        if self._discarding:  # past the hard cap: the rest of the stream is dropped
            return []
        try:
            self._raw += chunk
            self._advance(final=False)
        except Exception as exc:  # noqa: BLE001 - fail closed on ANY validator error
            self._fail(exc)
        return self._drain()

    def finish(self) -> list[str]:
        """End of stream: judge the last sentence, log unclosed spans, add the template if needed."""
        if not self._dead:
            try:
                self._log_unclosed()
                self._advance(final=True)
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)
        if self._seen_text and (not self._kept_any or self.failed_closed):
            try:
                fallback = (
                    self._fallback
                    if self._fallback is not None
                    else grounded_fallback_reply(list(self._shown))
                )
            except Exception:  # noqa: BLE001 - even the template must not break the turn
                fallback = "Sorry, I could not put a reply together. Please try again."
            self.flags.append("all_dropped" if not self._kept_any else "template_after_error")
            self.replaced = True
            lead = "\n\n" if self.text.strip() else ""  # keep an already-sent fragment apart
            self._send(lead + fallback, raw_consumed=False)
        return self._drain()

    # ------------------------------------------------------------------ internals

    def _ids(self) -> dict[str, dict[str, Any]]:
        if self._by_id_cache is None:
            self._items = list(self._shown)
            self._extras_list = list(self._extras)
            self._by_id_cache = {
                str(it.get("article_id")): it
                for it in self._items
                if it.get("article_id") is not None
            }
        return self._by_id_cache

    def _drain(self) -> list[str]:
        out, self._outbox = self._outbox, []
        return out

    def _fail(self, exc: Exception) -> None:
        logger.warning("[grounding] stream validator error, emitting nothing more: %r", exc)
        self._dead = True
        self.failed_closed = True
        self.flags.append("validator_error")

    def _send(self, text: str, *, raw_consumed: bool = True) -> None:
        """Record a piece; leading whitespace is dropped once a sentence was removed."""
        if raw_consumed and self._dropped_any and not self.text.strip():
            text = text.lstrip()
        if text:
            self._sent.append(text)
            self._outbox.append(text)

    def _advance(self, *, final: bool) -> None:
        self._ids()
        while self._raw.strip():
            spans = _sentences(self._raw)
            if len(spans) >= 2 or (final and spans):
                end = spans[0][1]
                sentence, self._raw = self._raw[:end], self._raw[end:]
                self._complete(sentence)
                continue
            break
        if len(self._raw) > MAX_OPEN_SENTENCE_CHARS:
            self._force_cut()
        if not final:
            self._emit_partial()

    def _force_cut(self) -> None:
        """Hard cap: judge the sentence so far (cut at the last whitespace), discard the rest."""
        head = self._raw[:MAX_OPEN_SENTENCE_CHARS]
        spaces = [m.start() for m in re.finditer(r"\s", head)]
        cut = spaces[-1] if spaces else MAX_OPEN_SENTENCE_CHARS
        sentence, self._raw = self._raw[:cut], ""
        self._discarding = True
        self.flags.append("overlong_sentence")
        self._complete(sentence)

    def _view(self, raw: str) -> str:
        """The sentence so far with closed, known ``[item:ID]`` spans rendered to names."""
        by_id = self._ids()

        def sub(m: re.Match[str]) -> str:
            item = by_id.get(m.group("id"))
            is_item = item is not None and m.group("kind").lower() == "item"
            name = _record_name(item) if item is not None and is_item else ""
            return name or m.group(0)

        return _TOKEN_RE.sub(sub, raw)

    def _emit_partial(self) -> None:
        if self._hold_sentences:
            return
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
            return
        self._emitted_view = view[:safe_end]
        self._send(view[done:safe_end])

    def _complete(self, sentence: str) -> None:
        """Judge a finished sentence; emit its un-emitted remainder, or nothing."""
        rendered = self._judge(sentence)
        emitted = self._emitted_view
        self._emitted_view = ""
        if rendered is None:
            self._dropped_any = True
            if emitted.strip():
                self.fragments += 1
            return
        if not rendered.startswith(emitted):
            raise _StreamStateError("rendered sentence does not extend the emitted lead-in")
        self._kept_any = True
        self._send(rendered[len(emitted) :])

    def _judge(self, sentence: str) -> str | None:
        """Same rules as ground_reply, applied to one sentence; None = drop it."""
        rendered = _render_sentence(sentence, self._ids(), self.flags)
        if rendered is None:
            return None
        extras = self._extras_list
        violations = check(rendered, self._items, user_text=self._user_text, extra_amounts=extras)
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
            bad = check(rendered, self._prev_scope, user_text=self._user_text, extra_amounts=extras)
        else:
            bad = [
                v
                for it in self._items
                for v in check(rendered, [it], user_text=self._user_text, extra_amounts=extras)
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
