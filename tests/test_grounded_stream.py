"""V5 amendment 1: StreamGrounder (token streaming with grounded rendering).

Acceptance rules (f)-(i) in docs/grounding-v5-acceptance.md. Hermetic and deterministic: no LLM,
no network, no randomness. Property-style loops split every corpus reply at EVERY position.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

import src.agents.grounded_stream as gs
from src.agents.claim_checker import check
from src.agents.grounded_render import ground_reply
from src.agents.grounded_stream import StreamGrounder
from src.agents.grounding import GROUNDED_FALLBACK_PREFIX
from tests import _grounding_v5_corpus as corpus

ITEMS = corpus.ITEMS


def run_stream(chunks: Iterator[str] | list[str], user: str = "") -> tuple[str, StreamGrounder]:
    """Feed chunks, finish, and return the emitted text plus the grounder."""
    g = StreamGrounder(ITEMS, user_text=user)
    for c in chunks:
        g.feed(c)
    g.finish()
    return g.text, g


def splits(text: str) -> Iterator[list[str]]:
    """Every two-chunk split of text, then the char-by-char split."""
    for i in range(1, len(text)):
        yield [text[:i], text[i:]]
    yield list(text)


# ------------------------------------------------------------------ (f) safety over all splits


@pytest.mark.parametrize("case", corpus.clean(), ids=lambda c: c["id"])
def test_clean_reply_streams_to_the_expected_text_at_every_split(case: dict[str, Any]) -> None:
    for chunks in splits(case["reply"]):
        out, g = run_stream(chunks, case["user_text"])
        assert out == case["expected"], chunks[:2]
        assert not g.replaced and g.fragments == 0


@pytest.mark.parametrize("case", corpus.adversarial(), ids=lambda c: c["id"])
def test_adversarial_reply_never_leaks_a_claim_at_any_split(case: dict[str, Any]) -> None:
    for chunks in splits(case["reply"]):
        out, _ = run_stream(chunks, case["user_text"])
        assert [b for b in case["bad_spans"] if b in out] == [], chunks[:2]
        assert check(out, ITEMS, user_text=case["user_text"]) == [], chunks[:2]
        assert all(k in out for k in case["must_keep"]), chunks[:2]
        assert "[item" not in out and "[price" not in out and "[size" not in out


@pytest.mark.parametrize(
    "case", [c for c in corpus.clean() if "[" in c["reply"]], ids=lambda c: c["id"]
)
def test_tag_split_across_chunks_renders_like_the_unsplit_stream(case: dict[str, Any]) -> None:
    whole, _ = run_stream([case["reply"]], case["user_text"])
    for chunks in splits(case["reply"]):
        out, _ = run_stream(chunks, case["user_text"])
        assert out == whole


def test_stream_matches_the_batch_validator_when_the_bad_sentence_starts_with_a_trigger() -> None:
    reply = "[item:1001] at [price:1001] is lovely. 30% off today. [item:1002] is a good pick."
    batch = ground_reply(reply, ITEMS).text
    out, g = run_stream(splits(reply).__next__())
    assert out.split() == batch.split() and g.fragments == 0


# ------------------------------------------------------------------ malformed and nested tags


@pytest.mark.parametrize(
    "reply",
    [
        "See [item:[item:1001]] now. Then [item:1002] is fine.",
        "See [[item:1001]] now. Then [item:1002] is fine.",
        "See [item:1001]] now. Then [item:1002] is fine.",
        "See [item:] now. Then [item:1002] is fine.",
        "See [item 1001] now. Then [item:1002] is fine.",
        "See [price:] now. Then [item:1002] is fine.",
        "See [item:1001][item:1002] now. Then [item:1002] is fine.",
    ],
)
def test_nested_and_malformed_tags_never_reach_the_user_raw(reply: str) -> None:
    for chunks in splits(reply):
        out, _ = run_stream(chunks)
        assert "item:" not in out.replace("[item 1001]", "") and "price:" not in out
        assert "Gloss Wrap Skirt is fine." in out


# ------------------------------------------------------------------ unclosed span at the end


def test_unclosed_span_at_the_end_is_dropped_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=gs.logger.name):
        out, g = run_stream(["Nice pick. ", "Love [item:1001", " and more"])
    assert "[item" not in out and "1001" not in out and "and more" not in out
    assert out.startswith("Nice pick.")
    events = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("{") and "grounded_unclosed_span" in r.getMessage()
    ]
    assert events == [{"event": "grounded_unclosed_span", "count": 1}]
    assert g.unclosed_spans == 1
    assert not any("Love" in r.getMessage() for r in caplog.records)  # never the reply text


def test_no_unclosed_event_for_a_clean_stream(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=gs.logger.name):
        run_stream(["Lovely [item:1001] today."])
    assert not [r for r in caplog.records if "grounded_unclosed_span" in r.getMessage()]


# ------------------------------------------------------------------ removed sentences


def test_sentence_starting_with_a_trigger_emits_nothing_of_itself() -> None:
    out, g = run_stream(["It was fine. ", "Free ", "shipping today for you. ", "Great."])
    assert "Free" not in out and "shipping" not in out and "today" not in out
    assert out.strip().startswith("It was fine.") and out.strip().endswith("Great.")
    assert g.fragments == 0


def test_claim_checker_hit_trims_the_cue_words_of_a_multiword_claim() -> None:
    # "last few" is a stock claim the keyword trigger does not know; the checker hit on the
    # sentence so far moves the hold point back to its first word ("last"), which is not emitted.
    out, g = run_stream(["Grab it, last few."])
    assert out.startswith("Grab it,") and "last" not in out and "few" not in out
    assert g.fragments == 1


def test_removed_sentence_keeps_only_a_claim_free_lead_in_and_counts_it() -> None:
    """Documented residual G2: the lead-in before the first trigger was already sent."""
    out, g = run_stream(["Nice. ", "It is in ", "stock now. ", "Bye."])
    assert "stock" not in out and "now" not in out
    assert g.fragments == 1
    assert check(out, ITEMS) == []


def test_everything_removed_appends_the_template() -> None:
    out, g = run_stream(["Half price today.", " Free shipping too."])
    assert out.startswith(GROUNDED_FALLBACK_PREFIX) and g.replaced
    assert "Half price" not in out and "shipping" not in out


def test_lead_in_fragment_is_separated_from_the_template() -> None:
    out, _ = run_stream(["Try [item:424242] today."])
    assert out.startswith("Try") and GROUNDED_FALLBACK_PREFIX in out and "424242" not in out
    assert "Try\n\n" in out


def test_leading_whitespace_is_dropped_when_the_first_sentence_is_removed() -> None:
    out, _ = run_stream(["30% off today. ", "Gloss Wrap Skirt is a fine pick."])
    assert out == "Gloss Wrap Skirt is a fine pick."


def test_unattributed_claim_uses_the_previous_named_item() -> None:
    out, g = run_stream(["[item:1001] suits a daytime event. ", "It is in stock."])
    assert out.startswith("Riviera Cotton Sundress suits a daytime event.")
    assert "stock" not in out and g.fragments == 1  # only the claim-free lead-in "It is" got out
    ok, _ = run_stream(["[item:1005] suits a daytime event. ", "It is in stock."])
    assert ok == "Linen Co-ord Set suits a daytime event. It is in stock."


# ------------------------------------------------------------------ fail closed


def test_validator_error_stops_emission_and_serves_the_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = StreamGrounder(ITEMS, fallback="TEMPLATE")
    g.feed("Fine sentence. Another one. ")
    before = g.text

    real_check = gs.check
    calls = {"n": 0}

    def flaky(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:  # fails once, then "recovers": a dead stream must stay dead anyway
            raise RuntimeError("checker exploded")
        return real_check(*a, **k)

    monkeypatch.setattr(gs, "check", flaky)
    assert g.feed("Third one is here. Fourth.") == [] and g.failed_closed
    assert g.feed("More text that must not pass. And more.") == []
    g.finish()
    assert g.text.startswith(before.rstrip()) and "Third" not in g.text and "more" not in g.text
    assert "validator_error" in g.flags


def test_validator_error_before_anything_kept_serves_the_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(gs, "check", boom)
    g = StreamGrounder(ITEMS, fallback="TEMPLATE")
    g.feed("First sentence here. Second one.")
    g.finish()
    assert g.text.strip().endswith("TEMPLATE") and g.failed_closed and g.replaced


def test_blank_stream_emits_nothing() -> None:
    out, g = run_stream(["", "  ", "\n"])
    assert out.strip() == "" and not g.replaced


# ------------------------------------------------------------------ pass-through behaviour


def test_plain_words_are_emitted_before_the_sentence_ends() -> None:
    g = StreamGrounder(ITEMS)
    pieces = g.feed("Lovely breezy sundress for ")
    assert "".join(pieces) == "Lovely breezy sundress for"  # no trigger; last complete word
    assert g.feed("the be") == [" the"]  # the word in progress is held
    assert g.feed("ach.") == []  # still no terminator seen
    assert g.finish() == [" beach."]


def test_item_span_renders_without_holding_the_sentence() -> None:
    g = StreamGrounder(ITEMS)
    assert "".join(g.feed("Love the [item:1001] ")) == "Love the Riviera Cotton Sundress"
    assert "".join(g.feed("for brunch, ")) == " for brunch,"  # still streaming after the span


def test_price_span_holds_the_rest_of_the_sentence_until_it_ends() -> None:
    g = StreamGrounder(ITEMS)
    assert "".join(g.feed("Love the [item:1001] at [price:1001] for brunch. ")) == (
        "Love the Riviera Cotton Sundress at"
    )
    # "Next" starts a new sentence, so the first one is judged and its remainder released
    assert "".join(g.feed("Next")) == " ₹2,999 for brunch."
    assert "".join(g.finish()) == " Next"
    assert g.text == "Love the Riviera Cotton Sundress at ₹2,999 for brunch. Next"


# ------------------------------------------------------------------ hold_sentences mode (no fragments)


def run_held(chunks: list[str], user: str = "") -> tuple[str, StreamGrounder]:
    g = StreamGrounder(ITEMS, user_text=user, hold_sentences=True)
    for c in chunks:
        g.feed(c)
    g.finish()
    return g.text, g


@pytest.mark.parametrize("case", corpus.adversarial() + corpus.clean(), ids=lambda c: c["id"])
def test_hold_mode_equals_the_batch_validator_and_never_leaves_a_fragment(
    case: dict[str, Any],
) -> None:
    batch = ground_reply(case["reply"], ITEMS, user_text=case["user_text"]).text
    for chunks in splits(case["reply"]):
        out, g = run_held(chunks, case["user_text"])
        assert out.split() == batch.split(), chunks[:2]
        assert g.fragments == 0


def test_hold_mode_emits_whole_sentences_only() -> None:
    g = StreamGrounder(ITEMS, hold_sentences=True)
    assert g.feed("Lovely breezy sundress for the be") == []
    assert g.feed("ach. Next one comes") == ["Lovely breezy sundress for the beach."]
    assert g.finish() == [" Next one comes"]


# ------------------------------------------------------------------ amendment 3


def _boom_once(monkeypatch: pytest.MonkeyPatch, nth: int = 1) -> None:
    real_check = gs.check
    calls = {"n": 0}

    def flaky(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == nth:  # fails once, then "recovers": a dead stream must stay dead anyway
            raise RuntimeError("checker exploded")
        return real_check(*a, **k)

    monkeypatch.setattr(gs, "check", flaky)


def test_error_after_a_kept_sentence_appends_the_template_and_pins_the_full_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = StreamGrounder(ITEMS, fallback="TEMPLATE")
    seen = g.feed("Fine sentence. Another bit here. ")
    _boom_once(monkeypatch)
    seen += g.feed("Third bit is here. And")
    seen += g.feed("More text that must not pass.")
    seen += g.finish()
    assert g.text == "Fine sentence. Another bit here.\n\nTEMPLATE"
    assert "".join(seen) == g.text  # what the caller was handed is exactly what is persisted
    assert g.failed_closed and g.replaced and "template_after_error" in g.flags


def test_a_dead_stream_emits_nothing_more_even_when_the_checker_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The error hits a trigger-first sentence (no lead-in was sent, so there is no stale state
    # that would make later sentences fail on their own): only the dead-stream guard stops them.
    g = StreamGrounder(ITEMS, fallback="TEMPLATE")
    _boom_once(monkeypatch, nth=2)
    seen = g.feed("Fine sentence. Free shipping now. Third bit is here. Fourth.")
    assert g.failed_closed
    seen += g.feed(" Fifth bit is here. Sixth bit.")
    seen += g.feed(" Seventh bit.")
    seen += g.finish()
    assert g.text == "Fine sentence.\n\nTEMPLATE"
    assert "".join(seen) == g.text


def test_error_with_a_dangling_lead_in_still_ends_in_the_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = StreamGrounder(ITEMS, fallback="TEMPLATE")
    seen = g.feed("Fine sentence. And then the")
    _boom_once(monkeypatch)
    seen += g.feed(" skirt. Next")  # sentence 2 is judged here and the checker raises
    seen += g.finish()
    assert g.text.endswith("\n\nTEMPLATE") and g.text.startswith("Fine sentence.")
    assert "Next" not in g.text and "".join(seen) == g.text


def test_constructor_does_no_work_that_can_raise() -> None:
    class Hostile:
        def __iter__(self) -> Iterator[Any]:
            raise RuntimeError("must not be touched by the constructor")

    g = StreamGrounder(Hostile(), extra_amounts=Hostile())  # type: ignore[arg-type]
    assert g.text == ""
    g.feed("Hello there. More text.")  # the failure shows up inside the guarded path
    g.finish()
    assert g.failed_closed and g.text.strip() != ""


# hard cap on the sentence in progress

CAP = gs.MAX_OPEN_SENTENCE_CHARS


def _drip(text: str, size: int = 7) -> StreamGrounder:
    g = StreamGrounder(ITEMS, fallback="TEMPLATE")
    for i in range(0, len(text), size):
        g.feed(text[i : i + size])
    g.finish()
    return g


def test_four_thousand_characters_without_a_boundary_finish_quickly() -> None:
    import time

    text = "word " * 800  # 4000 characters, no sentence boundary
    t0 = time.perf_counter()
    g = _drip(text)
    assert time.perf_counter() - t0 < 1.0  # unbounded rescans took about 3.7 s here
    assert "overlong_sentence" in g.flags
    assert len(g.text) <= CAP + len("\n\nTEMPLATE")


def test_one_giant_chunk_is_capped_too() -> None:
    import time

    t0 = time.perf_counter()
    g = StreamGrounder(ITEMS, fallback="TEMPLATE")
    g.feed("word " * 800)
    g.finish()
    assert time.perf_counter() - t0 < 1.0 and "overlong_sentence" in g.flags


def test_claim_inside_the_first_cap_characters_is_still_removed_and_the_tail_is_discarded() -> None:
    text = "word " * 60 + "it is 40% off today " + "word " * 300 + "ENDMARK"
    g = _drip(text)
    assert "40%" not in g.text and "ENDMARK" not in g.text
    assert g.text.endswith("\n\nTEMPLATE") and g.replaced
    assert "overlong_sentence" in g.flags


def test_a_clean_overlong_sentence_is_kept_up_to_the_cut_and_the_rest_discarded() -> None:
    text = "Nice. " + "calm " * 400 + "ENDMARK"
    g = _drip(text)
    assert g.text.startswith("Nice. calm calm") and "ENDMARK" not in g.text
    assert "TEMPLATE" not in g.text  # something was kept, so no template
    assert len(g.text) <= CAP + 10 and "overlong_sentence" in g.flags


def test_a_sentence_just_under_the_cap_streams_normally() -> None:
    text = ("calm " * 150).strip() + "."  # 749 characters
    assert len(text) < CAP
    g = _drip(text)
    assert g.text == text and "overlong_sentence" not in g.flags


# trigger vocabulary (amendment 3): each group holds the sentence from the trigger


@pytest.mark.parametrize(
    ("chunk", "emitted"),
    [
        ("It has 3 pockets and more ", "It has"),  # digit
        ("Nice dress at ₹ ", "Nice dress at"),  # currency symbol
        ("Nice dress at $ ", "Nice dress at"),
        ("The skirt is two thousand ", "The skirt is"),  # number word
        ("The skirt is a hundred ", "The skirt is a"),
        ("Quick note: ½ ", "Quick note:"),  # fraction character
        ("Quick note: Buy ", "Quick note:"),  # buy
        ("Sixty per cent ", ""),  # the number word is the trigger itself
        ("It is 20 percent ", "It is"),
        ("A quarter ", "A"),  # quarter
        ("It is a third ", "It is a"),
        ("Lovely breezy sundress for ", "Lovely breezy sundress for"),  # no trigger: streams
    ],
)
def test_trigger_vocabulary_holds_a_sentence_from_the_trigger(chunk: str, emitted: str) -> None:
    g = StreamGrounder(ITEMS)
    assert "".join(g.feed(chunk)) == emitted


@pytest.mark.parametrize(
    ("chunks", "must_not_leak"),
    [
        (["The Gloss Wrap Skirt is two thousand ", "bucks today."], ["two", "thousand"]),
        (["Quick note: ", "Buy one get one free."], ["Buy", "get one"]),
        (["Quick note: ", "½ price today."], ["½"]),
        (["The skirt is ", "sixty per cent off."], ["sixty", "per"]),
        (["Only ", "two left of the skirt."], ["two left"]),
    ],
)
def test_word_number_claims_leave_no_fragment_of_the_claim(
    chunks: list[str], must_not_leak: list[str]
) -> None:
    out, _ = run_stream(chunks)
    assert all(s not in out for s in must_not_leak), out
