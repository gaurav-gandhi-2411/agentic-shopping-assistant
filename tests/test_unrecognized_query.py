"""Regression tests for the gibberish->clarify guard (2026-07-10 sweep, P0-4):
a keyboard-mash first message got a confident product recommendation instead of
a clarifying question."""
from __future__ import annotations

from src.agents.graph import _is_unrecognized_query
from src.retrieval.sparse_search import SparseRetriever


class _StubSparse:
    """Vocabulary stub mirroring SparseRetriever.has_any_known_token."""

    def __init__(self, vocab: set[str]) -> None:
        self._vocab = vocab

    def has_any_known_token(self, query: str) -> bool:
        return any(t in self._vocab for t in query.lower().split())


class _StubRetriever:
    def __init__(self, vocab: set[str]) -> None:
        self.sparse = _StubSparse(vocab)


_VOCAB = {
    "lehenga", "saree", "kurta", "sangeet", "wedding", "black", "dress", "cotton",
    "purple", "shoes",
}


class TestUnrecognizedQueryGuard:
    def test_keyboard_mash_is_flagged(self) -> None:
        assert _is_unrecognized_query("asdfgh qwerty zxcvb", _StubRetriever(_VOCAB))

    def test_mixed_gibberish_majority_flagged(self) -> None:
        # Live P0 repro (2026-07-12): "asdkfjhqwoiuerlkj purple flying shoes" — 2 of
        # 4 tokens ("purple", "shoes") are real catalogue vocabulary, so the OLD
        # any-token-recognized check judged this "recognized" and the LLM wrote a
        # confident recommendation pitch with zero acknowledgment that most of the
        # query was gibberish. Must be flagged for clarification instead.
        assert _is_unrecognized_query(
            "asdkfjhqwoiuerlkj purple flying shoes", _StubRetriever(_VOCAB)
        )

    def test_mixed_gibberish_second_repro_flagged(self) -> None:
        # Second independent live repro of the same defect: "qwxyz" here is the
        # obvious keyboard-mash token; "purple"/"shoes" are real catalogue terms.
        assert _is_unrecognized_query(
            "purple flying unicorn shoes qwxyz", _StubRetriever(_VOCAB)
        )

    def test_real_shopping_query_passes(self) -> None:
        assert not _is_unrecognized_query(
            "lehenga for my sister's sangeet", _StubRetriever(_VOCAB)
        )

    def test_intent_words_pass_without_catalogue_tokens(self) -> None:
        # "show me something nice" has zero catalogue tokens but is a real request.
        assert not _is_unrecognized_query("show me something nice", _StubRetriever(_VOCAB))

    def test_refinement_word_cheaper_passes(self) -> None:
        assert not _is_unrecognized_query("cheaper", _StubRetriever(_VOCAB))

    def test_short_or_empty_input_not_flagged(self) -> None:
        assert not _is_unrecognized_query("", _StubRetriever(_VOCAB))
        assert not _is_unrecognized_query("ok", _StubRetriever(_VOCAB))

    def test_missing_sparse_index_never_blocks(self) -> None:
        class _NoSparse:
            sparse = None

        assert not _is_unrecognized_query("asdfgh qwerty", _NoSparse())


class TestSparseKnownTokens:
    def test_bm25_vocabulary_membership(self) -> None:
        import pandas as pd

        retr = SparseRetriever(config={})
        df = pd.DataFrame(
            {
                "article_id": ["A1", "A2"],
                "search_text": [
                    "red silk saree wedding festive",
                    "blue cotton kurta casual summer",
                ],
            }
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            retr.build_index(df, Path(d))
        assert retr.has_any_known_token("a saree please")
        assert not retr.has_any_known_token("asdfgh qwerty zxcvb")

    def test_no_index_returns_true(self) -> None:
        retr = SparseRetriever(config={})
        assert retr.has_any_known_token("anything at all")
