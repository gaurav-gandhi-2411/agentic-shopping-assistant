"""Occasion-merchandise leak into apparel results — live-proven bug (2026-07-23).

Live-proven bug: "what should I wear for raksha bandhan" (occasion keyword,
NO garment noun, "wear" apparel intent) returned 3 of 5 items as literal
Rakhi threads/gift objects (product_type_name="Rakhi", e.g. "Ram Mandir
Blessings Rakhi", "Divine Rudraksha Thread Rakhi For Brother"), and the LLM's
own reply celebrated them as gifts. Queries WITH a garment noun ("kurti for
raksha bandhan") and the outfit-composer path ("raksha bandhan outfit for
sister") were both already clean — the leak is specifically in the plain
search path when garment_type=None.

Root cause: BM25/dense retrieval on occasion text (e.g. "raksha bandhan")
naturally ranks the catalogue's literal "Rakhi"/"Gift Hamper"/"Idols" rows
highly since they genuinely match the occasion keyword lexically — there was
no exclusion mechanism distinguishing apparel from occasion merchandise
(unlike kids items / fabric bolts, see tests/test_kids_leak_exclusion.py,
whose mechanism shape this file mirrors).

Fixes verified here:
  A. is_occasion_merchandise_type (src.catalogue.cleaning) — a hard,
     data-grounded product-type exclusion set (Rakhi/Rakhi Hamper/Rakhi Gift
     Hamper/Silver Rakhi/Gift Hamper/Idols). Jewellery and Potli bags are
     deliberately NOT excluded (apparel-adjacent, legitimately styled).
  B. _apply_occasion_merchandise_gate (src.agents.graph) — applies A only
     when an occasion is set, garment_type is None, AND the query is not an
     explicit merchandise request ("rakhi for my brother", "gift for
     raksha bandhan" must still surface rakhis/hampers).
  C. End-to-end: the plain search path (search_node) never surfaces
     occasion-merchandise for a bare apparel-intent occasion query, against
     the real unified catalogue.
  D. 2026-07-23 FOLLOW-UP (live-proof on revision asa-stylist-api-00084-7t4):
     is_occasion_merchandise_name (src.catalogue.cleaning) — a NAME-level
     complement to A, catching merchandise a store tagged with a GENERIC
     catalog bucket ("Fashion"/"Others"/"Article"/...) instead of a
     dedicated "Rakhi"/"Idols" type. Live-proven residual leak: "White And
     Pink Beautiful Floral Designer Bhaiya Bhabhi Rakhi Set" (store=ishhaara,
     product_type_name="Fashion") ranked #1 of only 2 results for "what
     should I wear for raksha bandhan" — A's type-only check missed it. A
     genuine apparel item whose name ALSO mentions rakhi/gift ("Men's Yellow
     Lehariya Cotton Kurta Rakhi Gift Box for Brother", typed "kurta") is
     never excluded — its type is real apparel, not a generic bucket.
  E. 2026-07-24 CONCEPT-BROADENING (hand-labeled in strict_gold_labels.yaml
     Batch 12, query occ_adv_002): "bright haldi look for women" surfaced
     "Ellaichi Brooch" (store=ishhaara, product_type_name="Fashion") — its
     own detail_desc frames it as a "Haldi & Mehendi Favours" guest return-
     gift, not apparel. Unlike D's residual leak, this item's NAME carries NO
     merchandise marker at all — only the shared collection description
     does, so is_occasion_merchandise_name now also checks detail_desc. A
     full catalogue audit (see cleaning.py's _OCCASION_MERCHANDISE_NAME_RE
     2026-07-24 comment) found genuine, non-redundant support for exactly one
     new term family ("favour(s)"/"favor(s)") and confirmed the same 37-row
     ishhaara collection is a real leak risk under haldi (37/37 rows),
     mehendi (36/37), and wedding_guest (37/37) — all three get regression
     coverage below.
"""
from __future__ import annotations

import pytest

from src.agents.graph import _OCCASION_MERCHANDISE_REQUEST_RE, _apply_occasion_merchandise_gate
from src.catalogue.cleaning import is_occasion_merchandise_name, is_occasion_merchandise_type

# ---------------------------------------------------------------------------
# A. is_occasion_merchandise_type
# ---------------------------------------------------------------------------


class TestIsOccasionMerchandiseType:
    def test_rakhi_flagged(self) -> None:
        assert is_occasion_merchandise_type("Rakhi") is True

    def test_rakhi_case_insensitive(self) -> None:
        assert is_occasion_merchandise_type("rakhi") is True
        assert is_occasion_merchandise_type("RAKHI") is True

    def test_silver_rakhi_flagged(self) -> None:
        assert is_occasion_merchandise_type("Silver Rakhi") is True

    def test_rakhi_hamper_variants_flagged(self) -> None:
        assert is_occasion_merchandise_type("Rakhi Hamper") is True
        assert is_occasion_merchandise_type("Rakhi Gift Hamper") is True

    def test_gift_hamper_flagged(self) -> None:
        assert is_occasion_merchandise_type("Gift Hamper") is True

    def test_idols_flagged(self) -> None:
        assert is_occasion_merchandise_type("Idols") is True

    def test_hampers_flagged(self) -> None:
        """2026-07-30 addition: bare "Hampers" (plural, the actual catalogue
        facet value) was missing — only compound "gift hamper" phrases were
        covered. Confirmed via catalogue audit: product_type_name=="Hampers"
        is exactly 3 rows, all genuine "... Bridal Hamper Box" gift sets."""
        assert is_occasion_merchandise_type("Hampers") is True
        assert is_occasion_merchandise_type("hamper") is True

    def test_gift_card_variants_flagged(self) -> None:
        """2026-07-30 addition: live-proven via "anniversary party outfit for
        women" ranking "Anniversary Day E-Gift Card" #1 of 5 -- catalogue
        audit confirmed 21 rows, 100% genuine gift cards, zero false-positive
        risk. Covers all three real facet-value spellings/cases."""
        assert is_occasion_merchandise_type("gift card") is True
        assert is_occasion_merchandise_type("Gift Cards") is True
        assert is_occasion_merchandise_type("Gift-Card") is True

    def test_apparel_types_not_flagged(self) -> None:
        for pt in ("kurta", "kurti", "lehenga", "saree", "sherwani", "dupatta"):
            assert is_occasion_merchandise_type(pt) is False

    def test_jewellery_not_flagged(self) -> None:
        """Jewellery is apparel-adjacent and must never be excluded — see the
        multi-family accessory retrieval fix (commit 1717265)."""
        for pt in ("Necklace", "Earrings", "jhumka", "Bangles", "jewellery", "Ring"):
            assert is_occasion_merchandise_type(pt) is False

    def test_potli_not_flagged(self) -> None:
        """Potli bags appear in the diwali-keyword-matching set but are
        legitimately styled as accessories in real looks."""
        assert is_occasion_merchandise_type("Potli") is False
        assert is_occasion_merchandise_type("Potlis") is False

    def test_none_and_empty_safe(self) -> None:
        assert is_occasion_merchandise_type(None) is False
        assert is_occasion_merchandise_type("") is False


# ---------------------------------------------------------------------------
# B. _apply_occasion_merchandise_gate
# ---------------------------------------------------------------------------

_RAKHI_ITEM = {"article_id": "r1", "product_type": "Rakhi", "prod_name": "Ram Mandir Blessings Rakhi"}
_HAMPER_ITEM = {"article_id": "h1", "product_type": "Gift Hamper", "prod_name": "Voylla Diwali Hamper"}
_KURTA_ITEM = {"article_id": "k1", "product_type": "kurta", "prod_name": "Blue Cotton Kurta"}
_NECKLACE_ITEM = {"article_id": "n1", "product_type": "Necklace", "prod_name": "Gold Plated Necklace"}


class TestApplyOccasionMerchandiseGate:
    def test_bare_wear_query_strips_rakhi(self) -> None:
        """The exact live-reproduced query shape: apparel intent ('wear'), no
        garment noun."""
        items = [_RAKHI_ITEM, _KURTA_ITEM, _NECKLACE_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "raksha_bandhan", None, "what should I wear for raksha bandhan"
        )
        assert _RAKHI_ITEM not in out
        assert _KURTA_ITEM in out
        assert _NECKLACE_ITEM in out  # jewellery survives

    def test_bare_occasion_query_no_wear_word_still_strips(self) -> None:
        """A bare occasion query with no garment noun must return apparel
        even without the literal word 'wear' — see the gate's docstring."""
        items = [_RAKHI_ITEM, _KURTA_ITEM]
        out = _apply_occasion_merchandise_gate(items, "raksha_bandhan", None, "raksha bandhan")
        assert _RAKHI_ITEM not in out
        assert _KURTA_ITEM in out

    def test_diwali_gift_hamper_strips_hamper(self) -> None:
        items = [_HAMPER_ITEM, _KURTA_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "diwali", None, "what should I wear for diwali"
        )
        assert _HAMPER_ITEM not in out
        assert _KURTA_ITEM in out

    def test_explicit_rakhi_request_bypasses_gate(self) -> None:
        """'rakhi for my brother' is an explicit merchandise ask — suppressing
        it would be a NEW bug."""
        items = [_RAKHI_ITEM, _KURTA_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "raksha_bandhan", None, "rakhi for my brother"
        )
        assert out == items

    def test_explicit_gift_request_bypasses_gate(self) -> None:
        items = [_HAMPER_ITEM, _KURTA_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "diwali", None, "gift for raksha bandhan"
        )
        assert out == items

    def test_buy_rakhi_bypasses_gate(self) -> None:
        items = [_RAKHI_ITEM]
        out = _apply_occasion_merchandise_gate(items, "raksha_bandhan", None, "buy rakhi")
        assert out == items

    def test_garment_type_present_is_noop(self) -> None:
        """'kurti for raksha bandhan' already hard-filters retrieval to
        product_type_name=kurti — re-checking here is a deliberate no-op."""
        items = [_RAKHI_ITEM, _KURTA_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "raksha_bandhan", "kurti", "kurti for raksha bandhan"
        )
        assert out == items

    def test_no_occasion_is_noop(self) -> None:
        items = [_RAKHI_ITEM, _KURTA_ITEM]
        out = _apply_occasion_merchandise_gate(items, None, None, "anything")
        assert out == items

    def test_pool_not_underflow_protected(self) -> None:
        """Mirrors _apply_loungewear_gate's discipline: occasion merchandise
        is never an acceptable apparel substitute, even as a last resort —
        this MAY legitimately empty the result list."""
        items = [_RAKHI_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "raksha_bandhan", None, "what should I wear for raksha bandhan"
        )
        assert out == []

    def test_generic_typed_rakhi_set_stripped(self) -> None:
        """The exact live-proof residual leak (revision
        asa-stylist-api-00084-7t4): a Fashion-typed rakhi set that
        is_occasion_merchandise_type's type-only check alone would miss."""
        items = [_GENERIC_RAKHI_SET_ITEM, _KURTA_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "raksha_bandhan", None, "what should I wear for raksha bandhan"
        )
        assert _GENERIC_RAKHI_SET_ITEM not in out
        assert _KURTA_ITEM in out

    def test_generic_typed_rakhi_gift_box_kurta_never_stripped(self) -> None:
        """A REAL kurta bundled with a rakhi ("... Kurta Rakhi Gift Box for
        Brother", typed "kurta") must survive — the name mentions rakhi/gift
        but the type IS real apparel, not a generic bucket."""
        items = [_KURTA_RAKHI_GIFT_BOX_ITEM]
        out = _apply_occasion_merchandise_gate(
            items, "raksha_bandhan", None, "what should I wear for raksha bandhan"
        )
        assert out == items


# ---------------------------------------------------------------------------
# D. is_occasion_merchandise_name — the generic-type NAME-level complement
# ---------------------------------------------------------------------------

_GENERIC_RAKHI_SET_ITEM = {
    "article_id": "g1", "product_type": "Fashion",
    "prod_name": "White And Pink Beautiful Floral Designer Bhaiya Bhabhi Rakhi Set",
}
_KURTA_RAKHI_GIFT_BOX_ITEM = {
    "article_id": "kg1", "product_type": "kurta",
    "prod_name": "Men's Yellow Lehariya Cotton Kurta Rakhi Gift Box for Brother",
}


class TestIsOccasionMerchandiseName:
    def test_generic_typed_rakhi_set_flagged(self) -> None:
        """The exact live-proof residual item."""
        assert is_occasion_merchandise_name(
            "White And Pink Beautiful Floral Designer Bhaiya Bhabhi Rakhi Set", "Fashion"
        ) is True

    def test_generic_typed_bare_occasion_phrase_flagged(self) -> None:
        """No literal "rakhi" word, but the bare occasion phrase alone under
        a generic type is still merchandise (the 1 residual raksha_bandhan
        row this pattern was audited against)."""
        assert is_occasion_merchandise_name("Raksha Bandhan Gift For Brother", "Fashion") is True

    def test_generic_typed_idol_flagged(self) -> None:
        assert is_occasion_merchandise_name(
            "Voylla 92.5 Sterling Silver Kamdhenu Sacred Cow Idol", "Article"
        ) is True

    def test_generic_typed_tealight_holder_flagged(self) -> None:
        assert is_occasion_merchandise_name("Festive Decorative Tealight Holder", "Others") is True

    def test_generic_typed_showpiece_flagged(self) -> None:
        assert is_occasion_merchandise_name("Divine Bond Festive Showpiece", "Others") is True

    def test_generic_typed_gift_card_flagged(self) -> None:
        """2026-07-30 addition: the exact live-proven leak -- "Anniversary
        Day E-Gift Card" is typed "Fashion" (a generic bucket), so only the
        NAME-level check catches it."""
        assert is_occasion_merchandise_name("Anniversary Day E-Gift Card", "Fashion") is True

    def test_apparel_typed_anniversary_lehenga_never_flagged(self) -> None:
        """A genuine anniversary-occasion apparel item must never be flagged
        -- this predicate only ever fires on gift-card/hamper/rakhi/idol
        vocabulary, never on the bare occasion word."""
        assert is_occasion_merchandise_name(
            "Anniversary Party Wear Lehenga", "lehenga"
        ) is False

    def test_kurta_typed_rakhi_gift_box_never_flagged(self) -> None:
        """The critical negative case: a REAL kurta whose name mentions
        rakhi/gift must NEVER be excluded — its type IS real apparel."""
        assert is_occasion_merchandise_name(
            "Men's Yellow Lehariya Cotton Kurta Rakhi Gift Box for Brother", "kurta"
        ) is False

    def test_nightwear_typed_rakhi_gift_box_never_flagged(self) -> None:
        assert is_occasion_merchandise_name(
            "Men's Red Bandhni Rayon Kurta & Pyjama Rakhi Gift Box for Brother", "nightwear"
        ) is False

    def test_bracelets_typed_rakhi_bracelet_never_flagged(self) -> None:
        """Jewellery carve-out: "Bracelets" is a real accessory type, not a
        generic bucket — a literal "Rakhi Bracelet" stays included."""
        assert is_occasion_merchandise_name(
            "Multicolor Gold-Plated Jadau Kundan Rakhi Bracelet", "Bracelets"
        ) is False

    def test_tie_set_typed_gift_set_never_flagged(self) -> None:
        """"gift" alone under a real accessory type ("Tie Set") is not
        merchandise — only rakhi/idol/hamper/showpiece/tealight/raksha-
        bandhan words trigger this predicate, and only under a generic type."""
        assert is_occasion_merchandise_name(
            "Classic Dot Pattern Tie and Cufflink Gift Set", "Tie Set"
        ) is False

    def test_generic_type_with_no_merchandise_word_not_flagged(self) -> None:
        assert is_occasion_merchandise_name("Chand Shaped Pachi Kundan Studded Earchains", "Fashion") is False

    def test_none_prod_name_safe(self) -> None:
        assert is_occasion_merchandise_name(None, "Fashion") is False

    def test_none_product_type_treated_as_generic(self) -> None:
        """A missing product_type_name (None) is itself a "no apparel
        signal" case — treated as generic, per _GENERIC_PRODUCT_TYPES
        including the empty string."""
        assert is_occasion_merchandise_name("Rakhi Combo", None) is True

    def test_none_product_type_with_no_merchandise_word_safe(self) -> None:
        assert is_occasion_merchandise_name("Blue Cotton Kurta", None) is False

    def test_empty_strings_safe(self) -> None:
        assert is_occasion_merchandise_name("", "") is False


# ---------------------------------------------------------------------------
# E. is_occasion_merchandise_name — 2026-07-24 favour/favor + detail_desc
# ---------------------------------------------------------------------------

_ELLAICHI_BROOCH_DESC = (
    "Are you looking for the perfect way to thank your guests with a touch "
    "of tradition? Welcome to Ishhaara's Haldi & Mehendi Favours, where "
    "every token is a blend of love, culture, and celebration!"
)


class TestIsOccasionMerchandiseNameFavour:
    def test_desc_only_favour_flagged(self) -> None:
        """The exact live-proof item: NO merchandise word in the name
        itself, only in the shared collection description."""
        assert is_occasion_merchandise_name(
            "Ellaichi Brooch", "Fashion", _ELLAICHI_BROOCH_DESC
        ) is True

    def test_name_only_favour_flagged(self) -> None:
        assert is_occasion_merchandise_name(
            "Special Hamper For Wedding Favours", "Fashion", None
        ) is True

    def test_favor_american_spelling_flagged(self) -> None:
        assert is_occasion_merchandise_name(
            "Party Favor Box", "Fashion", "A cute little wedding favor for guests."
        ) is True

    def test_favourite_not_a_false_positive(self) -> None:
        """"Favourite" (a common real-apparel marketing word) must never
        trip the word-bounded favour/favor pattern."""
        assert is_occasion_merchandise_name(
            "Pink Cotton Kurta Set", "Fashion",
            "This kurta set is sure to become your favourite in no time.",
        ) is False

    def test_desc_favour_under_real_apparel_type_never_flagged(self) -> None:
        """A genuine kurta whose desc happens to mention "favourite"/
        "favors" stays included — its type IS real apparel, not generic."""
        assert is_occasion_merchandise_name(
            "Blue Solid Cotton Kurta", "kurta",
            "Present as a thoughtful gift or use as a bridal favour.",
        ) is False

    def test_desc_favour_potli_bag_type_never_flagged(self) -> None:
        """Real dual-use accessories (tjori's silk Potli bags, genuinely
        marketed as wearable AND a bridal favour) stay included — "Potlis"
        is a real accessory type, not a generic bucket."""
        assert is_occasion_merchandise_name(
            "Sapphire Blue Embroidered Silk Potli", "Potlis",
            "Pair with a sapphire blue lehenga for a regal look. Present as "
            "a thoughtful gift or use as a bridal favour for weddings.",
        ) is False

    def test_none_detail_desc_safe(self) -> None:
        assert is_occasion_merchandise_name("Blue Cotton Kurta", "Fashion", None) is False


class TestOccasionMerchandiseRequestReFavour:
    def test_haldi_favours_request_matches(self) -> None:
        assert _OCCASION_MERCHANDISE_REQUEST_RE.search("haldi favours for guests".lower())

    def test_wedding_favors_request_matches(self) -> None:
        assert _OCCASION_MERCHANDISE_REQUEST_RE.search("wedding favors for mehendi".lower())

    def test_favourite_does_not_match(self) -> None:
        """The bypass regex must not fire on ordinary "favourite" copy."""
        assert not _OCCASION_MERCHANDISE_REQUEST_RE.search(
            "this kurta is my favourite for haldi".lower()
        )


# ---------------------------------------------------------------------------
# C. End-to-end: search_node never surfaces occasion merchandise (real index)
# ---------------------------------------------------------------------------


class TestSearchNodeOccasionMerchandiseRealIndex:
    """Reproduces the live bug end-to-end against the real unified catalogue."""

    @staticmethod
    def _run_search(query: str):
        import pandas as pd

        from src.agents.graph import build_graph
        from src.memory.conversation import ConversationMemory
        from src.retrieval.dense_search import DenseRetriever
        from src.retrieval.hybrid_search import HybridRetriever
        from src.retrieval.sparse_search import SparseRetriever

        unified_dir = "data/processed/unified"
        config: dict = {
            "agent": {"max_iterations": 3},
            "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
            "retrieval": {
                "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dense_dim": 384,
                "rrf_k": 60,
                "top_k": 50,
                "final_k": 10,
                "store_diversity": 0.2,
            },
        }
        dense = DenseRetriever.load(config, unified_dir)
        sparse = SparseRetriever.load(config, unified_dir)
        catalogue_df = pd.read_parquet(f"{unified_dir}/catalogue.parquet")
        retriever = HybridRetriever(dense, sparse, catalogue_df, config)

        llm = _RealIndexMockLLM(["Here you go."] * 5)
        memory = ConversationMemory(llm, config)
        agent = build_graph(retriever, catalogue_df, llm, config, streaming_mode=True)

        state = {
            "messages": [{"role": "user", "content": query}],
            "user_query": query,
            "current_plan": None,
            "tool_calls": [],
            "retrieved_items": [],
            "filters": {},
            "final_answer": None,
            "iteration": 0,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
            "anchor_article_id": None,
            "outfit_rationale": None,
            "outfit_variants": None,
            "_memory": memory,
        }
        return agent.invoke(state)

    @pytest.mark.requires_index
    def test_what_should_i_wear_for_raksha_bandhan_no_rakhi_leak(self) -> None:
        """The exact live-reproduced query."""
        result = self._run_search("what should I wear for raksha bandhan")
        items = result.get("retrieved_items", [])
        assert items, "expected items for 'what should I wear for raksha bandhan'"
        leaked = [it for it in items if (it.get("product_type") or "").lower() == "rakhi"]
        assert not leaked, f"Rakhi item(s) leaked: {[it.get('prod_name') for it in leaked]}"

    @pytest.mark.requires_index
    def test_rakhi_for_my_brother_still_returns_rakhis(self) -> None:
        """Explicit merchandise request must still surface rakhis — the
        inverse-direction regression this fix must not introduce."""
        result = self._run_search("rakhi for my brother")
        items = result.get("retrieved_items", [])
        assert items, "expected items for 'rakhi for my brother'"
        rakhis = [it for it in items if (it.get("product_type") or "").lower() == "rakhi"]
        assert rakhis, "expected at least one Rakhi item for an explicit rakhi request"

    # -- 2026-07-24 favour/favor concept-broadening regressions -------------

    _ELLAICHI_BROOCH_ARTICLE_ID = "7332515610667"

    @pytest.mark.requires_index
    def test_bright_haldi_look_no_ellaichi_brooch_leak(self) -> None:
        """The exact hand-labeled live-proof bug (strict_gold_labels.yaml
        occ_adv_002): "bright haldi look for women" must never surface the
        "Ellaichi Brooch" wedding-favour item, or any other item this
        classifier considers occasion merchandise."""
        result = self._run_search("bright haldi look for women")
        items = result.get("retrieved_items", [])
        assert items, "expected items for 'bright haldi look for women'"
        leaked_ids = [it for it in items if str(it.get("article_id")) == self._ELLAICHI_BROOCH_ARTICLE_ID]
        assert not leaked_ids, "Ellaichi Brooch (7332515610667) leaked into 'bright haldi look' results"
        merch = [
            it for it in items
            if is_occasion_merchandise_type(it.get("product_type"))
            or is_occasion_merchandise_name(
                it.get("prod_name") or it.get("display_name"),
                it.get("product_type"),
                it.get("detail_desc"),
            )
        ]
        assert not merch, f"occasion merchandise leaked: {[it.get('prod_name') for it in merch]}"

    @pytest.mark.requires_index
    def test_mehendi_look_no_favour_collection_leak(self) -> None:
        """Audit-confirmed genuine risk: 36/37 rows in the ishhaara "Haldi &
        Mehendi Favours" collection also match the mehendi keyword."""
        result = self._run_search("mehendi look for women")
        items = result.get("retrieved_items", [])
        assert items, "expected items for 'mehendi look for women'"
        merch = [
            it for it in items
            if is_occasion_merchandise_type(it.get("product_type"))
            or is_occasion_merchandise_name(
                it.get("prod_name") or it.get("display_name"),
                it.get("product_type"),
                it.get("detail_desc"),
            )
        ]
        assert not merch, f"occasion merchandise leaked: {[it.get('prod_name') for it in merch]}"

    @pytest.mark.requires_index
    def test_wedding_guest_look_no_favour_collection_leak(self) -> None:
        """Audit-confirmed genuine risk: 37/37 rows in the ishhaara favours
        collection also mention "wedding" (e.g. "Special Hamper For Wedding
        Favours"), so a wedding_guest-occasion query is equally exposed."""
        result = self._run_search("wedding guest look for women")
        items = result.get("retrieved_items", [])
        assert items, "expected items for 'wedding guest look for women'"
        merch = [
            it for it in items
            if is_occasion_merchandise_type(it.get("product_type"))
            or is_occasion_merchandise_name(
                it.get("prod_name") or it.get("display_name"),
                it.get("product_type"),
                it.get("detail_desc"),
            )
        ]
        assert not merch, f"occasion merchandise leaked: {[it.get('prod_name') for it in merch]}"

    @pytest.mark.requires_index
    def test_haldi_favours_for_guests_still_returns_merchandise(self) -> None:
        """Explicit merchandise request must still surface the favours
        collection — the inverse-direction regression this fix must not
        introduce (same discipline as test_rakhi_for_my_brother above)."""
        result = self._run_search("haldi favours for guests")
        items = result.get("retrieved_items", [])
        assert items, "expected items for 'haldi favours for guests'"
        favours = [
            it for it in items
            if is_occasion_merchandise_name(
                it.get("prod_name") or it.get("display_name"),
                it.get("product_type"),
                it.get("detail_desc"),
            )
        ]
        assert favours, "expected at least one favour item for an explicit favours request"


class _RealIndexMockLLM:
    """Minimal fixed-response LLM stub — mirrors
    tests/test_kids_leak_exclusion.py's identical helper."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs):
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs):
        yield self._next()
