"""Regression tests for the 2026-08-11 anchor-vs-complement gate unification
(src.agents.outfit.composer._anchor_matches_occasion).

Root cause: the anchor (seed item) resolution path used a hand-duplicated,
narrower coherence check (ethnic_lean vs is_ethnic_item only) instead of the
same src.agents.outfit.coherence.is_coherent_candidate every COMPLEMENT is
checked against. That narrower check was a structural no-op for every
EITHER-lean occasion — which includes exactly the 3 occasions that carry a
register gate in is_coherent_candidate specifically because ethnic_lean alone
isn't protection enough for them: office (gate 4), gym (gate 5), party_evening
(gate 6). Confirmed empirically (not just structurally) via
eval/anchor_complement_gate_class_probe.py: 11/66 real composed-look probes
leaked across all 3 occasions before this fix, 0/66 after.
"""
from __future__ import annotations

from src.agents.outfit.composer import _anchor_matches_occasion


def _item(prod_name: str, product_type: str, gender: str = "women") -> dict:
    return {"prod_name": prod_name, "product_type": product_type, "gender": gender}


class TestWesternRegisterOccasionOffice:
    def test_ethnic_anchor_rejected(self) -> None:
        """The exact live-reproduced leak shape: an ethnic saree/kurta anchoring
        an office look, which is_coherent_candidate would reject as a complement."""
        item = _item("Sangria Black Belted Organza Saree", "saree")
        assert _anchor_matches_occasion(item, "office", "women") is False

    def test_casual_marker_anchor_rejected(self) -> None:
        item = _item("Women Black Casual V-Neck Top", "top")
        assert _anchor_matches_occasion(item, "office", "women") is False

    def test_formal_western_anchor_passes(self) -> None:
        item = _item("White & Black Printed Slim Fit Formal Shirt", "shirt", gender="men")
        assert _anchor_matches_occasion(item, "office", "men") is True


class TestAthleticRegisterOccasionGym:
    def test_ethnic_anchor_rejected(self) -> None:
        item = _item("Indo Era Floral Cotton Blend Calf Length Kurta Set", "kurta")
        assert _anchor_matches_occasion(item, "gym", "women") is False

    def test_athletic_anchor_passes(self) -> None:
        item = _item("Zip-Up Sports Bra", "top")
        assert _anchor_matches_occasion(item, "gym", "women") is True


class TestCasualExclusionOccasionPartyEvening:
    def test_casual_marker_anchor_rejected(self) -> None:
        item = _item("Men's Navy Blue Striped Cotton Short Kurta", "kurta", gender="men")
        assert _anchor_matches_occasion(item, "party_evening", "men") is False

    def test_dressy_anchor_passes(self) -> None:
        item = _item("Black Off-Shoulder Party Dress", "dress")
        assert _anchor_matches_occasion(item, "party_evening", "women") is True


class TestEthnicRegisterOccasionsUnchanged:
    """Regression guard: ethnic_only/ethnic_heavy occasions already had real
    anchor-side protection via the old ethnic_lean check — confirm the
    unification preserves that behaviour exactly, not just the 3 previously-
    unprotected occasions."""

    def test_western_anchor_rejected_for_ethnic_only(self) -> None:
        item = _item("Blue Denim Jacket", "jacket")
        assert _anchor_matches_occasion(item, "sangeet", "women") is False

    def test_ethnic_anchor_passes_for_ethnic_heavy(self) -> None:
        item = _item("Biba Women Yellow Ethnic Motifs Printed Anarkali Kurta", "kurta")
        assert _anchor_matches_occasion(item, "haldi", "women") is True


class TestSkipGenderGateContract:
    def test_gender_text_conflict_not_rejected_here(self) -> None:
        """skip_gender_gate=True means this function deliberately does NOT
        reject a gender-text-conflicting item — the caller (compose_outfit's
        anchor resolution, and partner.py's mirror) already applies
        gender_allowed()/has_gender_text_conflict() explicitly alongside this
        call. Double-gating here would be redundant, not a correctness gain."""
        item = _item("Men's Yellow - Dupatta", "dupatta", gender="women")
        # Occasion-register-wise this is fine for an ethnic_heavy occasion;
        # the gender-text-conflict is deliberately NOT this function's job.
        assert _anchor_matches_occasion(item, "haldi", "women") is True
