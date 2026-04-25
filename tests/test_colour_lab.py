"""Unit tests for src/utils/colour_lab."""
import pytest
import numpy as np

from src.utils.colour_lab import COLOUR_TO_LAB, delta_e_2000


class TestKnownDistances:
    def test_black_to_white_near_100(self):
        de = delta_e_2000(COLOUR_TO_LAB["Black"], COLOUR_TO_LAB["White"])
        assert 99 <= de <= 101, f"Black-White deltaE expected ~100, got {de:.2f}"

    def test_black_to_dark_grey_under_20(self):
        de = delta_e_2000(COLOUR_TO_LAB["Black"], COLOUR_TO_LAB["Dark Grey"])
        assert de < 20, f"Black-Dark Grey deltaE expected < 20, got {de:.2f}"

    def test_white_to_light_grey_under_30(self):
        de = delta_e_2000(COLOUR_TO_LAB["White"], COLOUR_TO_LAB["Light Grey"])
        assert de < 30, f"White-Light Grey deltaE expected < 30, got {de:.2f}"

    def test_light_pink_to_white_within_25(self):
        # Light Pink is tonally close to White — must pass the neutral-palette threshold
        de = delta_e_2000(COLOUR_TO_LAB["Light Pink"], COLOUR_TO_LAB["White"])
        assert de <= 25, f"Light Pink-White deltaE expected ≤ 25, got {de:.2f}"

    def test_blue_far_from_neutral_palette(self):
        neutral_palette = ["Black", "White", "Off White", "Grey", "Dark Grey", "Light Grey", "Beige", "Light Beige"]
        palette_labs = [COLOUR_TO_LAB[c] for c in neutral_palette]
        blue_lab = COLOUR_TO_LAB["Blue"]
        min_de = min(delta_e_2000(blue_lab, p) for p in palette_labs)
        assert min_de > 25, f"Blue should be far from neutral palette (min ΔE={min_de:.2f})"

    def test_symmetric(self):
        de_ab = delta_e_2000(COLOUR_TO_LAB["Black"], COLOUR_TO_LAB["White"])
        de_ba = delta_e_2000(COLOUR_TO_LAB["White"], COLOUR_TO_LAB["Black"])
        assert abs(de_ab - de_ba) < 1e-6


class TestPaletteLoading:
    EXPECTED_COLOURS = [
        "Beige", "Black", "Blue", "Bronze/Copper", "Dark Beige", "Dark Blue",
        "Dark Green", "Dark Grey", "Dark Orange", "Dark Pink", "Dark Purple",
        "Dark Red", "Dark Turquoise", "Dark Yellow", "Gold", "Green",
        "Greenish Khaki", "Grey", "Greyish Beige", "Light Beige", "Light Blue",
        "Light Green", "Light Grey", "Light Orange", "Light Pink", "Light Purple",
        "Light Red", "Light Turquoise", "Light Yellow", "Off White", "Orange",
        "Other", "Other Blue", "Other Green", "Other Orange", "Other Pink",
        "Other Purple", "Other Red", "Other Turquoise", "Other Yellow", "Pink",
        "Purple", "Red", "Silver", "Transparent", "Turquoise", "Unknown",
        "White", "Yellow", "Yellowish Brown",
    ]

    def test_all_catalogue_colours_loaded(self):
        missing = [c for c in self.EXPECTED_COLOURS if c not in COLOUR_TO_LAB]
        assert not missing, f"Missing LAB entries: {missing}"

    def test_lab_values_are_arrays(self):
        for name, lab in COLOUR_TO_LAB.items():
            assert isinstance(lab, np.ndarray), f"{name}: expected ndarray, got {type(lab)}"
            assert lab.shape == (3,), f"{name}: expected shape (3,), got {lab.shape}"

    def test_black_lab_is_origin(self):
        lab = COLOUR_TO_LAB["Black"]
        assert abs(lab[0]) < 1.0, f"Black L* should be ~0, got {lab[0]:.4f}"
        assert abs(lab[1]) < 1.0, f"Black a* should be ~0, got {lab[1]:.4f}"
        assert abs(lab[2]) < 1.0, f"Black b* should be ~0, got {lab[2]:.4f}"

    def test_white_lab_l_near_100(self):
        lab = COLOUR_TO_LAB["White"]
        assert abs(lab[0] - 100.0) < 1.0, f"White L* should be ~100, got {lab[0]:.4f}"
