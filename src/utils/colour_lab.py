"""CIELAB colour utilities for tonal compatibility scoring.

Converts catalogue colour names to CIE L*a*b* triplets and exposes
delta_e_2000 for perceptual distance comparisons.
"""
import logging

import numpy as np
import colour as _colour

# Hex reference values for every unique colour_group_name in the catalogue.
# "Other *" and sentinel values (Other, Unknown, Transparent) use plausible
# fallbacks so they still produce a meaningful LAB value.
_COLOUR_HEX: dict[str, str] = {
    "Beige":           "#F5F5DC",
    "Black":           "#000000",
    "Blue":            "#4169E1",
    "Bronze/Copper":   "#B87333",
    "Dark Beige":      "#C3B099",
    "Dark Blue":       "#00008B",
    "Dark Green":      "#006400",
    "Dark Grey":       "#404040",
    "Dark Orange":     "#FF8C00",
    "Dark Pink":       "#C71585",
    "Dark Purple":     "#4B0082",
    "Dark Red":        "#8B0000",
    "Dark Turquoise":  "#008080",
    "Dark Yellow":     "#B8860B",
    "Gold":            "#FFD700",
    "Green":           "#008000",
    "Greenish Khaki":  "#6B8E23",
    "Grey":            "#808080",
    "Greyish Beige":   "#B5A89A",
    "Light Beige":     "#E8E4D0",
    "Light Blue":      "#ADD8E6",
    "Light Green":     "#90EE90",
    "Light Grey":      "#D3D3D3",
    "Light Orange":    "#FFB347",
    "Light Pink":      "#FFB6C1",
    "Light Purple":    "#D8B4FE",
    "Light Red":       "#FF8080",
    "Light Turquoise": "#AFEEEE",
    "Light Yellow":    "#FFFFE0",
    "Off White":       "#FAF9F6",
    "Orange":          "#FFA500",
    "Other":           "#808080",
    "Other Blue":      "#6699CC",
    "Other Green":     "#4F7942",
    "Other Orange":    "#E25822",
    "Other Pink":      "#FF69B4",
    "Other Purple":    "#9370DB",
    "Other Red":       "#FF3333",
    "Other Turquoise": "#30D5C8",
    "Other Yellow":    "#FFEF00",
    "Pink":            "#FFC0CB",
    "Purple":          "#800080",
    "Red":             "#EE1111",
    "Silver":          "#C0C0C0",
    "Transparent":     "#FFFFFF",
    "Turquoise":       "#40E0D0",
    "Unknown":         "#808080",
    "White":           "#FFFFFF",
    "Yellow":          "#FFFF00",
    "Yellowish Brown": "#9B8540",
}


def _hex_to_lab(hex_code: str) -> np.ndarray:
    h = hex_code.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    xyz = _colour.sRGB_to_XYZ(rgb)
    return _colour.XYZ_to_Lab(xyz)


def _build_lab_map() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, hex_code in _COLOUR_HEX.items():
        try:
            out[name] = _hex_to_lab(hex_code)
        except Exception as exc:
            logging.warning("colour_lab: failed to convert %r (%r): %s", name, hex_code, exc)
    return out


# Computed once at import.
COLOUR_TO_LAB: dict[str, np.ndarray] = _build_lab_map()


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE deltaE 2000 perceptual colour distance between two L*a*b* triplets."""
    return float(_colour.delta_E(lab1, lab2, method="CIE 2000"))
