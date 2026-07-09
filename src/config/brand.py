from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

BRANDS_DIR = Path(__file__).parent.parent.parent / "brands"


class BrandConfig(BaseModel):
    """All brand-specific configuration for one deployment."""

    # Identity
    display_name: str
    logo_url: str | None = None
    primary_colour: str = "#000000"   # hex, used for Tailwind theme token
    accent_colour: str = "#ffffff"

    # Copy
    tagline: str | None = None
    suggestion_chips: list[str] = Field(default_factory=list)

    # Locale & pricing
    currency: str = "INR"             # ISO 4217 currency code
    locale: str = "en-IN"             # BCP-47 locale tag
    sizing_system: str = "IN"         # "IN" | "EU" | "alpha"
    gender_default: str = "women"     # "men" | "women" | "mixed" (mixed → infer from item)

    # Catalogue
    catalogue_path: str = "data/processed/catalogue.parquet"
    pdp_url_template: str = ""        # e.g. "https://snitch.co.in/products/{handle}"


@functools.lru_cache(maxsize=1)
def get_brand_config() -> BrandConfig:
    """Load and cache the active brand config.

    BRAND env var selects the brand (default: "unified", the cross-store B2C
    deployment — StyleMitra). This matches api/main.py's unified-mode
    detection, where an unset BRAND also means unified mode.
    """
    brand = os.environ.get("BRAND", "unified")
    config_path = BRANDS_DIR / f"{brand}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Brand config not found: {config_path}. "
            f"Available brands: {[p.stem for p in BRANDS_DIR.glob('*.yaml')]}"
        )
    # Explicit encoding: brand YAML files carry non-ASCII copy (e.g. "₹" in
    # suggestion_chips) and this must not depend on the platform's default
    # text encoding (cp1252 on Windows would mangle it into mojibake).
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BrandConfig.model_validate(data)
