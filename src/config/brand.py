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

    # Catalogue
    catalogue_path: str = "data/processed/catalogue.parquet"
    pdp_url_template: str = ""        # e.g. "https://snitch.co.in/products/{handle}"


@functools.lru_cache(maxsize=1)
def get_brand_config() -> BrandConfig:
    """Load and cache the active brand config. BRAND env var selects the brand (default: hm)."""
    brand = os.environ.get("BRAND", "hm")
    config_path = BRANDS_DIR / f"{brand}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Brand config not found: {config_path}. "
            f"Available brands: {[p.stem for p in BRANDS_DIR.glob('*.yaml')]}"
        )
    with config_path.open() as f:
        data = yaml.safe_load(f)
    return BrandConfig.model_validate(data)
