from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.config.brand import get_brand_config

router = APIRouter(tags=["brand"])


class BrandResponse(BaseModel):
    display_name: str
    logo_url: str | None
    primary_colour: str
    accent_colour: str
    tagline: str | None
    currency: str
    locale: str
    sizing_system: str
    suggestion_chips: list[str]
    pdp_url_template: str


@router.get("/api/brand", response_model=BrandResponse)
def get_brand() -> BrandResponse:
    """Return the active brand config for frontend theming."""
    cfg = get_brand_config()
    return BrandResponse(
        display_name=cfg.display_name,
        logo_url=cfg.logo_url,
        primary_colour=cfg.primary_colour,
        accent_colour=cfg.accent_colour,
        tagline=cfg.tagline,
        currency=cfg.currency,
        locale=cfg.locale,
        sizing_system=cfg.sizing_system,
        suggestion_chips=cfg.suggestion_chips,
        pdp_url_template=cfg.pdp_url_template,
    )
