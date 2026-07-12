"""Regression test for buy-link names (2026-07-10 sweep, P1-7): the catalogue's
display_name carries an internal "( category )" suffix on every row — buy links and
saved-look snapshots must use prod_name, the name the customer actually sees."""
from __future__ import annotations

from src.agents.outfit.cart_links import build_cart_action


class TestBuyLinkNames:
    def test_link_name_uses_prod_name_not_suffixed_display_name(self) -> None:
        items = [
            {
                "article_id": "X1",
                "prod_name": "Mods Western Star Self Design Sherwani",
                "display_name": "Mods Western Star Self Design Sherwani ( Kurtas, Ethnic Sets and Bottoms)",
                "store": "flipkart",
                # Flipkart pdp_handle is already a full URL (see stores.build_pdp_url)
                "pdp_handle": "https://www.flipkart.com/mods-sherwani/p/itm123",
            }
        ]
        result = build_cart_action(items, brand="unified")
        names = [link["name"] for link in result["item_links"]]
        assert names, f"no links built: {result}"
        assert names[0] == "Mods Western Star Self Design Sherwani"
        assert "Ethnic Sets and Bottoms" not in names[0]
