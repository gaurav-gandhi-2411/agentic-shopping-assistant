// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { ItemSummary } from "@/lib/api/types"
import { COPY } from "@/lib/copy"
import { ItemCard } from "./ItemCard"

// ItemCard reads the brand config through React Query; the card states under test do not
// depend on it, so stub the hook (no provider, no network).
vi.mock("@/hooks/useBrandConfig", () => ({ useBrandConfig: () => ({ data: undefined }) }))

afterEach(cleanup)

const BASE: ItemSummary = {
  article_id: "a1",
  prod_name: "Test Kurta",
  display_name: "Test Kurta",
  colour: "Blue",
  product_type: "Kurta",
  department: "Ethnic",
  image_url: null,
  detail_desc: null,
  score: 0.02,
  price_inr: 1999,
  store: "libas",
  store_display: "Libas",
  pdp_url: "https://libas.in/products/test-kurta",
  gender: "women",
}

describe("ItemCard live-check states", () => {
  it("normal card: price and Buy button, no status text", () => {
    render(<ItemCard item={BASE} />)
    expect(screen.getByText("₹1,999")).toBeTruthy()
    const buy = screen.getByRole("link", { name: /buy at libas/i })
    expect(buy.getAttribute("href")).toBe("https://libas.in/products/test-kurta")
    expect(screen.queryByText(COPY.cards.noLongerAvailable)).toBeNull()
    expect(screen.queryByText(COPY.cards.priceMayHaveChanged)).toBeNull()
  })

  it("price_may_have_changed: keeps the indexed price and Buy, adds the note", () => {
    render(<ItemCard item={{ ...BASE, price_may_have_changed: true }} />)
    expect(screen.getByText("₹1,999")).toBeTruthy()
    expect(screen.getByText(COPY.cards.priceMayHaveChanged)).toBeTruthy()
    expect(screen.getByRole("link", { name: /buy at libas/i })).toBeTruthy()
  })

  it("live price replaces the indexed price in the display", () => {
    render(<ItemCard item={{ ...BASE, price_inr: 2499, indexed_price_inr: 1999 }} />)
    expect(screen.getByText("₹2,499")).toBeTruthy()
    expect(screen.queryByText("₹1,999")).toBeNull()
  })

  it("unavailable: 'no longer available', no price and no Buy link", () => {
    const { container } = render(<ItemCard item={{ ...BASE, unavailable: true, pdp_url: null }} />)
    expect(screen.getByText(COPY.cards.noLongerAvailable)).toBeTruthy()
    expect(screen.queryByRole("link")).toBeNull()
    expect(screen.queryByText("₹1,999")).toBeNull()
    expect(container.querySelector('[data-unavailable="true"]')).not.toBeNull()
  })

  it("unavailable wins even if a stale pdp_url is still present", () => {
    render(<ItemCard item={{ ...BASE, unavailable: true }} />)
    expect(screen.queryByRole("link", { name: /buy at/i })).toBeNull()
  })

  it("unavailable does not also show the price-may-have-changed note", () => {
    render(<ItemCard item={{ ...BASE, unavailable: true, price_may_have_changed: true }} />)
    expect(screen.queryByText(COPY.cards.priceMayHaveChanged)).toBeNull()
  })
})
