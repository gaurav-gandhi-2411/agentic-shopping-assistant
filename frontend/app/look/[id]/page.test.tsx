// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { COPY } from "@/lib/copy"
import SharedLookPage from "./page"

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const PUBLIC_ITEM = {
  article_id: "p1",
  prod_name: "Public Kurta",
  price_inr: 1299,
  slot_role: "seed",
  store: "snitch",
  store_display: "Snitch",
  buy_url: "https://snitch.co.in/products/p1",
}
// What GET /looks/{id} returns for a non-public store's item (api/look_render.py).
const HIDDEN_ITEM = {
  article_id: "h1",
  prod_name: "Hidden Jacket",
  price_inr: 4999,
  slot_role: "complement",
  store: "myntra",
  store_display: "Myntra",
  buy_url: null,
  pdp_handle: null,
  unavailable: true,
}

async function renderLook(items: unknown[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        id: "x",
        brand: "unified",
        snapshot: { items, item_links: [], cart_url: null },
        created_at: "2026-11-16T00:00:00Z",
      }),
    })),
  )
  render(await SharedLookPage({ params: Promise.resolve({ id: "x" }) }))
}

describe("shared look page", () => {
  it("public item keeps its Buy button; non-public item is 'no longer available' with no link", async () => {
    await renderLook([PUBLIC_ITEM, HIDDEN_ITEM])
    const buy = screen.getAllByRole("link", { name: /buy at/i })
    expect(buy).toHaveLength(1)
    expect(buy[0].getAttribute("href")).toBe("https://snitch.co.in/products/p1")
    expect(screen.getByText(COPY.cards.noLongerAvailable)).toBeTruthy()
    expect(screen.queryByText("₹4,999")).toBeNull()
    expect(screen.getByText("₹1,299")).toBeTruthy()
    expect(document.querySelector('a[href*="myntra"]')).toBeNull()
  })

  it("a stale buy_url on an unavailable item is still never rendered", async () => {
    await renderLook([{ ...HIDDEN_ITEM, buy_url: "https://www.myntra.com/jacket/1" }])
    expect(screen.queryByRole("link", { name: /buy at/i })).toBeNull()
    expect(document.querySelector('a[href*="myntra"]')).toBeNull()
  })

  it("an all-public look is unchanged: no status text", async () => {
    await renderLook([PUBLIC_ITEM])
    expect(screen.queryByText(COPY.cards.noLongerAvailable)).toBeNull()
  })
})
