// @vitest-environment jsdom
import { cleanup, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { ItemSummary } from "@/lib/api/types"
import { OutfitBoard } from "./OutfitBoard"

// The board's own data hooks/imports are not under test here.
vi.mock("@/hooks/useBrandConfig", () => ({ useBrandConfig: () => ({ data: undefined }) }))
vi.mock("@/components/ProductImage", () => ({
  ProductImage: () => <span data-testid="img" />,
}))

afterEach(cleanup)

const item = (over: Partial<ItemSummary>): ItemSummary =>
  ({
    article_id: "x",
    prod_name: "Item",
    display_name: "Item",
    colour: "Pink",
    product_type: "kurta",
    department: "",
    image_url: null,
    detail_desc: null,
    score: 0.9,
    price_inr: 999,
    store: "libas",
    store_display: "Libas",
    pdp_url: "https://example.test/p/x",
    gender: "women",
    ...over,
  }) as ItemSummary

const LOOK: ItemSummary[] = [
  item({ article_id: "s1", prod_name: "Seed Kurti", slot_role: "seed", pdp_url: "https://example.test/s1" }),
  item({
    article_id: "c1",
    prod_name: "Gold Clutch",
    slot_role: "complement",
    outfit_slot: "accessory",
    pdp_url: "https://example.test/c1",
  }),
]

const OPTIONS: ItemSummary[] = [
  item({ article_id: "o1", prod_name: "Mustard Anarkali", price_inr: 849, pdp_url: "https://example.test/o1" }),
  item({ article_id: "o2", prod_name: "Berry Lehenga", price_inr: 997, store: "tjori", store_display: "Tjori", pdp_url: "https://example.test/o2" }),
]

function board(extra: Partial<Parameters<typeof OutfitBoard>[0]> = {}) {
  return render(
    <OutfitBoard
      items={LOOK}
      lookId="look-1"
      occasion="sangeet"
      lookGender="women"
      budgetTotalInr={1500}
      sessionId="m1"
      anchorItemId="s1"
      anchorCategory="kurti"
      brand="unified"
      {...extra}
    />,
  )
}

describe("OutfitBoard — More options row", () => {
  it("renders the heading, cards with buy links and CTA, below the look", () => {
    board({ moreOptions: OPTIONS, moreOptionsBudgetInr: 3000 })
    const row = screen.getByTestId("more-options-row")
    expect(within(row).getByRole("heading", { name: "More options under ₹3,000" })).toBeTruthy()
    const links = within(row).getAllByRole("link")
    expect(links).toHaveLength(2)
    expect(links[0].getAttribute("href")).toBe("https://example.test/o1") // server-built pdp_url
    expect(links[0].getAttribute("target")).toBe("_blank")
    expect(links[0].getAttribute("rel")).toContain("noopener")
    expect(links[0].getAttribute("aria-label")).toBe(
      "Mustard Anarkali, ₹849, Libas. Opens the store page in a new tab.",
    )
    expect(within(links[0]).getByText("Buy at Libas")).toBeTruthy()
    expect(within(links[0]).getByText("Option")).toBeTruthy() // badge, not a slot name
    // below the look: the row comes after the look's own cards in document order
    const lookCard = screen.getByText("Seed Kurti")
    expect(lookCard.compareDocumentPosition(row) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it("the row is a labelled region and its cards are keyboard-focusable links", () => {
    board({ moreOptions: OPTIONS, moreOptionsBudgetInr: 3000 })
    const region = screen.getByRole("region", { name: "More options under ₹3,000" })
    for (const a of within(region).getAllByRole("link")) {
      expect(a.tabIndex).not.toBe(-1) // anchors with href are in the tab order
    }
  })

  it("empty state: an empty list hides the row entirely", () => {
    board({ moreOptions: [], moreOptionsBudgetInr: 3000 })
    expect(screen.queryByTestId("more-options-row")).toBeNull()
    expect(screen.queryByText(/More options/)).toBeNull()
  })

  it("absent without a budget: nothing rendered", () => {
    board({ moreOptions: null, moreOptionsBudgetInr: null })
    expect(screen.queryByTestId("more-options-row")).toBeNull()
    cleanup()
    board({}) // old-shape payload: field not present at all
    expect(screen.queryByTestId("more-options-row")).toBeNull()
  })

  it("loading state: skeleton (aria-busy) with its own accessible name", () => {
    board({ moreOptionsLoading: true })
    const sk = screen.getByTestId("more-options-loading")
    expect(sk.getAttribute("aria-busy")).toBe("true")
    expect(sk.getAttribute("aria-label")).toBe("Loading more options")
    expect(screen.queryByTestId("more-options-row")).toBeNull()
  })

  it("error state: malformed payload shows one status line; the look still renders", () => {
    board({
      moreOptions: [{ nope: true }] as unknown as ItemSummary[],
      moreOptionsBudgetInr: 3000,
    })
    expect(screen.getByTestId("more-options-error").textContent).toBe(
      "More options couldn't be shown right now.",
    )
    expect(screen.getByText("Seed Kurti")).toBeTruthy()
  })

  it("an old client payload (no more_options) leaves the look untouched", () => {
    board()
    expect(screen.getByText("Seed Kurti")).toBeTruthy()
    expect(screen.getByText("Gold Clutch")).toBeTruthy()
  })
})
