// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { OCCASION_CHIPS } from "@/lib/copy"
import { OccasionChips } from "./OccasionChips"

afterEach(cleanup)

describe("OccasionChips", () => {
  it("offers 4 to 6 chips and includes the spec's example queries", () => {
    expect(OCCASION_CHIPS.length).toBeGreaterThanOrEqual(4)
    expect(OCCASION_CHIPS.length).toBeLessThanOrEqual(6)
    for (const q of [
      "Sangeet outfit under ₹3,000",
      "Haldi kurta set",
      "Wedding guest saree",
      "Office wear under ₹1,500",
    ]) {
      expect(OCCASION_CHIPS).toContain(q)
    }
  })

  it("renders one real button per chip, named by its label, with a 44 px tap height", () => {
    render(<OccasionChips chips={OCCASION_CHIPS} onSelect={() => {}} />)
    const group = screen.getByRole("region", { name: /try one of these/i })
    const buttons = within(group).getAllByRole("button")
    expect(buttons).toHaveLength(OCCASION_CHIPS.length)
    for (const [i, button] of buttons.entries()) {
      expect(button.textContent).toBe(OCCASION_CHIPS[i])
      // min-h-11 = 2.75rem = 44 px at the default 16 px root; the real pixel size is
      // asserted in the Playwright test (e2e/landing.spec.ts) with bounding boxes.
      expect(button.className).toContain("min-h-11")
    }
  })

  it("tapping a chip submits exactly that chip's query, once", () => {
    const onSelect = vi.fn()
    render(<OccasionChips chips={OCCASION_CHIPS} onSelect={onSelect} />)
    fireEvent.click(screen.getByRole("button", { name: "Haldi kurta set" }))
    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(onSelect).toHaveBeenCalledWith("Haldi kurta set")
  })
})
