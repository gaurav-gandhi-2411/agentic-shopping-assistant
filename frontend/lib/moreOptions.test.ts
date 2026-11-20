import { describe, expect, it } from "vitest"
import {
  MORE_OPTIONS_COPY,
  expectsMoreOptions,
  formatInr,
  isValidOptionList,
  messageStatesBudget,
} from "./moreOptions"

describe("MORE_OPTIONS_COPY (PENDING GG APPROVAL strings)", () => {
  it("heading carries the stated budget with Indian grouping", () => {
    expect(MORE_OPTIONS_COPY.heading(3000)).toBe("More options under ₹3,000")
    expect(MORE_OPTIONS_COPY.heading(15000)).toBe("More options under ₹15,000")
    expect(formatInr(100000)).toBe("1,00,000")
  })
  it("card strings", () => {
    expect(MORE_OPTIONS_COPY.buyCta("Tjori")).toBe("Buy at Tjori")
    expect(MORE_OPTIONS_COPY.cardAriaLabel("Cotton Kurta", 999, "Libas")).toBe(
      "Cotton Kurta, ₹999, Libas. Opens the store page in a new tab.",
    )
    expect(MORE_OPTIONS_COPY.cardAriaLabel("Cotton Kurta", null, null)).toBe(
      "Cotton Kurta. Opens the store page in a new tab.",
    )
  })
})

describe("messageStatesBudget / expectsMoreOptions", () => {
  it.each([
    "sangeet outfit under ₹3000 for women",
    "Sangeet outfit under ₹3,000",
    "office look below rs 2500",
    "wedding guest outfit within 6000",
    "mehendi outfit ₹4000 budget",
    "casual outfit upto 2500 for men",
  ])("detects a budget in %s", (t) => {
    expect(messageStatesBudget(t)).toBe(true)
    expect(expectsMoreOptions(t)).toBe(true)
  })
  it.each(["sangeet outfit for women", "haldi look", "show me red sarees"])(
    "no budget in %s",
    (t) => {
      expect(messageStatesBudget(t)).toBe(false)
      expect(expectsMoreOptions(t)).toBe(false)
    },
  )
  it("a budgeted product search is not an outfit ask", () => {
    expect(messageStatesBudget("red saree under ₹2000")).toBe(true)
    expect(expectsMoreOptions("red saree under ₹2000")).toBe(false)
  })
})

describe("isValidOptionList", () => {
  it("accepts well-formed and empty lists", () => {
    expect(isValidOptionList([])).toBe(true)
    expect(isValidOptionList([{ article_id: "a", prod_name: "A" }])).toBe(true)
  })
  it.each([
    [null],
    [undefined],
    ["x"],
    [{}],
    [[{ article_id: "", prod_name: "A" }]],
    [[{ article_id: "a" }]],
    [[null]],
  ])("rejects malformed %j", (v) => {
    expect(isValidOptionList(v)).toBe(false)
  })
})
