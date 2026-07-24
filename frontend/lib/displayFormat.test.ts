/**
 * Unit tests for `displayFormat.ts` — the canonical humanizer for raw
 * `product_type`/`garment_type`/`outfit_slot` catalogue and internal facet
 * values, gating the class of bug behind the live-proven "sports_bra" leak
 * (see module docstring): a raw snake_case identifier rendered verbatim,
 * underscore intact, into a user-facing surface.
 */
import { describe, expect, it } from "vitest"
import { formatProductType, formatSlotLabel } from "./displayFormat"

describe("formatProductType", () => {
  it("replaces underscores with spaces for known snake_case product_type values", () => {
    expect(formatProductType("sports_bra")).toBe("sports bra")
    expect(formatProductType("track_pants")).toBe("track pants")
    expect(formatProductType("cargo_pants")).toBe("cargo pants")
  })

  it("leaves already-clean values unchanged", () => {
    expect(formatProductType("kurta")).toBe("kurta")
    expect(formatProductType("dress")).toBe("dress")
  })

  it("leaves already space-separated, already-cased raw catalogue values unchanged", () => {
    // Real catalogue product_type_name values left untouched by the
    // normalizer at low confidence (data/processed/unified_enriched/
    // catalogue.parquet) — already correctly cased, no underscore to fix.
    expect(formatProductType("T-Shirts")).toBe("T-Shirts")
    expect(formatProductType("Co-ords")).toBe("Co-ords")
    expect(formatProductType("Accessories")).toBe("Accessories")
  })

  it("is null/undefined/blank safe", () => {
    expect(formatProductType(null)).toBe("")
    expect(formatProductType(undefined)).toBe("")
    expect(formatProductType("")).toBe("")
    expect(formatProductType("   ")).toBe("")
  })

  it("trims surrounding whitespace", () => {
    expect(formatProductType("  sports_bra  ")).toBe("sports bra")
  })
})

describe("formatSlotLabel", () => {
  it("title-cases a clean slot name", () => {
    expect(formatSlotLabel("footwear")).toBe("Footwear")
    expect(formatSlotLabel("top")).toBe("Top")
  })

  it("sanitizes underscores before title-casing — the same failure class as the product_type leak, defended before it can ever fire on a real slot name", () => {
    expect(formatSlotLabel("some_new_slot")).toBe("Some New Slot")
  })

  it("is null/undefined/blank safe", () => {
    expect(formatSlotLabel(null)).toBe("")
    expect(formatSlotLabel(undefined)).toBe("")
    expect(formatSlotLabel("")).toBe("")
  })
})
