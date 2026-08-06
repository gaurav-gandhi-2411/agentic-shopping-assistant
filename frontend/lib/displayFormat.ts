/**
 * Canonical text-formatting helpers for raw catalogue/internal facet values
 * (`product_type`, `garment_type`, `outfit_slot`/slot names) rendered into
 * ANY user-facing surface.
 *
 * Root-cause context (live-proven bug, 2026-07-24): a gym look's item card
 * showed the raw internal identifier "sports_bra" (underscore intact) in its
 * facet label, right under the product name — `ItemCard.tsx` read
 * `item.product_type` directly into JSX text. This is a SEPARATE surface from
 * the backend's deterministic prose (`src/agents/outfit/rationale.py`'s
 * `_display_noun`, fixed the same day) — that fix does not cover raw API
 * fields the frontend renders itself, because the API intentionally sends the
 * raw `product_type` facet value as-is (see `api/schemas.py::ItemSummary` —
 * there is no separate humanized `display_product_type` field; humanizing
 * catalogue facets for display is frontend-owned by design).
 *
 * `product_type`/`garment_type` values are produced by
 * `src/catalogue/normalizer.py` (confirmed against
 * `data/processed/unified_enriched/catalogue.parquet` and the normalizer's
 * compound-term table) and are always one of two shapes:
 *   - a lowercase, underscore-joined slug when the normalizer is confident,
 *     e.g. "sports_bra", "track_pants", "cargo_pants", "dress", "kurta";
 *   - the original catalogue `product_type_name`, already space-separated and
 *     already correctly cased (e.g. "T-Shirts", "Co-ords", "Accessories"),
 *     when normalizer confidence is low and the value is left untouched.
 * No irregular-casing exceptions (acronyms, camelCase, etc.) exist in either
 * population today — a plain underscore -> space replacement is correct for
 * both, and is a no-op on values that already have no underscore (case 2
 * above). If a future catalogue source ever needs an exception, add it to
 * `IRREGULAR_CASING_OVERRIDES` below rather than hand-patching a call site.
 *
 * `outfit_slot` values (`src/agents/outfit/slots.py`'s `SlotSpec.slot_name`)
 * are single English words today ("top", "bottom", "footwear", "outerwear",
 * "accessory") with no underscores — but a hand-rolled
 * `slug.charAt(0).toUpperCase() + slug.slice(1)` at the render site (as
 * `OutfitBoard.tsx`'s `SlotCard` and `/look/[id]/page.tsx` both did before
 * this fix) is the EXACT bug shape that produced the `product_type` leak,
 * just dormant because no slot name has an underscore YET. Routing both
 * through `formatSlotLabel` closes that class before it can ever fire live.
 *
 * THIS is the ONLY module any user-facing surface may use to render a raw
 * `product_type` / `garment_type` / `outfit_slot` / slot-name value — never
 * inline a `.replace("_", " ")` or `.charAt(0).toUpperCase()` at a new call
 * site; add a case here instead.
 */

/**
 * Per-value casing/wording exceptions, keyed by the raw value's lowercased
 * form. Empty today — see module docstring: no real catalogue value has
 * needed one yet (plain space-replacement covers every value seen in
 * `data/processed/unified_enriched/catalogue.parquet` and every slug the
 * normalizer's compound-term table can produce). Add entries here — not at
 * call sites — the day one genuinely does.
 */
const IRREGULAR_CASING_OVERRIDES: Readonly<Record<string, string>> = {}

/**
 * Humanize a raw `product_type`/`garment_type` catalogue facet for
 * chrome-style display text (e.g. ItemCard's "Snitch · sports bra · Black"
 * row) — replaces underscores with spaces and applies any known override;
 * otherwise leaves casing exactly as the catalogue supplied it (mirrors the
 * backend's `_display_noun` convention of never forcing title case).
 *
 * @param raw - the raw `product_type`/`garment_type` value, or null/undefined.
 * @returns the humanized string, or "" for null/undefined/blank input.
 */
export function formatProductType(raw: string | null | undefined): string {
  if (!raw) return ""
  const trimmed = raw.trim()
  if (!trimmed) return ""
  const override = IRREGULAR_CASING_OVERRIDES[trimmed.toLowerCase()]
  if (override) return override
  return trimmed.replace(/_/g, " ")
}

/**
 * Humanize a raw `outfit_slot`/slot-name value into a short Title Case UI
 * label (e.g. a slot-card badge or a suppressed-slot note) — underscore ->
 * space (via {@link formatProductType}), then title-cases every word. Safe
 * on values that already have no underscore (a no-op there beyond casing).
 *
 * @param raw - the raw slot name, or null/undefined.
 * @returns the Title Case label, or "" for null/undefined/blank input.
 */
export function formatSlotLabel(raw: string | null | undefined): string {
  const spaced = formatProductType(raw)
  if (!spaced) return ""
  return spaced.replace(/\b\w/g, (c) => c.toUpperCase())
}
