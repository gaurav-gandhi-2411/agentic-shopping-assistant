/**
 * Client-side store slug → display name map.
 *
 * Used as a fallback when an ItemSummary arrives without a populated
 * `store_display` field (e.g. legacy per-brand responses or future new
 * stores not yet in the server-side config).
 *
 * Keep in sync with src/config/stores.py STORE_CONFIG.
 */

export const STORE_DISPLAY_NAMES = {
  hm: "H&M",
  myntra: "Myntra",
  flipkart: "Flipkart",
  snitch: "Snitch",
  fashor: "Fashor",
  powerlook: "Powerlook",
  virgio: "Virgio",
} as const

export type StoreSlug = keyof typeof STORE_DISPLAY_NAMES

/**
 * Resolve a human-readable display name for a store slug.
 *
 * Falls back to the raw slug (capitalised) when the slug is not in the map,
 * and to undefined when the slug is null/undefined.
 */
export function getStoreDisplayName(slug: string | null | undefined): string | undefined {
  if (!slug) return undefined
  return (
    STORE_DISPLAY_NAMES[slug as StoreSlug] ??
    slug.charAt(0).toUpperCase() + slug.slice(1)
  )
}
