// "More options" row: alternate anchors under the user's budget, shown below the look.
// Pure helpers + ALL user-visible strings for the row, in one place.

/**
 * PENDING GG APPROVAL: every string below is PROPOSED wording and has NOT been approved.
 * Rules (same as lib/copy.ts): neutral and accurate; no counts, accuracy figures,
 * superlatives, or claims about stock or availability. Do not merge as final copy until GG
 * signs off. Strings are exactly what the UI renders (functions fill the placeholders).
 */
export const MORE_OPTIONS_COPY = {
  /** Row heading and the region's accessible name. `budget` is the stated budget in INR. */
  heading: (budget: number): string => `More options under ₹${formatInr(budget)}`,
  /** Small badge on each card (where the look's cards show the slot name). */
  cardLabel: "Option",
  /** Per-card call to action; the whole card is also the link. `store` is the store name. */
  buyCta: (store: string): string => `Buy at ${store}`,
  /** Accessible name of a card link: name, price, store, and what the link does. */
  cardAriaLabel: (name: string, price: number | null, store: string | null): string =>
    `${name}${price != null ? `, ₹${formatInr(price)}` : ""}${store ? `, ${store}` : ""}. Opens the store page in a new tab.`,
  /** Accessible name while the row is loading (skeleton cards are decorative). */
  loadingLabel: "Loading more options",
  /** Shown when the options payload is malformed; the look itself is unaffected. */
  errorMessage: "More options couldn't be shown right now.",
  // Empty state: the row is hidden entirely, so there is deliberately no string.
} as const

/** Indian digit grouping, e.g. 3000 -> "3,000" (matches the rest of the board). */
export function formatInr(n: number): string {
  return Math.round(n).toLocaleString("en-IN")
}

/** Cards in the loading skeleton (the backend returns at most 6). */
export const MORE_OPTIONS_SKELETON_COUNT = 6

/**
 * True when a chat message states a budget ("under ₹3000", "under 3,000", "below rs 2500",
 * "₹1500 budget"...). Used only to decide whether to show the row's loading skeleton while the
 * answer streams; the server remains the source of truth for whether a row appears.
 */
export function messageStatesBudget(text: string): boolean {
  return /(?:₹|\brs\.?\s*|\binr\s*)\s*\d|\b(?:under|below|within|upto|up to)\s+(?:₹|rs\.?\s*)?\d/i.test(
    text,
  )
}

/** A budgeted, outfit-style ask: the only turns that can produce a "More options" row. */
export function expectsMoreOptions(text: string): boolean {
  return messageStatesBudget(text) && /\b(outfit|look|wear|style|dress)\b/i.test(text)
}

export interface OptionLike {
  article_id: string
  prod_name: string
}

/** Runtime shape check for the WS/REST payload: a non-empty-id, named item list. */
export function isValidOptionList(value: unknown): value is OptionLike[] {
  return (
    Array.isArray(value) &&
    value.every(
      (v) =>
        typeof v === "object" &&
        v !== null &&
        typeof (v as OptionLike).article_id === "string" &&
        (v as OptionLike).article_id !== "" &&
        typeof (v as OptionLike).prod_name === "string",
    )
  )
}
