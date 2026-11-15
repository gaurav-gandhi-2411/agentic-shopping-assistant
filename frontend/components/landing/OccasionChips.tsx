"use client"

import { COPY } from "@/lib/copy"

interface Props {
  chips: readonly string[]
  /** Called with the chip's text, which is the exact query to run. */
  onSelect: (query: string) => void
}

/**
 * Tappable occasion prompts. One tap runs a real query (first answer in <= 2 taps).
 * Each chip is a real <button> with its label as its accessible name and a 44 px
 * minimum tap height (min-h-11 = 2.75rem). Text uses the ink colour, never champagne
 * (champagne on cream is 2.49:1, below WCAG).
 */
export function OccasionChips({ chips, onSelect }: Props) {
  return (
    <section aria-labelledby="occasion-chips-heading">
      <h2
        id="occasion-chips-heading"
        className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground"
      >
        {COPY.landing.chipsHeading}
      </h2>
      <ul className="flex flex-wrap gap-2">
        {chips.map((chip) => (
          <li key={chip}>
            <button
              type="button"
              onClick={() => onSelect(chip)}
              className="inline-flex min-h-11 items-center justify-center rounded-full border border-champagne/40 bg-background px-4 py-2 text-sm text-foreground transition-colors hover:border-primary hover:bg-primary hover:text-primary-foreground"
            >
              {chip}
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
