"use client"

import { useState } from "react"
import { COPY } from "@/lib/copy"
import type { GenderRefineOption } from "@/lib/api/types"

interface Props {
  /** From the backend (`gender_refine`): each option's `query` is the same request constrained to one gender. */
  options: readonly GenderRefineOption[] | null | undefined
  /** Sends the option's query as the next user message. */
  onSelect: (query: string) => void
}

/**
 * One-tap Women / Men refine for a plain search whose query named no gender (the results
 * mix both). Labels come from COPY.refine only.
 *
 * States: empty = renders nothing when the backend offered no options (a gendered query such
 * as "Men's kurta ..." or "saree"); loading = a tap disables both buttons at once (no double
 * send) and the new turn then replaces this message as the latest assistant message, so the
 * control leaves while the chat's own typing indicator runs; error = the control has no error UI of
 * its own: a failed turn becomes the latest assistant message with the existing chat error
 * text (and rate-limit / busy notice), so this control is no longer shown and the shopper
 * re-asks the same way as after any other failed turn.
 *
 * Real <button>s (keyboard: Tab / Enter / Space), a visible plum focus ring, 44 px minimum
 * tap height (min-h-11) like the landing chips.
 */
export function GenderRefine({ options, onSelect }: Props) {
  const [pending, setPending] = useState(false)
  const known = (options ?? []).filter((o) => o.gender === "women" || o.gender === "men")
  if (known.length === 0) return null

  return (
    <div className="flex flex-wrap gap-2" aria-busy={pending}>
      {known.map((option) => (
        <button
          key={option.gender}
          type="button"
          disabled={pending}
          onClick={() => {
            setPending(true)
            onSelect(option.query)
          }}
          className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-full border border-champagne/40 bg-background px-4 py-2 text-sm text-foreground transition-colors hover:border-primary hover:bg-primary hover:text-primary-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
        >
          {COPY.refine[option.gender]}
        </button>
      ))}
    </div>
  )
}
