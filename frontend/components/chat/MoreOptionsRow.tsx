"use client"

import type { ReactNode } from "react"
import { MORE_OPTIONS_COPY, MORE_OPTIONS_SKELETON_COUNT } from "@/lib/moreOptions"

/**
 * The "More options" section below a look. Presentational: the caller passes the already-rendered
 * cards (the look's own SlotCard, so buy links and layout are identical) and one of three states.
 *   - loading: a skeleton grid (aria-busy) while the answer streams;
 *   - error:   one short line (role="status") when the payload was malformed;
 *   - ready:   the heading + cards.
 * Empty is not a state of this component: with no cards the caller renders nothing at all.
 * The grid mirrors the look's own (2 columns on a phone, 3 from sm), so 390 px shows 2 across.
 */
export type MoreOptionsStatus = "loading" | "error" | "ready"

export function MoreOptionsRow({
  status,
  budget,
  children,
}: {
  status: MoreOptionsStatus
  budget: number
  children?: ReactNode
}) {
  const heading = MORE_OPTIONS_COPY.heading(budget)

  if (status === "error") {
    return (
      <p
        role="status"
        data-testid="more-options-error"
        className="text-xs text-muted-foreground px-1"
      >
        {MORE_OPTIONS_COPY.errorMessage}
      </p>
    )
  }

  if (status === "loading") {
    return (
      <section
        aria-busy="true"
        aria-label={MORE_OPTIONS_COPY.loadingLabel}
        data-testid="more-options-loading"
        className="space-y-2"
      >
        <div className="h-4 w-40 rounded bg-muted animate-pulse" aria-hidden />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2" aria-hidden>
          {Array.from({ length: MORE_OPTIONS_SKELETON_COUNT }, (_, i) => (
            <div key={i} className="rounded-lg border bg-background overflow-hidden">
              <div className="aspect-[4/5] bg-muted animate-pulse" />
              <div className="p-1.5 space-y-1">
                <div className="h-3 w-5/6 rounded bg-muted animate-pulse" />
                <div className="h-3 w-1/3 rounded bg-muted animate-pulse" />
              </div>
            </div>
          ))}
        </div>
      </section>
    )
  }

  return (
    <section aria-label={heading} data-testid="more-options-row" className="space-y-2">
      <h4 className="text-sm font-semibold text-foreground">{heading}</h4>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">{children}</div>
    </section>
  )
}
