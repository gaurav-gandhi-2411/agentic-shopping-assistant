"use client"

import { COPY } from "@/lib/copy"

/**
 * Slow-start state: shown when the API has not answered within 4 s (cold start can take
 * about a minute). A pulsing skeleton plus a plain-language message, in a polite live
 * region so assistive tech announces it once. The underlying request stays alive.
 */
export function WakingSkeleton() {
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="waking-skeleton"
      className="mx-4 my-3 space-y-3"
    >
      <p className="text-sm text-foreground">
        <span className="font-semibold">{COPY.slowStart.title}.</span>{" "}
        <span className="text-muted-foreground">{COPY.slowStart.message}</span>
      </p>
      <div aria-hidden className="flex gap-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-28 flex-1 animate-pulse rounded-xl bg-muted" />
        ))}
      </div>
      <span className="sr-only">{COPY.slowStart.skeletonLabel}</span>
    </div>
  )
}
