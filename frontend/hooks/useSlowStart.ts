"use client"

import { useEffect, useState } from "react"

/** The API is considered slow when it has not answered within this long (Wave 1b V2). */
export const SLOW_START_MS = 4000

/**
 * True once `active` has been continuously true for `thresholdMs`; false again as soon as
 * `active` goes false. Only drives the "waking up" UI: it never cancels or restarts the
 * request it is watching (the Cloud Run cold start takes about a minute and must stay alive).
 */
export function useSlowStart(active: boolean, thresholdMs: number = SLOW_START_MS): boolean {
  const [slow, setSlow] = useState(false)

  useEffect(() => {
    if (!active) {
      setSlow(false)
      return
    }
    const id = setTimeout(() => setSlow(true), thresholdMs)
    return () => clearTimeout(id)
  }, [active, thresholdMs])

  return active && slow
}
