"use client"

import { useCallback, useEffect, useRef, useState } from "react"

// Bootstraps the anonymous demo session for `/` — the same POST /demo/session and the
// same sessionStorage keys the former /demo page used, so the chat hook (useChatStream),
// the saved-look flow and the image upload keep reading credentials exactly as before.

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

/**
 * Give up on session creation after this long. A cold Cloud Run start measured 49-80 s;
 * browsers' own fetch has no short timeout, so this is the only cap and sits well above it.
 */
export const SESSION_TIMEOUT_MS = 150_000

/** Demo tokens last 1 h (expires_in); reuse a stored one only while it is comfortably fresh. */
const SESSION_REUSE_MAX_AGE_MS = 50 * 60 * 1000

export type DemoSessionStatus = "creating" | "ready" | "error" | "limit"

interface DemoSessionResponse {
  session_token: string
}

function storedSessionIsFresh(): boolean {
  try {
    if (!sessionStorage.getItem("demo_session_token")) return false
    const at = Number(sessionStorage.getItem("demo_session_at"))
    // No timestamp = written by an older flow; its age is unknown, so don't trust it.
    return Number.isFinite(at) && at > 0 && Date.now() - at < SESSION_REUSE_MAX_AGE_MS
  } catch {
    return false
  }
}

export function useDemoSession(): { status: DemoSessionStatus; retry: () => void } {
  const [status, setStatus] = useState<DemoSessionStatus>("creating")
  // One request at a time, and none at all on a StrictMode double-mount.
  const inFlight = useRef(false)

  const create = useCallback(async (): Promise<void> => {
    if (inFlight.current) return
    inFlight.current = true
    setStatus("creating")
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), SESSION_TIMEOUT_MS)
    try {
      const res = await fetch(`${BACKEND_URL}/demo/session`, {
        method: "POST",
        signal: controller.signal,
      })
      if (res.status === 429) {
        setStatus("limit")
        return
      }
      if (!res.ok) throw new Error(`Backend returned ${res.status}`)
      const data = (await res.json()) as DemoSessionResponse
      sessionStorage.setItem("demo_session_token", data.session_token)
      sessionStorage.setItem("demo_session_at", String(Date.now()))
      sessionStorage.setItem("demo_backend_url", BACKEND_URL)
      // "unified" is the cross-store default brand id (see the former /demo page).
      sessionStorage.setItem("demo_brand_id", "unified")
      sessionStorage.setItem("demo_brand_name", "Style Maitri")
      setStatus("ready")
    } catch {
      setStatus("error")
    } finally {
      clearTimeout(timer)
      inFlight.current = false
    }
  }, [])

  useEffect(() => {
    if (storedSessionIsFresh()) {
      setStatus("ready")
      return
    }
    void create()
  }, [create])

  return { status, retry: () => void create() }
}
