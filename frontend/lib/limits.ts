// Turning the backend's limit/busy signals into friendly, structured notices.
// Pure functions (no React) so they are unit-testable in plain node.
//
// Backend contract (api/routes/chat.py, src/llm/resilience.py):
//  - WS first frame {type:"error", code:"rate_limited", message:"Rate limit exceeded. Retry in <N>s"}
//  - WS first frame {type:"error", code:"demo_limit",   message:"Demo limit reached for today ..."}
//  - HTTP 429 from POST /demo/session when the daily cap is hit
//  - LLM chain exhausted: LLM_BUSY_MESSAGE streamed as ordinary assistant text.

/**
 * "session_expired" is not a limit: the demo token was rejected (it lasts 1 h). The landing
 * recovers from it once by creating a fresh session; it is never shown as a rate limit.
 */
export type NoticeKind = "rate_limited" | "demo_limit" | "busy" | "session_expired"

export interface LimitNotice {
  kind: NoticeKind
  /** Seconds until a retry is expected to work; only known for rate_limited. */
  retryAfterSeconds: number | null
}

/** Start of the backend's LLM_BUSY_MESSAGE (src/llm/resilience.py); matched as a prefix. */
export const BUSY_MESSAGE_PREFIX = "I'm getting a lot of requests right now"

/** True when an assistant reply is the backend's "LLM busy" text rather than an answer. */
export function isBusyReply(content: string): boolean {
  return content.trimStart().startsWith(BUSY_MESSAGE_PREFIX)
}

/** Extract N from "Rate limit exceeded. Retry in <N>s"; null when absent or nonsensical. */
export function parseRetryAfterSeconds(message: string): number | null {
  const m = /retry in\s+(\d+)\s*s/i.exec(message)
  if (!m) return null
  const seconds = Number.parseInt(m[1], 10)
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null
}

/** Human phrase for a wait: "less than a minute", "about 12 minutes", "about 2 hours". */
export function formatRetryAfter(seconds: number): string {
  if (seconds < 60) return "less than a minute"
  const minutes = Math.ceil(seconds / 60)
  if (minutes === 1) return "about a minute"
  if (minutes < 60) return `about ${minutes} minutes`
  const hours = Math.ceil(minutes / 60)
  return hours === 1 ? "about an hour" : `about ${hours} hours`
}

/**
 * Map a WS error frame to a notice, or null for any other error code (those keep the
 * existing in-bubble error text).
 */
export function noticeFromErrorFrame(code: string, message: string): LimitNotice | null {
  if (code === "rate_limited") {
    return { kind: "rate_limited", retryAfterSeconds: parseRetryAfterSeconds(message) }
  }
  if (code === "demo_limit") return { kind: "demo_limit", retryAfterSeconds: null }
  return null
}

/**
 * Classify a bare 1008 close (no error frame) by its reason text. The backend closes with
 * "Policy violation: invalid or missing token" for a bad/expired demo token, "Demo limit
 * reached" for the daily cap, and "Rate limit exceeded" for the per-IP limiter. Anything
 * unrecognised is treated as a rate limit, the pre-existing behaviour.
 */
export function noticeFromCloseReason(reason: string | undefined): LimitNotice {
  const r = reason ?? ""
  if (/invalid or missing token|policy violation/i.test(r)) {
    return { kind: "session_expired", retryAfterSeconds: null }
  }
  if (/demo limit/i.test(r)) return { kind: "demo_limit", retryAfterSeconds: null }
  return { kind: "rate_limited", retryAfterSeconds: null }
}
