"use client"

import { COPY } from "@/lib/copy"
import { formatRetryAfter, type NoticeKind } from "@/lib/limits"

export type StreamNoticeKind = NoticeKind | "session_error"

interface Props {
  kind: StreamNoticeKind
  /** Seconds until a retry should work (rate_limited only). */
  retryAfterSeconds?: number | null
  /** Retry affordance; omitted for demo_limit, where retrying today cannot help. */
  onRetry?: () => void
}

function messageFor(kind: StreamNoticeKind, retryAfterSeconds: number | null | undefined): string {
  switch (kind) {
    case "rate_limited":
      return COPY.notices.rateLimited(
        retryAfterSeconds == null ? null : formatRetryAfter(retryAfterSeconds),
      )
    case "demo_limit":
      return COPY.notices.demoLimit
    case "busy":
      return COPY.notices.busy
    case "session_expired": // only reached if automatic recovery already failed once
    case "session_error":
      return COPY.notices.sessionError
  }
}

/**
 * Friendly state for every "the stylist cannot answer right now" case: per-IP rate limit,
 * daily demo cap, busy LLM, and a failed session start. Never shows a raw error code.
 * role="alert" so screen readers announce it when it appears.
 */
export function StreamNotice({ kind, retryAfterSeconds, onRetry }: Props) {
  const showRetry = onRetry != null && kind !== "demo_limit"
  return (
    <div
      role="alert"
      data-notice={kind}
      className="mx-4 my-3 rounded-xl border border-border bg-muted px-4 py-3 text-sm text-foreground"
    >
      <p>{messageFor(kind, retryAfterSeconds)}</p>
      {showRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 inline-flex min-h-11 items-center justify-center rounded-lg bg-primary px-4 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
        >
          {COPY.notices.retryButton}
        </button>
      )}
    </div>
  )
}
