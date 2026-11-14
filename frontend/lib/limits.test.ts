import { describe, expect, it } from "vitest"
import {
  formatRetryAfter,
  isBusyReply,
  noticeFromCloseReason,
  noticeFromErrorFrame,
  parseRetryAfterSeconds,
} from "./limits"

describe("parseRetryAfterSeconds", () => {
  it("reads N from the backend's rate-limit message", () => {
    expect(parseRetryAfterSeconds("Rate limit exceeded. Retry in 720s")).toBe(720)
    expect(parseRetryAfterSeconds("Rate limit exceeded. Retry in 12s")).toBe(12)
  })

  it("returns null when there is no number (hostile / unexpected input)", () => {
    expect(parseRetryAfterSeconds("Rate limit exceeded")).toBeNull()
    expect(parseRetryAfterSeconds("")).toBeNull()
    expect(parseRetryAfterSeconds("Retry in -5s")).toBeNull()
  })
})

describe("formatRetryAfter", () => {
  it("speaks in human units", () => {
    expect(formatRetryAfter(12)).toBe("less than a minute")
    expect(formatRetryAfter(60)).toBe("about a minute")
    expect(formatRetryAfter(720)).toBe("about 12 minutes")
    expect(formatRetryAfter(721)).toBe("about 13 minutes")
    expect(formatRetryAfter(3600)).toBe("about an hour")
    expect(formatRetryAfter(7300)).toBe("about 3 hours")
  })
})

describe("noticeFromErrorFrame", () => {
  it("maps rate_limited with the parsed wait", () => {
    expect(noticeFromErrorFrame("rate_limited", "Rate limit exceeded. Retry in 90s")).toEqual({
      kind: "rate_limited",
      retryAfterSeconds: 90,
    })
  })

  it("maps demo_limit and ignores every other code", () => {
    expect(noticeFromErrorFrame("demo_limit", "Demo limit reached")?.kind).toBe("demo_limit")
    expect(noticeFromErrorFrame("stream_error", "Stream generation failed")).toBeNull()
  })
})

describe("isBusyReply", () => {
  it("recognises the backend's busy text and nothing else", () => {
    expect(
      isBusyReply(
        "I'm getting a lot of requests right now and couldn't put together a reply. Please try again in a minute.",
      ),
    ).toBe(true)
    expect(isBusyReply("Here are some sangeet outfits.")).toBe(false)
  })
})

describe("noticeFromCloseReason (bare 1008 close)", () => {
  it("an invalid/expired token is session_expired, never a rate limit", () => {
    expect(noticeFromCloseReason("Policy violation: invalid or missing token").kind).toBe(
      "session_expired",
    )
    expect(noticeFromCloseReason("Policy violation").kind).toBe("session_expired")
  })

  it("keeps the daily-cap and rate-limit reasons", () => {
    expect(noticeFromCloseReason("Demo limit reached").kind).toBe("demo_limit")
    expect(noticeFromCloseReason("Rate limit exceeded").kind).toBe("rate_limited")
  })

  it("unknown or missing reasons stay a rate limit (pre-existing behaviour)", () => {
    expect(noticeFromCloseReason("").kind).toBe("rate_limited")
    expect(noticeFromCloseReason(undefined).kind).toBe("rate_limited")
  })
})
