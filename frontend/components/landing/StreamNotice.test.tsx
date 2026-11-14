// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { StreamNotice } from "./StreamNotice"
import { WakingSkeleton } from "./WakingSkeleton"

afterEach(cleanup)

describe("StreamNotice (limit / busy / error states)", () => {
  it("rate_limited says when to retry in human terms, never the raw code", () => {
    render(<StreamNotice kind="rate_limited" retryAfterSeconds={720} onRetry={() => {}} />)
    const alert = screen.getByRole("alert")
    expect(alert.textContent).toContain("try again in about 12 minutes")
    expect(alert.textContent).not.toMatch(/rate_limited|Retry in 720s/)
  })

  it("rate_limited without a known wait still gives a friendly time frame", () => {
    render(<StreamNotice kind="rate_limited" retryAfterSeconds={null} onRetry={() => {}} />)
    expect(screen.getByRole("alert").textContent).toContain("try again in a little while")
  })

  it("demo_limit says try again tomorrow and offers no pointless retry button", () => {
    render(<StreamNotice kind="demo_limit" onRetry={() => {}} />)
    expect(screen.getByRole("alert").textContent).toContain("try again tomorrow")
    expect(screen.queryByRole("button")).toBeNull()
  })

  it("busy says to retry in about a minute and its retry button calls back", () => {
    const onRetry = vi.fn()
    render(<StreamNotice kind="busy" onRetry={onRetry} />)
    expect(screen.getByRole("alert").textContent).toContain("about a minute")
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it("retry buttons are at least 44 px tall (min-h-11)", () => {
    render(<StreamNotice kind="session_error" onRetry={() => {}} />)
    expect(screen.getByRole("button", { name: "Try again" }).className).toContain("min-h-11")
  })
})

describe("WakingSkeleton", () => {
  it("shows the plain waking-up message in a polite live region", () => {
    render(<WakingSkeleton />)
    const region = screen.getByRole("status")
    expect(region.textContent).toContain("waking up")
    expect(region.textContent).toContain("about a minute")
  })
})
