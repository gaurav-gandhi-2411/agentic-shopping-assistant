// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SESSION_TIMEOUT_MS, useDemoSession } from "./useDemoSession"

type Resolver = (res: Response) => void

let fetchMock: ReturnType<typeof vi.fn>
let pending: { resolve: Resolver; signal: AbortSignal | undefined }[]

beforeEach(() => {
  vi.useFakeTimers()
  sessionStorage.clear()
  pending = []
  // A fetch that stays open until the test resolves it, and rejects when aborted,
  // like a real request held by a cold-starting server.
  fetchMock = vi.fn((_url: string, init?: RequestInit) => {
    return new Promise<Response>((resolve, reject) => {
      pending.push({ resolve, signal: init?.signal ?? undefined })
      init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
    })
  })
  vi.stubGlobal("fetch", fetchMock)
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status })
}

describe("useDemoSession", () => {
  it("keeps one request alive through a 70 s cold start, then becomes ready", async () => {
    const { result } = renderHook(() => useDemoSession())
    expect(result.current.status).toBe("creating")
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      vi.advanceTimersByTime(70_000)
    })
    // Still creating: not aborted, not retried, no duplicate request.
    expect(result.current.status).toBe("creating")
    expect(pending[0].signal?.aborted).toBe(false)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      pending[0].resolve(json({ session_token: "tok" }))
    })
    expect(result.current.status).toBe("ready")
    expect(sessionStorage.getItem("demo_session_token")).toBe("tok")
    expect(sessionStorage.getItem("demo_brand_id")).toBe("unified")
  })

  it("reports an error after the 150 s cap instead of hanging forever", async () => {
    const { result } = renderHook(() => useDemoSession())
    await act(async () => {
      vi.advanceTimersByTime(SESSION_TIMEOUT_MS - 1)
    })
    expect(result.current.status).toBe("creating")
    await act(async () => {
      vi.advanceTimersByTime(2)
    })
    expect(result.current.status).toBe("error")
  })

  it("maps HTTP 429 to the limit state and 5xx to an error, and retry starts a new request", async () => {
    const { result } = renderHook(() => useDemoSession())
    await act(async () => {
      pending[0].resolve(json({ detail: "capped" }, 429))
    })
    expect(result.current.status).toBe("limit")

    await act(async () => {
      result.current.retry()
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    await act(async () => {
      pending[1].resolve(json({}, 503))
    })
    expect(result.current.status).toBe("error")
  })

  it("reuses a fresh stored session without any request", async () => {
    sessionStorage.setItem("demo_session_token", "stored")
    sessionStorage.setItem("demo_session_at", String(Date.now()))
    const { result } = renderHook(() => useDemoSession())
    await act(async () => {})
    expect(result.current.status).toBe("ready")
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("does not trust a stored token with no timestamp", async () => {
    sessionStorage.setItem("demo_session_token", "unknown-age")
    renderHook(() => useDemoSession())
    await act(async () => {})
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
