// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SLOW_START_MS, useSlowStart } from "./useSlowStart"

beforeEach(() => {
  vi.useFakeTimers()
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("useSlowStart", () => {
  it("is quiet at 3.9 s and true at 4 s", () => {
    const { result } = renderHook(() => useSlowStart(true))
    expect(SLOW_START_MS).toBe(4000)
    act(() => {
      vi.advanceTimersByTime(3900)
    })
    expect(result.current).toBe(false)
    act(() => {
      vi.advanceTimersByTime(100)
    })
    expect(result.current).toBe(true)
  })

  it("stays true for as long as the wait lasts (70 s)", () => {
    const { result } = renderHook(() => useSlowStart(true))
    act(() => {
      vi.advanceTimersByTime(70_000)
    })
    expect(result.current).toBe(true)
  })

  it("clears the moment the wait ends and re-arms on the next wait", () => {
    const { result, rerender } = renderHook(({ active }) => useSlowStart(active), {
      initialProps: { active: true },
    })
    act(() => {
      vi.advanceTimersByTime(5000)
    })
    expect(result.current).toBe(true)
    rerender({ active: false })
    expect(result.current).toBe(false)
    rerender({ active: true })
    act(() => {
      vi.advanceTimersByTime(3900)
    })
    expect(result.current).toBe(false)
  })

  it("never fires while inactive", () => {
    const { result } = renderHook(() => useSlowStart(false))
    act(() => {
      vi.advanceTimersByTime(60_000)
    })
    expect(result.current).toBe(false)
  })
})
