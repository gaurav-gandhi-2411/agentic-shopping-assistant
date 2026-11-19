import { afterEach, describe, expect, it, vi } from "vitest"

// withSentryConfig only wraps the object for build tooling; unwrap it so the test sees
// the real Next config (and never loads the Sentry build plugin).
vi.mock("@sentry/nextjs", () => ({ withSentryConfig: (config: unknown) => config }))

import nextConfig from "./next.config"

afterEach(() => {
  vi.unstubAllEnvs()
})

describe("next.config redirects: /demo and /demo/chat become aliases of /", () => {
  it("redirects both to / (temporary; Next keeps the query string for redirects)", async () => {
    vi.stubEnv("NEXT_PUBLIC_MULTI_BRAND", "")
    const redirects = await nextConfig.redirects!()
    expect(redirects).toEqual([
      { source: "/demo", destination: "/", permanent: false },
      { source: "/demo/chat", destination: "/", permanent: false },
    ])
    // A fixed destination with no :param / ? in it: Next appends the incoming query itself,
    // so /demo?ref=x lands on /?ref=x.
    for (const r of redirects) expect(r.destination).toBe("/")
  })

  it("is skipped when the shelved per-brand picker is on (NEXT_PUBLIC_MULTI_BRAND=1)", async () => {
    vi.stubEnv("NEXT_PUBLIC_MULTI_BRAND", "1")
    expect(await nextConfig.redirects!()).toEqual([])
  })

  it("only the exact value '1' disables it", async () => {
    vi.stubEnv("NEXT_PUBLIC_MULTI_BRAND", "true")
    expect((await nextConfig.redirects!()).length).toBe(2)
  })
})
