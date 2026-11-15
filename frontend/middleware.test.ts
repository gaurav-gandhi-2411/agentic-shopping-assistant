import { beforeEach, describe, expect, it, vi } from "vitest"
import { NextRequest } from "next/server"

// The middleware's only external dependency is Supabase's session lookup; stub it so
// each test chooses "anonymous" or "signed in" and no network is touched.
let currentUser: { id: string } | null = null
vi.mock("@supabase/ssr", () => ({
  createServerClient: () => ({
    auth: { getUser: async () => ({ data: { user: currentUser } }) },
  }),
}))

import { middleware } from "./middleware"

const ORIGIN = "https://stylemaitri.vercel.app"

async function run(pathname: string): Promise<Response> {
  return middleware(new NextRequest(`${ORIGIN}${pathname}`))
}

/** Where the middleware sent the request, or null when it was let through. */
function redirectedTo(res: Response): string | null {
  if (res.status < 300 || res.status >= 400) return null
  const loc = res.headers.get("location")
  return loc ? new URL(loc).pathname : null
}

describe("middleware: anonymous visitor", () => {
  beforeEach(() => {
    currentUser = null
  })

  it.each(["/", "/robots.txt", "/sitemap.xml", "/login", "/demo", "/look/abc", "/saved-looks"])(
    "serves %s without redirecting",
    async (path) => {
      expect(redirectedTo(await run(path))).toBeNull()
    },
  )

  it.each(["/chat", "/dashboard", "/some/private/page"])(
    "still redirects %s to /login",
    async (path) => {
      expect(redirectedTo(await run(path))).toBe("/login")
    },
  )
})

describe("middleware: signed-in user", () => {
  beforeEach(() => {
    currentUser = { id: "u1" }
  })

  it("sends / and /login to /chat (signed-in behaviour unchanged)", async () => {
    expect(redirectedTo(await run("/"))).toBe("/chat")
    expect(redirectedTo(await run("/login"))).toBe("/chat")
  })

  it("lets /chat through", async () => {
    expect(redirectedTo(await run("/chat"))).toBeNull()
  })
})
