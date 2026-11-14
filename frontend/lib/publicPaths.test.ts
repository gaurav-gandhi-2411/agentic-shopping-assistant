import { describe, expect, it } from "vitest"
import { isPublicPath } from "./publicPaths"

describe("isPublicPath", () => {
  it("allows the landing page and the crawler files", () => {
    expect(isPublicPath("/")).toBe(true)
    expect(isPublicPath("/robots.txt")).toBe(true)
    expect(isPublicPath("/sitemap.xml")).toBe(true)
  })

  it("keeps every previously public prefix public", () => {
    for (const p of [
      "/login",
      "/auth/callback",
      "/demo",
      "/demo/chat",
      "/embed/snitch",
      "/pdp-demo/snitch",
      "/look/3f2c",
      "/saved-looks",
      "/opengraph-image",
    ]) {
      expect(isPublicPath(p), p).toBe(true)
    }
  })

  it("does not open the authenticated app", () => {
    for (const p of ["/chat", "/dashboard", "/chat/anything", "/admin"]) {
      expect(isPublicPath(p), p).toBe(false)
    }
  })

  it("treats `/` as an exact match only (not a prefix of everything)", () => {
    expect(isPublicPath("/private")).toBe(false)
  })

  it.each([
    "/login",
    "/auth",
    "/demo",
    "/embed",
    "/pdp-demo",
    "/look",
    "/saved-looks",
    "/opengraph-image",
  ])("every pre-existing public prefix stays public: %s", (prefix) => {
    expect(isPublicPath(prefix)).toBe(true)
    expect(isPublicPath(`${prefix}/x`)).toBe(true)
  })

  it.each(["/chat", "/dashboard", "/Chat", "/%2e%2e/chat", "//chat", "/chat/", "/./chat"])(
    "protected or hostile path stays protected: %s",
    (path) => {
      expect(isPublicPath(path)).toBe(false)
    },
  )
})
