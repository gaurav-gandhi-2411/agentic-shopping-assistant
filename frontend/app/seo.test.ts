import { describe, expect, it } from "vitest"
import robots from "./robots"
import sitemap from "./sitemap"

describe("robots.txt", () => {
  it("allows / and keeps sign-in and the authenticated app out, with the sitemap URL", () => {
    const r = robots()
    const rules = Array.isArray(r.rules) ? r.rules : [r.rules]
    expect(rules).toHaveLength(1)
    const rule = rules[0]
    expect(rule.userAgent).toBe("*")
    expect(rule.allow).toBe("/")
    expect(rule.disallow).toEqual(["/login", "/auth", "/chat", "/dashboard"])
    expect(r.sitemap).toBe("https://stylemaitri.vercel.app/sitemap.xml")
  })
})

describe("sitemap.xml", () => {
  it("lists exactly the landing page", () => {
    expect(sitemap()).toEqual([{ url: "https://stylemaitri.vercel.app/" }])
  })
})
