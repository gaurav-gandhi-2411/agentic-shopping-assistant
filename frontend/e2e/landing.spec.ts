import { expect, test, type Page, type Route } from "@playwright/test"

// Anonymous landing page on a phone viewport (390x844, touch, 2x), against a production
// build. The backend is fully stubbed: every request to API is intercepted and the chat
// WebSocket is mocked, so the test needs no network and makes no LLM or production call.

const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8099"
const WS_URL = /\/chat\/stream/

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
}

const CANNED_ANSWER = "Here is a canned sangeet answer."
const ITEM = {
  article_id: "e2e-1",
  prod_name: "E2E Test Kurta Set",
  display_name: "E2E Test Kurta Set",
  colour: "Blue",
  product_type: "Kurta Set",
  department: "Ethnic",
  image_url: null,
  detail_desc: null,
  score: 0.9,
  price_inr: 1999,
  store: "teststore",
  store_display: "Test Store",
  pdp_url: "https://example.test/p/1",
  gender: "women",
}

type WsBehaviour = "answer" | "rate_limited"

interface StubOptions {
  /** Hold POST /demo/session this long, like a cold-starting Cloud Run instance. */
  sessionDelayMs?: number
  ws?: WsBehaviour
}

/** Intercept the whole API origin and the chat socket. Returns the recorded user queries. */
async function stubBackend(page: Page, opts: StubOptions = {}): Promise<string[]> {
  const queries: string[] = []
  const json = (route: Route, body: unknown): Promise<void> =>
    route.fulfill({
      status: 200,
      headers: { ...CORS, "content-type": "application/json" },
      body: JSON.stringify(body),
    })

  await page.route(`${API}/**`, async (route) => {
    const req = route.request()
    const path = new URL(req.url()).pathname
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS })
    if (path === "/demo/session") {
      if (opts.sessionDelayMs) await new Promise((r) => setTimeout(r, opts.sessionDelayMs))
      return json(route, {
        session_token: "e2e-token",
        ws_ticket: "e2e-ticket",
        expires_in: 3600,
        brand: "unified",
      })
    }
    if (path === "/auth/ws-ticket") return json(route, { ticket: "e2e-ticket" })
    return route.fulfill({ status: 404, headers: CORS, body: "{}" })
  })

  await page.routeWebSocket(WS_URL, (ws) => {
    ws.onMessage((raw) => {
      const frame = JSON.parse(String(raw)) as { type: string; message?: string }
      if (frame.type !== "user_message") return
      queries.push(frame.message ?? "")
      if (opts.ws === "rate_limited") {
        ws.send(
          JSON.stringify({
            type: "error",
            code: "rate_limited",
            message: "Rate limit exceeded. Retry in 720s",
          }),
        )
        ws.close({ code: 1008, reason: "Rate limit exceeded" })
        return
      }
      ws.send(JSON.stringify({ type: "session", conversation_id: "c-e2e" }))
      ws.send(JSON.stringify({ type: "items", items: [ITEM] }))
      ws.send(JSON.stringify({ type: "token", text: CANNED_ANSWER }))
      ws.send(
        JSON.stringify({
          type: "done",
          message_id: null,
          final_state: {
            filters: {},
            out_of_catalogue: false,
            new_items_this_turn: true,
            response: CANNED_ANSWER,
          },
        }),
      )
    })
  })
  return queries
}

test("anonymous / loads the assistant with no sign-in; chips and input are above the fold", async ({
  page,
}) => {
  await stubBackend(page)
  const response = await page.goto("/")
  expect(response?.status()).toBe(200)
  await expect(page).toHaveURL(/127\.0\.0\.1:\d+\/$/) // not redirected to /login

  await expect(page.getByRole("heading", { level: 1 })).toBeVisible()
  const chips = page.getByRole("region", { name: /try one of these/i }).getByRole("button")
  await expect(chips).toHaveCount(5)

  const viewportHeight = 844
  for (let i = 0; i < 5; i++) {
    const box = await chips.nth(i).boundingBox()
    expect(box, `chip ${i} has a box`).not.toBeNull()
    expect(box!.height, `chip ${i} tap height`).toBeGreaterThanOrEqual(44)
    expect(box!.y + box!.height, `chip ${i} above the fold`).toBeLessThanOrEqual(viewportHeight)
  }
  const input = page.getByPlaceholder("Ask about outfits…")
  const inputBox = await input.boundingBox()
  expect(inputBox!.y + inputBox!.height).toBeLessThanOrEqual(viewportHeight)

  // No horizontal scroll at 390 px.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  )
  expect(overflow).toBeLessThanOrEqual(0)

  // crawlers: robots.txt allows /, sitemap lists /
  const robots = await page.request.get("/robots.txt")
  expect(robots.status()).toBe(200)
  expect(await robots.text()).toContain("Allow: /")
  const sitemap = await page.request.get("/sitemap.xml")
  expect(sitemap.status()).toBe(200)
  expect(await sitemap.text()).toContain("https://stylemaitri.vercel.app/")
})

test("one tap on a chip streams a real answer (first answer in <= 2 taps)", async ({ page }) => {
  const queries = await stubBackend(page)
  await page.goto("/")
  await page.getByRole("button", { name: "Haldi kurta set" }).tap() // tap 1
  await expect(page.getByText(CANNED_ANSWER)).toBeVisible()
  await expect(page.getByText("E2E Test Kurta Set")).toBeVisible()
  expect(queries).toEqual(["Haldi kurta set"])
})

test("slow start: nothing before 4 s, then skeleton + 'waking up'; the queued tap is answered", async ({
  page,
}) => {
  const queries = await stubBackend(page, { sessionDelayMs: 7_000 })
  await page.goto("/")
  const waking = page.getByTestId("waking-skeleton")

  await page.getByRole("button", { name: "Wedding guest saree" }).tap() // queued: no 2nd tap needed
  await page.waitForTimeout(2_500)
  await expect(waking).toHaveCount(0) // well under 4 s: quiet

  await expect(waking).toBeVisible({ timeout: 4_000 }) // appears at ~4 s
  await expect(waking).toContainText(/waking up/i)
  await expect(waking).toContainText(/about a minute/i)

  // Session arrives at ~7 s; the request was kept alive, the queued query goes out
  // once, and the answer replaces the skeleton.
  await expect(page.getByText(CANNED_ANSWER)).toBeVisible({ timeout: 15_000 })
  await expect(waking).toHaveCount(0)
  expect(queries).toEqual(["Wedding guest saree"])
})

test("rate_limited shows a friendly retry-time message, not an error code", async ({ page }) => {
  await stubBackend(page, { ws: "rate_limited" })
  await page.goto("/")
  await page.getByRole("button", { name: "Office wear under ₹1,500" }).tap()
  // Next's own route announcer is also role=alert, so target the notice by its marker.
  const alert = page.locator('[role="alert"][data-notice="rate_limited"]')
  await expect(alert).toContainText("try again in about 12 minutes")
  await expect(alert).not.toContainText(/rate_limited|720/)
  await expect(alert.getByRole("button", { name: "Try again" })).toBeVisible()
})
