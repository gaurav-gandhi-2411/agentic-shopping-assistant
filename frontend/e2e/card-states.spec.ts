import { createServer, type Server } from "node:http"
import { mkdirSync } from "node:fs"
import path from "node:path"
import { expect, test, type Page, type Route } from "@playwright/test"

// Card states from the live price check (V3(d)) on a phone viewport, against the same stubbed
// backend as landing.spec.ts: API requests are intercepted, the chat socket is mocked, and a
// tiny in-process HTTP server stands in for GET /looks/{id} (the look page fetches it on the
// server, where page.route cannot reach). No real store or backend is contacted.
//
// Set E2E_SCREENSHOT_DIR to also write PNGs (used for PR evidence); nothing is written otherwise.
// E2E_STRIP_LIVE_FIELDS=1 sends the same payloads WITHOUT the new fields, i.e. what the app
// received before V3(d), for the "before" screenshots.

const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8099"
const WS_URL = /\/chat\/stream/
const SHOT_DIR = process.env.E2E_SCREENSHOT_DIR
const STRIP = process.env.E2E_STRIP_LIVE_FIELDS === "1"
const PREFIX = STRIP ? "before" : "after"

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
}

const card = (id: string, name: string, extra: Record<string, unknown> = {}) => ({
  article_id: id,
  prod_name: name,
  display_name: name,
  colour: "Blue",
  product_type: "Kurta Set",
  department: "Ethnic",
  image_url: null,
  detail_desc: null,
  score: 0.02,
  price_inr: 1999,
  store: "libas",
  store_display: "Libas",
  pdp_url: `https://libas.in/products/${id}`,
  gender: "women",
  ...extra,
})

const LIVE_FIELDS = ["unavailable", "price_may_have_changed", "indexed_price_inr"]
function maybeStrip<T extends Record<string, unknown>>(o: T): T {
  if (!STRIP) return o
  const copy: Record<string, unknown> = { ...o }
  for (const k of LIVE_FIELDS) delete copy[k]
  return copy as T
}

const CHAT_ITEMS = [
  card("ok-1", "Indigo Cotton Kurta Set"),
  card("chg-1", "Maroon Silk Kurta Set", { price_may_have_changed: true }),
  card("live-1", "Teal Straight Kurta", { price_inr: 2499, indexed_price_inr: 1999 }),
  card("gone-1", "Ivory Anarkali Set", { unavailable: true, pdp_url: null }),
].map(maybeStrip)

const LOOK_ITEMS = [
  {
    article_id: "p1",
    prod_name: "Snitch Linen Shirt",
    price_inr: 1299,
    slot_role: "seed",
    outfit_slot: "top",
    store: "snitch",
    store_display: "Snitch",
    buy_url: "https://snitch.co.in/products/p1",
  },
  maybeStrip({
    article_id: "h1",
    prod_name: "Myntra Bomber Jacket",
    price_inr: 4999,
    slot_role: "complement",
    outfit_slot: "outerwear",
    store: "myntra",
    store_display: "Myntra",
    buy_url: STRIP ? "https://www.myntra.com/jacket/1" : null,
    pdp_handle: STRIP ? "jacket/1" : null,
    unavailable: true,
  }),
]

let stub: Server

test.beforeAll(async () => {
  const port = Number(new URL(API).port)
  stub = createServer((req, res) => {
    const headers = { ...CORS, "content-type": "application/json" }
    if (req.url?.startsWith("/looks/")) {
      res.writeHead(200, headers)
      res.end(
        JSON.stringify({
          id: "card-states",
          brand: "unified",
          occasion: "casual",
          look_gender: "men",
          look_total_inr: 6298,
          snapshot: { items: LOOK_ITEMS, item_links: [], cart_url: null },
          created_at: "2026-11-16T00:00:00Z",
        }),
      )
      return
    }
    res.writeHead(404, headers)
    res.end("{}")
  })
  await new Promise<void>((resolve) => stub.listen(port, "127.0.0.1", resolve))
})

test.afterAll(async () => {
  await new Promise<void>((resolve) => stub.close(() => resolve()))
})

async function stubChat(page: Page): Promise<void> {
  const json = (route: Route, body: unknown) =>
    route.fulfill({
      status: 200,
      headers: { ...CORS, "content-type": "application/json" },
      body: JSON.stringify(body),
    })
  await page.route(`${API}/**`, async (route) => {
    const req = route.request()
    const p = new URL(req.url()).pathname
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS })
    if (p === "/demo/session") {
      return json(route, { session_token: "t", ws_ticket: "w", expires_in: 3600, brand: "unified" })
    }
    if (p === "/auth/ws-ticket") return json(route, { ticket: "w" })
    return route.fulfill({ status: 404, headers: CORS, body: "{}" })
  })
  await page.routeWebSocket(WS_URL, (ws) => {
    ws.onMessage((raw) => {
      if ((JSON.parse(String(raw)) as { type: string }).type !== "user_message") return
      ws.send(JSON.stringify({ type: "session", conversation_id: "c-cards" }))
      ws.send(JSON.stringify({ type: "items", items: CHAT_ITEMS }))
      ws.send(JSON.stringify({ type: "token", text: "Here are some kurtas." }))
      ws.send(
        JSON.stringify({
          type: "done",
          message_id: null,
          final_state: {
            filters: {},
            out_of_catalogue: false,
            new_items_this_turn: true,
            response: "Here are some kurtas.",
          },
        }),
      )
    })
  })
}

async function shot(page: Page, name: string): Promise<void> {
  if (!SHOT_DIR) return
  mkdirSync(SHOT_DIR, { recursive: true })
  // The dev server's floating Next.js badge is not part of the app; hide it in evidence shots.
  await page.addStyleTag({ content: "nextjs-portal { display: none !important }" })
  await page.screenshot({ path: path.join(SHOT_DIR, `${PREFIX}-${name}.png`), fullPage: true })
}

test("chat cards: unavailable has no Buy; may-have-changed keeps Buy; live price shown", async ({
  page,
}) => {
  await stubChat(page)
  // The chat thread scrolls inside the page, so evidence shots use a viewport tall enough to
  // hold all four cards.
  if (SHOT_DIR) await page.setViewportSize({ width: 390, height: 3400 })
  await page.goto("/")
  await page.getByRole("button", { name: "Haldi kurta set" }).tap()
  await expect(page.getByText("Here are some kurtas.")).toBeVisible()
  await expect(page.getByText("Indigo Cotton Kurta Set")).toBeVisible()
  await page.waitForTimeout(300) // let image placeholders settle for a stable screenshot
  await shot(page, "chat-cards")

  if (STRIP) return // "before" run only captures pixels
  const unavailable = page.locator('[data-unavailable="true"]')
  await expect(unavailable).toHaveCount(1)
  await expect(unavailable).toContainText("No longer available")
  await expect(unavailable.getByRole("link")).toHaveCount(0)
  await expect(page.getByRole("link", { name: /buy at libas/i })).toHaveCount(3)
  await expect(page.getByText("Price may have changed")).toHaveCount(1)
  await expect(page.getByText("₹2,499")).toBeVisible()
})

test("shared look: non-public item is 'no longer available' with no Buy and no link", async ({
  page,
}) => {
  // The page caches its server-side fetch per URL (ISR), so each mode uses its own id.
  await page.goto(`/look/card-states-${PREFIX}`)
  await expect(page.getByText("Snitch Linen Shirt")).toBeVisible()
  await page.waitForTimeout(300)
  await shot(page, "shared-look")

  if (STRIP) return
  await expect(page.getByText("No longer available")).toHaveCount(1)
  await expect(page.getByRole("link", { name: /buy at snitch/i })).toHaveCount(1)
  await expect(page.getByRole("link", { name: /buy at myntra/i })).toHaveCount(0)
  expect(await page.locator('a[href*="myntra"]').count()).toBe(0)
})
