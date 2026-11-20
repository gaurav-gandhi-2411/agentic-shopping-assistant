import { mkdirSync } from "node:fs"
import path from "node:path"
import { expect, test, type Page, type Route } from "@playwright/test"

// "More options" row (alternate anchors under the budget) on the anonymous landing chat, against a
// production build with the backend fully stubbed: every request to API is intercepted and the
// chat WebSocket is mocked (no network, no LLM, no production call). Product data below is a
// recorded sample of the local-index response for "Sangeet outfit under ₹3,000" (women); image
// URLs are null so no store CDN is hit (cards show the neutral garment glyph).
//
// Screenshots (rule 15c) are written only when E2E_SHOTS=<dir> is set, so a normal run writes
// nothing:  E2E_SHOTS=../reports/screenshots/sangeet-more-options npm run test:e2e -- more-options

const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8099"
const WS_URL = /\/chat\/stream/
const SHOTS = process.env.E2E_SHOTS
const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
}

// Default is the phone viewport from playwright.config.ts (390x844). E2E_DESKTOP=1 runs the same
// tests at 1280x900 for the desktop screenshots.
if (process.env.E2E_DESKTOP) {
  test.use({ viewport: { width: 1280, height: 900 }, isMobile: false, hasTouch: false, deviceScaleFactor: 1 })
}

const card = (
  id: string,
  name: string,
  type: string,
  price: number,
  store: string,
  storeDisplay: string,
  extra: Record<string, unknown> = {},
) => ({
  article_id: id,
  prod_name: name,
  display_name: name,
  colour: "Multi",
  product_type: type,
  department: "Ethnic",
  image_url: null,
  detail_desc: null,
  score: 0.9,
  price_inr: price,
  store,
  store_display: storeDisplay,
  pdp_url: `https://example.test/${store}/${id}`,
  gender: "women",
  ...extra,
})

const LOOK = [
  card("l1", "Set of 2: Snow White Cotton Anarkali Kurti with Flared Pants", "kurti", 1069, "tjori", "Tjori", { slot_role: "seed" }),
  card("l2", "Women's Red Handcrafted Potli Clutch", "bag", 489, "vastramay", "Vastramay", { slot_role: "complement", outfit_slot: "accessory" }),
  card("l3", "Zari Jutti (Gold)", "footwear", 1300, "taurjuttis", "Taur Juttis", { slot_role: "complement", outfit_slot: "footwear" }),
]

const OPTIONS = [
  card("o1", "White Printed Cotton Anarkali Short Kurti", "kurti", 499, "libas", "Libas"),
  card("o2", "Black Tie And Dye Cotton Anarkali Kurta", "kurta", 920, "tjori", "Tjori"),
  card("o3", "Mustard Embellished Silk Blend Anarkali Suit Set With Dupatta", "anarkali", 849, "libas", "Libas"),
  card("o4", "Set of 3 - Berry Blue Floral Handblock Cotton Lehenga Set", "lehenga", 997, "tjori", "Tjori"),
  card("o5", "Ethnic Printed & Embroidered Anarkali Flared Midi Dress", "dress", 999, "fashor", "Fashor"),
  card("o6", "Women's Mustard Embroidered Cotton Kurta", "kurta", 889, "vastramay", "Vastramay"),
]

interface Behaviour {
  /** Extra final_state fields (more_options etc.). */
  finalExtra?: Record<string, unknown>
  /** Hold the done frame this long (so the loading skeleton can be captured). */
  doneDelayMs?: number
}

async function stubBackend(page: Page, b: Behaviour = {}): Promise<void> {
  const json = (route: Route, body: unknown): Promise<void> =>
    route.fulfill({ status: 200, headers: { ...CORS, "content-type": "application/json" }, body: JSON.stringify(body) })
  await page.route(`${API}/**`, async (route) => {
    const req = route.request()
    const p = new URL(req.url()).pathname
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS })
    if (p === "/demo/session")
      return json(route, { session_token: "e2e-token", ws_ticket: "e2e-ticket", expires_in: 3600, brand: "unified" })
    if (p === "/auth/ws-ticket") return json(route, { ticket: "e2e-ticket" })
    if (p === "/brand/config" || p.startsWith("/brand")) return json(route, {})
    return route.fulfill({ status: 404, headers: CORS, body: "{}" })
  })
  await page.routeWebSocket(WS_URL, (ws) => {
    ws.onMessage((raw) => {
      const frame = JSON.parse(String(raw)) as { type: string }
      if (frame.type !== "user_message") return
      ws.send(JSON.stringify({ type: "session", conversation_id: "c-e2e" }))
      ws.send(JSON.stringify({ type: "items", items: LOOK }))
      ws.send(JSON.stringify({ type: "token", text: "**Outfit suggestion**" }))
      setTimeout(() => {
        ws.send(
          JSON.stringify({
            type: "done",
            message_id: null,
            final_state: {
              filters: {},
              out_of_catalogue: false,
              new_items_this_turn: true,
              response: "**Outfit suggestion**",
              look_id: "look-e2e",
              occasion: "sangeet",
              look_gender: "women",
              budget_total_inr: 2858,
              outfit_rationale: "Cotton anarkali with a potli clutch and gold juttis.",
              ...b.finalExtra,
            },
          }),
        )
      }, b.doneDelayMs ?? 0)
    })
  })
}

async function ask(page: Page): Promise<void> {
  await page.goto("/")
  await page.getByRole("button", { name: "Sangeet outfit under ₹3,000" }).click()
}

// The chat scrolls inside its own container (fullPage cannot see it), so grow the viewport.

async function shot(page: Page, name: string): Promise<void> {
  if (!SHOTS) return
  mkdirSync(SHOTS, { recursive: true })
  // Tall viewport so the whole chat fits without its inner scroll, then a plain page capture.
  const w = page.viewportSize()!.width
  await page.setViewportSize({ width: w, height: w <= 480 ? 2400 : 1700 })
  await page.waitForTimeout(400)
  await page.screenshot({ path: path.join(SHOTS, name)})
}

const noOverflow = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)

test("BEFORE: payload without more_options renders the look only (no row)", async ({ page }) => {
  await stubBackend(page)
  await ask(page)
  await expect(page.getByText("Zari Jutti (Gold)")).toBeVisible()
  await expect(page.getByTestId("more-options-row")).toHaveCount(0)
  await expect(page.getByTestId("more-options-loading")).toHaveCount(0)
  await shot(page, `before-${page.viewportSize()!.width}.png`)
})

test("AFTER: six options under the look with buy links, 390 px layout, keyboard reachable", async ({ page }) => {
  await stubBackend(page, { finalExtra: { more_options: OPTIONS, more_options_budget_inr: 3000 } })
  await ask(page)
  const row = page.getByTestId("more-options-row")
  await expect(row).toBeVisible()
  await expect(row.getByRole("heading", { name: "More options under ₹3,000" })).toBeVisible()

  const links = row.getByRole("link")
  await expect(links).toHaveCount(6)
  for (let i = 0; i < 6; i++) {
    const href = await links.nth(i).getAttribute("href")
    expect(href).toBe(`https://example.test/${OPTIONS[i].store}/${OPTIONS[i].article_id}`)
    expect(await links.nth(i).getAttribute("target")).toBe("_blank")
    expect(await links.nth(i).getAttribute("rel")).toContain("noopener")
  }
  await expect(row.getByText("Buy at Libas").first()).toBeVisible()
  // the removed per-card "Option" badge (and any slot badge) must not exist on option cards
  await expect(row.getByText("Option", { exact: true })).toHaveCount(0)
  await expect(row.getByText("Item", { exact: true })).toHaveCount(0)

  // below the look, not overlapping it
  const lookBox = await page.getByText("Zari Jutti (Gold)").boundingBox()
  const rowBox = await row.boundingBox()
  expect(rowBox!.y).toBeGreaterThan(lookBox!.y)

  // 2 cards across on a phone, none wider than the viewport, no horizontal scroll
  const w = page.viewportSize()!.width
  // One synchronous read: the chat auto-scrolls smoothly, so separate boundingBox() calls can straddle a scroll step.
  const [first, second, third] = await links.evaluateAll((els) =>
    els.slice(0, 3).map((e) => e.getBoundingClientRect().toJSON() as { x: number; y: number; width: number }),
  )
  if (w <= 480) {
    expect(Math.abs(first.y - second.y)).toBeLessThan(2) // 2nd is beside the 1st
    expect(third.y).toBeGreaterThan(first.y + 10) // 3rd wraps
  }
  expect(first.width).toBeGreaterThan(100)
  expect(await noOverflow(page)).toBeLessThanOrEqual(0)

  // keyboard: Tab reaches the first option link, which shows a visible focus ring
  await links.nth(0).focus()
  await page.keyboard.press("Tab") // keyboard modality: :focus-visible only applies after a key press
  await page.keyboard.press("Shift+Tab")
  await expect(links.nth(0)).toBeFocused()
  const ring = await links.nth(0).evaluate((el) => getComputedStyle(el).boxShadow)
  expect(ring).not.toBe("none")

  await shot(page, `after-${w}.png`)
})

test("loading: skeleton while the budgeted answer streams, replaced by the row on done", async ({ page }) => {
  await stubBackend(page, {
    doneDelayMs: 2500,
    finalExtra: { more_options: OPTIONS, more_options_budget_inr: 3000 },
  })
  await ask(page)
  const sk = page.getByTestId("more-options-loading")
  await expect(sk).toBeVisible()
  await expect(sk).toHaveAttribute("aria-busy", "true")
  await shot(page, `loading-${page.viewportSize()!.width}.png`)
  await expect(page.getByTestId("more-options-row")).toBeVisible({ timeout: 8000 })
  await expect(sk).toHaveCount(0)
})

test("empty: more_options [] hides the row; absent field never shows a skeleton after done", async ({ page }) => {
  await stubBackend(page, { finalExtra: { more_options: [], more_options_budget_inr: 3000 } })
  await ask(page)
  await expect(page.getByText("Zari Jutti (Gold)")).toBeVisible()
  await expect(page.getByTestId("more-options-row")).toHaveCount(0)
  await expect(page.getByTestId("more-options-loading")).toHaveCount(0)
})

test("error: a malformed payload shows one status line and the look still renders", async ({ page }) => {
  await stubBackend(page, { finalExtra: { more_options: [{ bogus: true }], more_options_budget_inr: 3000 } })
  await ask(page)
  await expect(page.getByTestId("more-options-error")).toHaveText("More options couldn't be shown right now.")
  await expect(page.getByText("Zari Jutti (Gold)")).toBeVisible()
  await shot(page, `error-${page.viewportSize()!.width}.png`)
})
