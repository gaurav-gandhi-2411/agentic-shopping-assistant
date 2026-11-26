import { mkdirSync } from "node:fs"
import path from "node:path"
import { expect, test, type Page, type Route } from "@playwright/test"

// Plain-search quality: one-tap Women / Men refine under results whose query named no gender, on
// the anonymous landing chat. Production build with the backend fully stubbed (every request to
// API is intercepted, the chat WebSocket is mocked: no network, no LLM, no production call).
// Product data is a recorded sample of the local-index response (gate run 2026-11-25, see
// reports/plain_search_quality/live_gate_v2.json); image URLs are null so no store CDN is hit.
//
// Screenshots (rule 15c) are written only when E2E_SHOTS=<dir> is set:
//   E2E_SHOTS=../reports/screenshots/plain-search-quality npm run test:e2e -- plain-search-refine

const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8099"
const WS_URL = /\/chat\/stream/
const SHOTS = process.env.E2E_SHOTS
const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
}

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
  gender: "women" | "men",
) => ({
  article_id: id,
  prod_name: name,
  display_name: name,
  colour: "Yellow",
  product_type: type,
  department: "Ethnic",
  image_url: null,
  detail_desc: null,
  score: 0.9,
  price_inr: price,
  store,
  store_display: storeDisplay,
  pdp_url: `https://example.test/${store}/${id}`,
  gender,
})

// BEFORE (retrieval order of the frozen pool, evals/fixtures/rerank_chip_pools/chip02.json): three
// cards from one store, one of them the same product twice, all men's.
const BEFORE = [
  card("b1", "TAJ HALDI SHORT KURTA SET", "kurta set", 17549, "kalpraag", "Kalpraag", "men"),
  card("b2", "Yellow Geometric Printed Kurta Set", "kurta set", 3299, "kisah", "Kisah", "men"),
  card("b3", "Chaitanya Haldi Kurta Set", "kurta set", 16818, "kalpraag", "Kalpraag", "men"),
  card("b4", "Yellow Printed Front Open Kurta Set", "kurta set", 3299, "kisah", "Kisah", "men"),
  card("b5", "Abhiijeet Saawant in Chaitanya Haldi Kurta Set", "kurta set", 16818, "kalpraag", "Kalpraag", "men"),
]

// AFTER (live gate v2, run 1, 'Haldi kurta set'): no duplicate, at most 2 per store, both genders.
const MIXED = [
  card("a1", "Yellow Geometric Printed Kurta Set", "kurta set", 3299, "kisah", "Kisah", "men"),
  card("a2", "Mustard Knit Regular Fit Kurta Set", "kurta set", 2299, "globalrepublic", "Global Republic", "women"),
  card("a3", "Yellow Printed Front Open Kurta Set", "kurta set", 3299, "kisah", "Kisah", "men"),
  card("a4", "Yellow Self Design Regular Fit Kurta Set | Azania", "kurta set", 3667, "jadeblue", "JadeBlue", "men"),
  card("a5", "Haldi Yellow Printed Silk Blend Kurta | The Kurta Company", "kurta", 984, "jadeblue", "JadeBlue", "men"),
]

const WOMEN = [
  card("w1", "Mustard Knit Regular Fit Kurta Set", "kurta set", 2299, "globalrepublic", "Global Republic", "women"),
  card("w2", "Black Printed Silk Blend Straight Kurta Set", "kurta set", 1369, "libas", "Libas", "women"),
  card("w3", "Multi Floral Printed Cotton Straight Kurta Set", "kurta set", 1799, "libas", "Libas", "women"),
]

const REFINE = [
  { gender: "women", query: "Haldi kurta set for women" },
  { gender: "men", query: "Haldi kurta set for men" },
]

interface Behaviour {
  first: { items: unknown[]; refine: unknown[] | null }
  /** What the refine tap returns (items for the gender-constrained query). */
  second?: { items: unknown[] }
  /** Hold the second turn's done frame this long so the loading state can be captured. */
  secondDelayMs?: number
}

async function stubBackend(page: Page, b: Behaviour): Promise<{ sent: string[] }> {
  const sent: string[] = []
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
  let turn = 0
  await page.routeWebSocket(WS_URL, (ws) => {
    ws.onMessage((raw) => {
      const frame = JSON.parse(String(raw)) as { type: string; message?: string }
      if (frame.type !== "user_message") return
      sent.push(String(frame.message))
      turn += 1
      const cur = turn === 1 ? b.first : { items: b.second?.items ?? [], refine: null }
      ws.send(JSON.stringify({ type: "session", conversation_id: "c-e2e" }))
      ws.send(JSON.stringify({ type: "items", items: cur.items }))
      ws.send(JSON.stringify({ type: "token", text: "Here are some options." }))
      setTimeout(
        () => {
          ws.send(
            JSON.stringify({
              type: "done",
              message_id: null,
              final_state: {
                filters: {},
                out_of_catalogue: false,
                new_items_this_turn: true,
                response: "Here are some options.",
                gender_refine: cur.refine,
              },
            }),
          )
        },
        turn === 1 ? 0 : (b.secondDelayMs ?? 0),
      )
    })
  })
  return { sent }
}

async function ask(page: Page, chip: string): Promise<void> {
  await page.goto("/")
  await page.getByRole("button", { name: chip }).click()
}

async function shot(page: Page, name: string): Promise<void> {
  if (!SHOTS) return
  mkdirSync(SHOTS, { recursive: true })
  const w = page.viewportSize()!.width
  await page.setViewportSize({ width: w, height: w <= 480 ? 2400 : 1700 })
  await page.waitForTimeout(400)
  await page.screenshot({ path: path.join(SHOTS, name) })
}

const noOverflow = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)

test("BEFORE: the old result (one store three times, a duplicate, men only) and no refine control", async ({ page }) => {
  await stubBackend(page, { first: { items: BEFORE, refine: null } })
  await ask(page, "Haldi kurta set")
  await expect(page.getByText("Abhiijeet Saawant in Chaitanya Haldi Kurta Set")).toBeVisible()
  await expect(page.getByRole("button", { name: "Women", exact: true })).toHaveCount(0)
  await shot(page, `before-${page.viewportSize()!.width}.png`)
})

test("AFTER: mixed results with Women / Men, 44 px targets, keyboard path, focus ring, tap re-runs the query", async ({
  page,
}) => {
  const { sent } = await stubBackend(page, {
    first: { items: MIXED, refine: REFINE },
    second: { items: WOMEN },
    secondDelayMs: 2500,
  })
  await ask(page, "Haldi kurta set")
  const women = page.getByRole("button", { name: "Women", exact: true })
  const men = page.getByRole("button", { name: "Men", exact: true })
  await expect(women).toBeVisible()
  await expect(men).toBeVisible()
  expect(await noOverflow(page)).toBeLessThanOrEqual(0)

  for (const b of [women, men]) {
    const box = await b.boundingBox()
    expect(box!.height).toBeGreaterThanOrEqual(44)
    expect(box!.width).toBeGreaterThanOrEqual(44)
  }

  // keyboard: focus lands on the control and shows a visible ring (keyboard modality first)
  await women.focus()
  await page.keyboard.press("Tab")
  await page.keyboard.press("Shift+Tab")
  await expect(women).toBeFocused()
  expect(await women.evaluate((el) => getComputedStyle(el).boxShadow)).not.toBe("none")
  await shot(page, `after-${page.viewportSize()!.width}.png`)

  // activate with the keyboard: the same query constrained to women goes out; loading state
  await page.keyboard.press("Enter")
  await expect.poll(() => sent.length).toBe(2)
  expect(sent[1]).toBe("Haldi kurta set for women")
  // the control leaves with the turn that replaces it (latest-assistant-message rule); the
  // shopper sees their own tapped query and the stylist typing, not a stale control
  await expect(page.getByText("Haldi kurta set for women", { exact: true })).toBeVisible()
  await expect(women).toHaveCount(0)
  await shot(page, `loading-${page.viewportSize()!.width}.png`)

  // the gender-constrained answer: women's cards, and no refine control any more (empty state)
  await expect(page.getByText("Black Printed Silk Blend Straight Kurta Set")).toBeVisible({ timeout: 8000 })
  await expect(page.getByRole("button", { name: "Women", exact: true })).toHaveCount(0)
  await expect(page.getByRole("button", { name: "Men", exact: true })).toHaveCount(0)
  await shot(page, `after-women-${page.viewportSize()!.width}.png`)
})

test("empty: a gendered query (Men's kurta for a wedding) shows no refine control", async ({ page }) => {
  await stubBackend(page, { first: { items: WOMEN, refine: null } })
  await ask(page, "Men's kurta for a wedding")
  await expect(page.getByText("Mustard Knit Regular Fit Kurta Set")).toBeVisible()
  await expect(page.getByRole("button", { name: "Women", exact: true })).toHaveCount(0)
  await expect(page.getByRole("button", { name: "Men", exact: true })).toHaveCount(0)
})
