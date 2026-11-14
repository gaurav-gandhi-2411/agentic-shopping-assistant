import { defineConfig } from "@playwright/test"

/**
 * End-to-end config for the anonymous-landing test (`npm run test:e2e`). Separate from
 * vitest on purpose: vitest only globs lib/ hooks/ components/, never e2e/.
 *
 * Needs a PRODUCTION build made with the stub URLs baked in (NEXT_PUBLIC_* is inlined at
 * build time), then serves it on E2E_PORT. No real backend is contacted: the spec
 * intercepts every request to E2E_API_URL and mocks the chat WebSocket.
 *
 *   NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8099 \
 *   NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:54321 NEXT_PUBLIC_SUPABASE_ANON_KEY=e2e \
 *     npm run build && npm run test:e2e
 *
 * Browsers are installed on demand (`npx playwright install chromium`), never committed.
 */
const PORT = Number(process.env.E2E_PORT ?? 3210)

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    // Phone-sized: 390x844 CSS px at 2x with touch, matching the Wave 1b above-the-fold target.
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
    browserName: "chromium",
    // Failure artifacts only; nothing is written on a green run.
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `npm run start -- -p ${PORT}`,
    url: `http://127.0.0.1:${PORT}/robots.txt`,
    reuseExistingServer: true,
    timeout: 120_000,
    env: { NEXT_TELEMETRY_DISABLED: "1" },
  },
})
