import path from "node:path"
import { defineConfig } from "vitest/config"

/**
 * Vitest config. Two kinds of test live here:
 *  - pure-function / middleware tests (node environment, the default), e.g.
 *    `lib/poseShape.ts`, `lib/publicPaths.ts`, `middleware.ts`;
 *  - component/hook tests (`*.test.tsx`), which opt into jsdom with a
 *    `// @vitest-environment jsdom` docblock on the first line of the file.
 *
 * Deliberately does NOT configure the `@mediapipe/tasks-vision` runtime —
 * `lib/poseLandmarker.ts` (the actual MediaPipe API boundary) is browser-only and is
 * proven correct via real-browser testing, not unit tests. See
 * lib/poseShape.test.ts's top-of-file comment for what this test layer can and
 * cannot catch. The Playwright browser test lives in `e2e/` and is NOT picked up
 * here (see `playwright.config.ts`, `npm run test:e2e`).
 */
export default defineConfig({
  // Same "@/..." alias as tsconfig.json so tests import app modules the way app code does.
  resolve: { alias: { "@": path.resolve(__dirname) } },
  // tsconfig has "jsx": "preserve" (Next compiles JSX itself); vitest must transform it.
  // Vitest 4 transforms with oxc (esbuild options are ignored), hence `oxc`, not `esbuild`.
  oxc: { jsx: { runtime: "automatic" } },
  test: {
    include: [
      "lib/**/*.test.ts",
      "hooks/**/*.test.{ts,tsx}",
      "components/**/*.test.{ts,tsx}",
      "app/**/*.test.ts",
      "middleware.test.ts",
      "next.config.test.ts",
    ],
    environment: "node",
  },
})
