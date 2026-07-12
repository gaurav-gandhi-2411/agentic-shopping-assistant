import { defineConfig } from "vitest/config"

/**
 * Minimal Vitest config, scoped to pure-function unit tests only (currently
 * `lib/poseShape.ts`). Deliberately does NOT configure a browser/DOM
 * environment or the `@mediapipe/tasks-vision` runtime — `lib/poseLandmarker.ts`
 * (the actual MediaPipe API boundary) is browser-only and is proven correct
 * via real-browser testing, not unit tests. See lib/poseShape.test.ts's
 * top-of-file comment for what this test layer can and cannot catch.
 */
export default defineConfig({
  test: {
    include: ["lib/**/*.test.ts"],
    environment: "node",
  },
})
