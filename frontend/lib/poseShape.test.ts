/**
 * Unit tests for the pure `classifyBodyShape` gate logic in `poseShape.ts`.
 *
 * SCOPE / HONESTY NOTE: this file can only prove that `classifyBodyShape`
 * behaves correctly GIVEN a landmark array — it cannot reproduce the actual
 * P0 bug (a flat-garment product photo being confidently classified as a
 * body shape), because that bug lives one layer up, at the MediaPipe API
 * boundary in `poseLandmarker.ts` (`minPoseDetectionConfidence` /
 * `minPosePresenceConfidence`, and MediaPipe's own decision that a "pose"
 * exists at all in a photo with no person in it). MediaPipe's
 * `PoseLandmarker` is a browser/WASM runtime backed by a real ML model —
 * there is no meaningful way to unit-test "does MediaPipe itself decide a
 * flat t-shirt photo contains a pose" without a browser and the actual
 * model, which is why real-browser verification (uploading `t-shirt.webp`
 * vs. a real person photo to the running app) is the authoritative proof for
 * this bug, not this file.
 *
 * What these tests DO cover, as defense-in-depth on the downstream gates:
 *  - the visibility gate rejects a low-confidence landmark set (the shape
 *    of signal a spurious/partial pose would plausibly produce) instead of
 *    guessing a shape from it;
 *  - the frontality gate rejects asymmetric (non-frontal / partial) poses;
 *  - the shoulder:hip dead zone never guesses one of rectangle/hourglass/
 *    apple;
 *  - missing landmarks and degenerate (zero-width) geometry both fail
 *    closed to `null`, never throw;
 *  - the happy path still classifies confidently-frontal, high-visibility
 *    landmarks correctly, so the gates aren't so strict they break the
 *    feature for real photos.
 *  - the gate constants themselves are pinned, so a future accidental
 *    loosening (e.g. dropping VISIBILITY_MIN to unblock some edge case)
 *    shows up as a failing test, not a silent regression.
 */
import { describe, expect, it } from "vitest"
import {
  classifyBodyShape,
  FRONTALITY_MAX_RATIO,
  FRONTALITY_MIN_RATIO,
  INVERTED_TRIANGLE_MIN_RATIO,
  LEFT_HIP_IDX,
  LEFT_SHOULDER_IDX,
  PEAR_MAX_RATIO,
  PHOTO_REACHABLE_SHAPES,
  RIGHT_HIP_IDX,
  RIGHT_SHOULDER_IDX,
  VISIBILITY_MIN,
  type PoseLandmarkPoint,
} from "./poseShape"

/** MediaPipe always returns 33 landmarks; only 11/12/23/24 matter here. */
const LANDMARK_COUNT = 33

/**
 * Builds a full 33-point landmark array with a frontal, symmetric,
 * high-visibility shoulder/hip quad at the given widths — every other point
 * is a harmless zero-visibility filler (classifyBodyShape never reads them).
 */
function buildLandmarks(opts: {
  shoulderWidth: number
  hipWidth: number
  visibility?: number
  /** Shift the whole shoulder pair off-center to break frontality symmetry. */
  shoulderXOffset?: number
}): PoseLandmarkPoint[] {
  const { shoulderWidth, hipWidth, visibility = 0.95, shoulderXOffset = 0 } = opts
  const landmarks: PoseLandmarkPoint[] = Array.from({ length: LANDMARK_COUNT }, () => ({
    x: 0,
    y: 0,
    visibility: 0,
  }))
  const centerX = 0.5
  landmarks[LEFT_SHOULDER_IDX] = {
    x: centerX - shoulderWidth / 2 + shoulderXOffset,
    y: 0.3,
    visibility,
  }
  landmarks[RIGHT_SHOULDER_IDX] = {
    x: centerX + shoulderWidth / 2 + shoulderXOffset,
    y: 0.3,
    visibility,
  }
  landmarks[LEFT_HIP_IDX] = { x: centerX - hipWidth / 2, y: 0.6, visibility }
  landmarks[RIGHT_HIP_IDX] = { x: centerX + hipWidth / 2, y: 0.6, visibility }
  return landmarks
}

describe("gate constants (pinned — loosening these is a deliberate, reviewed change)", () => {
  it("keeps the documented thresholds", () => {
    expect(VISIBILITY_MIN).toBe(0.7)
    expect(FRONTALITY_MIN_RATIO).toBe(0.7)
    expect(FRONTALITY_MAX_RATIO).toBe(1.3)
    expect(PEAR_MAX_RATIO).toBe(0.9)
    expect(INVERTED_TRIANGLE_MIN_RATIO).toBe(1.1)
  })

  it("only ever reaches pear / inverted_triangle", () => {
    expect(PHOTO_REACHABLE_SHAPES).toEqual(["pear", "inverted_triangle"])
  })
})

describe("classifyBodyShape happy path", () => {
  it("classifies a narrower-shoulder, wider-hip frontal pose as pear", () => {
    const landmarks = buildLandmarks({ shoulderWidth: 0.3, hipWidth: 0.4 })
    expect(classifyBodyShape(landmarks)).toBe("pear")
  })

  it("classifies a wider-shoulder, narrower-hip frontal pose as inverted_triangle", () => {
    const landmarks = buildLandmarks({ shoulderWidth: 0.42, hipWidth: 0.3 })
    expect(classifyBodyShape(landmarks)).toBe("inverted_triangle")
  })
})

describe("classifyBodyShape confidence gates (fail closed to null)", () => {
  it("rejects a low-visibility landmark set instead of guessing — the shape of signal a spurious/non-person pose would plausibly produce", () => {
    const landmarks = buildLandmarks({ shoulderWidth: 0.3, hipWidth: 0.4, visibility: 0.4 })
    expect(classifyBodyShape(landmarks)).toBeNull()
  })

  it("rejects visibility exactly at the boundary below the floor", () => {
    const landmarks = buildLandmarks({
      shoulderWidth: 0.3,
      hipWidth: 0.4,
      visibility: VISIBILITY_MIN - 0.01,
    })
    expect(classifyBodyShape(landmarks)).toBeNull()
  })

  it("accepts visibility exactly at the floor", () => {
    const landmarks = buildLandmarks({
      shoulderWidth: 0.3,
      hipWidth: 0.4,
      visibility: VISIBILITY_MIN,
    })
    expect(classifyBodyShape(landmarks)).toBe("pear")
  })

  it("rejects a non-frontal (asymmetric) pose even with high visibility", () => {
    const landmarks = buildLandmarks({
      shoulderWidth: 0.3,
      hipWidth: 0.4,
      shoulderXOffset: 0.15,
    })
    expect(classifyBodyShape(landmarks)).toBeNull()
  })

  it("never guesses rectangle/hourglass/apple in the shoulder:hip dead zone", () => {
    const landmarks = buildLandmarks({ shoulderWidth: 0.35, hipWidth: 0.35 })
    expect(classifyBodyShape(landmarks)).toBeNull()
  })

  it("fails closed when required landmarks are missing entirely", () => {
    const landmarks: PoseLandmarkPoint[] = []
    expect(classifyBodyShape(landmarks)).toBeNull()
  })

  it("fails closed on degenerate (zero-width) geometry instead of throwing", () => {
    const landmarks = buildLandmarks({ shoulderWidth: 0, hipWidth: 0.4 })
    expect(classifyBodyShape(landmarks)).toBeNull()
  })
})
