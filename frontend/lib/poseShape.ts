/**
 * Pure, framework-free body-shape classification from MediaPipe Pose
 * Landmarker output.
 *
 * Deliberately zero runtime imports (only plain numbers in/out) so this file
 * can be exercised in a plain Node script without pulling in
 * `@mediapipe/tasks-vision`, React, or a browser DOM — mirrors the
 * `src/agents/outfit/body_type.py` pattern of keeping the scoring logic in an
 * isolated, dependency-free module.
 *
 * NEVER asserts a numeric measurement. Output is either one of the 2
 * PHOTO_REACHABLE_SHAPES ("pear", "inverted_triangle" — the only 2 of the 5
 * `src/agents/outfit/body_type.py` `BASE_SHAPES` slugs this landmark-only
 * heuristic can confidently distinguish), or `null` when the confidence gate
 * fails. `null` means "say nothing" — callers must fall through to the
 * existing manual chip/type flow, never show an error. "rectangle", "apple",
 * and "hourglass" are deliberately NEVER returned by this module: a balanced
 * shoulder:hip ratio (this heuristic's only signal) is equally consistent
 * with all 3 of them, and only a waist measurement (which we don't have)
 * could tell them apart — see classifyBodyShape's dead-zone comment.
 *
 * Landmark indices follow MediaPipe's 33-point BlazePose topology:
 *   11 = left shoulder, 12 = right shoulder, 23 = left hip, 24 = right hip.
 */

export interface PoseLandmarkPoint {
  x: number
  y: number
  /** MediaPipe's per-landmark visibility score, 0-1 (undefined treated as 0). */
  visibility?: number
}

export type BodyShapeSlug =
  | "pear"
  | "apple"
  | "hourglass"
  | "rectangle"
  | "inverted_triangle"

/** Only these 2 are ever reachable from the photo heuristic (see module
 *  docstring: a balanced shoulder:hip ratio — the "dead zone" — is genuinely
 *  ambiguous among rectangle/hourglass/apple; none of those 3 can be
 *  distinguished from shoulder/hip landmarks alone, so the dead zone falls
 *  through to `null` ("not confident") rather than guessing one of them). */
export const PHOTO_REACHABLE_SHAPES: readonly BodyShapeSlug[] = [
  "pear",
  "inverted_triangle",
]

// ---------------------------------------------------------------------------
// Confidence-gate thresholds (verbatim from the feature spec — do not loosen
// without re-reviewing the "never assert, never overclaim" hard rule).
// ---------------------------------------------------------------------------

/** Gate (a): per-landmark visibility floor for both shoulders + both hips. */
export const VISIBILITY_MIN = 0.7

/** Gate (b): frontality — left/right distance-from-center symmetry ratio
 *  must fall within this band for BOTH the shoulder pair and the hip pair.
 *  Outside this band the pose is treated as a 3/4 or profile angle. */
export const FRONTALITY_MIN_RATIO = 0.7
export const FRONTALITY_MAX_RATIO = 1.3

/** Gate (c) / classification: shoulder-width : hip-width ratio thresholds.
 *  < PEAR_MAX_RATIO -> pear. > INVERTED_TRIANGLE_MIN_RATIO -> inverted
 *  triangle. Otherwise (the dead zone, [PEAR_MAX_RATIO, INVERTED_TRIANGLE_
 *  MIN_RATIO]) -> `null` ("not confident") — a balanced shoulder:hip ratio is
 *  genuinely ambiguous among rectangle/hourglass/apple (all three can present
 *  a near-1.0 ratio; only a waist measurement distinguishes them, which this
 *  landmark set does not have), so we deliberately do NOT guess one. */
export const PEAR_MAX_RATIO = 0.9
export const INVERTED_TRIANGLE_MIN_RATIO = 1.1

/** Landmark indices used (MediaPipe BlazePose topology). */
export const LEFT_SHOULDER_IDX = 11
export const RIGHT_SHOULDER_IDX = 12
export const LEFT_HIP_IDX = 23
export const RIGHT_HIP_IDX = 24

function ratioWithinBand(a: number, b: number, min: number, max: number): boolean {
  if (a <= 0 || b <= 0) return false
  const ratio = a / b
  return ratio >= min && ratio <= max
}

/**
 * Classify a body shape from a single detected pose's 33 landmarks.
 *
 * Returns `null` (no suggestion) if ANY gate fails:
 *  - a required landmark is missing,
 *  - any of the 4 key landmarks' visibility is below {@link VISIBILITY_MIN},
 *  - the pose fails the frontality symmetry check (gate b),
 *  - shoulder or hip width collapses to zero (degenerate geometry),
 *  - the shoulder:hip ratio falls in the ambiguous dead zone (gate c) — see
 *    {@link PEAR_MAX_RATIO}'s doc comment for why this is a "not confident"
 *    result rather than a guessed "rectangle".
 *
 * @param landmarks - Full landmark array as returned by
 *   `PoseLandmarkerResult.landmarks[0]` (33 points, index-addressed).
 */
export function classifyBodyShape(landmarks: PoseLandmarkPoint[]): BodyShapeSlug | null {
  const leftShoulder = landmarks[LEFT_SHOULDER_IDX]
  const rightShoulder = landmarks[RIGHT_SHOULDER_IDX]
  const leftHip = landmarks[LEFT_HIP_IDX]
  const rightHip = landmarks[RIGHT_HIP_IDX]
  if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return null

  // Gate (a): visibility floor on all 4 landmarks.
  const visibilities = [leftShoulder, rightShoulder, leftHip, rightHip].map(
    (lm) => lm.visibility ?? 0
  )
  if (visibilities.some((v) => v < VISIBILITY_MIN)) return null

  // Gate (b): frontality — left/right symmetry of each pair's distance from
  // the body's horizontal center (average x of all 4 landmarks). A 3/4 or
  // profile pose collapses one side toward the centerline asymmetrically
  // even when visibility alone stays above the floor.
  const centerX = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4
  const shoulderLeftDist = Math.abs(leftShoulder.x - centerX)
  const shoulderRightDist = Math.abs(rightShoulder.x - centerX)
  const hipLeftDist = Math.abs(leftHip.x - centerX)
  const hipRightDist = Math.abs(rightHip.x - centerX)

  const shoulderSymmetric = ratioWithinBand(
    shoulderLeftDist,
    shoulderRightDist,
    FRONTALITY_MIN_RATIO,
    FRONTALITY_MAX_RATIO
  )
  const hipSymmetric = ratioWithinBand(
    hipLeftDist,
    hipRightDist,
    FRONTALITY_MIN_RATIO,
    FRONTALITY_MAX_RATIO
  )
  if (!shoulderSymmetric || !hipSymmetric) return null

  // Gate (c) + classification: shoulder:hip width ratio, wide dead zone.
  const shoulderWidth = Math.abs(rightShoulder.x - leftShoulder.x)
  const hipWidth = Math.abs(rightHip.x - leftHip.x)
  if (shoulderWidth <= 0 || hipWidth <= 0) return null

  const ratio = shoulderWidth / hipWidth
  if (ratio < PEAR_MAX_RATIO) return "pear"
  if (ratio > INVERTED_TRIANGLE_MIN_RATIO) return "inverted_triangle"
  // 0.90-1.10 dead zone: NOT a confident "rectangle" — a balanced shoulder:hip
  // ratio is equally consistent with rectangle, hourglass, or apple (all 3
  // need a waist measurement to distinguish, which shoulder/hip landmarks
  // alone cannot provide; see src/agents/outfit/body_type.py's scope note).
  // Honest "not confident" beats an overclaimed guess — fall through to the
  // manual chip/type flow instead.
  return null
}

/**
 * Natural-language chat message for a confirmed shape, guaranteed to parse
 * via `src/agents/outfit/body_type.py` / `src/agents/intent_parser.py`'s
 * SYNONYMS map (both keep the bare shape word, e.g. "pear", "hourglass", and
 * the two-word "inverted triangle", as recognized aliases).
 */
const SHAPE_PHRASE: Record<BodyShapeSlug, string> = {
  pear: "a pear",
  apple: "an apple",
  // "hourglass" has a silent h ("OW-er-glass") -> takes "an", not "a".
  hourglass: "an hourglass",
  rectangle: "a rectangle",
  inverted_triangle: "an inverted triangle",
}

export function bodyShapeMessage(slug: BodyShapeSlug): string {
  return `I have ${SHAPE_PHRASE[slug]} silhouette`
}
