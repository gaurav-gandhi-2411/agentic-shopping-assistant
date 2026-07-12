/**
 * Browser-only MediaPipe Pose Landmarker loader.
 *
 * The WASM runtime and .task model file are fetched from MediaPipe's
 * official, documented CDN pattern (jsDelivr for the WASM fileset, Google
 * Cloud Storage for the model) rather than self-hosted in `public/` — see
 * the agent report for why: the "your photo never leaves your browser"
 * privacy guarantee is about the USER'S PHOTO (processed in-memory,
 * in-browser, never uploaded — see {@link detectPoseLandmarks}), not about
 * which server the LIBRARY's own model/WASM assets load from. Using
 * Google's own CDN for those doesn't touch that guarantee and avoids
 * committing ~27 MB of binaries to this repo.
 *
 * Both URLs are pinned to the exact `@mediapipe/tasks-vision` version this
 * repo has installed (see package.json) rather than "@latest" / "latest",
 * deliberately — an upstream release bump on either CDN path could otherwise
 * silently break the JS API <-> WASM/model compatibility this code assumes.
 *
 * The PoseLandmarker instance is created once (module-level singleton
 * promise) and reused across calls — MediaPipe's own guidance is to avoid
 * re-creating the WASM runtime per detection.
 */
import type { NormalizedLandmark, PoseLandmarker } from "@mediapipe/tasks-vision"

// Pinned to the installed @mediapipe/tasks-vision version (package.json) —
// keep in sync if that dependency is ever bumped.
const MEDIAPIPE_VERSION = "0.10.35"
const WASM_BASE_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`
const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

let landmarkerPromise: Promise<PoseLandmarker> | null = null

/**
 * Lazily create (and cache) the single PoseLandmarker instance.
 *
 * `@mediapipe/tasks-vision` is dynamically imported here (not statically at
 * module scope) so its ~130 KB JS bundle is only ever fetched the first time
 * a user actually opens the body-shape upload panel, instead of inflating
 * every chat page's initial bundle (see the agent report for the measured
 * before/after First Load JS delta).
 *
 * Forces the CPU delegate deliberately: this is a one-shot static-image
 * classification (not real-time video), so CPU is plenty fast and keeps the
 * fetched WASM path simpler (skips the GPU-delegate-specific bundle).
 */
function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const { FilesetResolver, PoseLandmarker: PoseLandmarkerCtor } = await import(
        "@mediapipe/tasks-vision"
      )
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE_PATH)
      return PoseLandmarkerCtor.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_ASSET_PATH,
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        numPoses: 1,
      })
    })()
    // Reset the cache on failure so a later retry can attempt a fresh load
    // instead of permanently caching a rejected promise.
    landmarkerPromise.catch(() => {
      landmarkerPromise = null
    })
  }
  return landmarkerPromise
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error("Could not decode image"))
    img.src = src
  })
}

/**
 * Run pose detection on a single image File, entirely client-side.
 *
 * Returns the first detected pose's 33 landmarks, or `null` when no pose was
 * detected in the image. Throws only for genuine load/decode/model failures
 * (corrupt file, unsupported browser, WASM load failure) — callers must
 * catch and treat any throw the same as a `null` result: fall through to the
 * manual chip/type flow, never surface a raw error for this optional,
 * low-stakes feature.
 */
export async function detectPoseLandmarks(file: File): Promise<NormalizedLandmark[] | null> {
  const landmarker = await getPoseLandmarker()
  const objectUrl = URL.createObjectURL(file)
  try {
    const image = await loadImageElement(objectUrl)
    const result = landmarker.detect(image)
    return result.landmarks?.[0] ?? null
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}
