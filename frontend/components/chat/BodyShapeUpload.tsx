"use client"

import { useRef, useState } from "react"
import { PersonStanding, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { detectPoseLandmarks } from "@/lib/poseLandmarker"
import { classifyBodyShape, bodyShapeMessage, type BodyShapeSlug } from "@/lib/poseShape"

/** Soft cap mirroring ChatInput.tsx's existing garment-photo upload limit. */
const MAX_IMAGE_BYTES = 15 * 1024 * 1024
const ACCEPTED_IMAGE_TYPES = "image/jpeg,image/png,image/webp"

const SHAPE_OPTIONS: ReadonlyArray<{ slug: BodyShapeSlug; label: string }> = [
  { slug: "pear", label: "Pear" },
  { slug: "apple", label: "Apple" },
  { slug: "hourglass", label: "Hourglass" },
  { slug: "rectangle", label: "Rectangle" },
  { slug: "inverted_triangle", label: "Inverted triangle" },
]

function shapeLabel(slug: BodyShapeSlug): string {
  return SHAPE_OPTIONS.find((opt) => opt.slug === slug)?.label.toLowerCase() ?? slug
}

// idle -> intro (privacy copy + choose/cancel) -> loading (model + detection)
// -> confident (suggestion + confirm/pick-different) | fallback (neutral, all
// 5 shapes as quick buttons). "picking" reuses the fallback panel's shape
// grid after "Pick a different one" from the confident state.
type Stage = "idle" | "intro" | "loading" | "confident" | "picking" | "fallback"

interface Props {
  /** Sends a natural-language message through the existing chat send path —
   *  the SAME mechanism the "What suits my body type?" suggestion chip already
   *  uses (see MessageBubble.tsx's suggestionChips rendering). */
  onSend: (message: string) => void
  disabled?: boolean
}

/**
 * "Photo -> body-shape suggestion" upload affordance.
 *
 * Visually and functionally distinct from ChatInput.tsx's existing garment/
 * inspiration-photo upload (ImagePlus icon -> POST /style/from-image -> CLIP
 * match): here, the user's PHOTO is processed on-device and never uploaded
 * or stored anywhere — decoded and scored entirely client-side via MediaPipe
 * Pose Landmarker (lib/poseLandmarker.ts) and a pure confidence-gated
 * classifier (lib/poseShape.ts). (The MediaPipe library's own WASM/model
 * assets are fetched from Google's public CDN on first use — see
 * lib/poseLandmarker.ts's module docstring for why that's a separate
 * concern from the photo-privacy guarantee.)
 *
 * Never asserts a shape — a strict confidence gate (visibility + frontality
 * + a wide shoulder:hip dead zone, see lib/poseShape.ts) must pass before any
 * suggestion is shown. Any gate failure, any MediaPipe/browser error, or an
 * unusable file all fail SILENTLY into the same neutral fallback panel that
 * hands off to the existing manual chip/type flow — this is an optional,
 * low-stakes feature and must never show a raw error.
 */
export function BodyShapeUpload({ onSend, disabled }: Props) {
  const [stage, setStage] = useState<Stage>("idle")
  const [suggestedSlug, setSuggestedSlug] = useState<BodyShapeSlug | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function reset(): void {
    setStage("idle")
    setSuggestedSlug(null)
  }

  function handleChoosePhotoClick(): void {
    fileInputRef.current?.click()
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = e.target.files?.[0]
    // Reset the input so the same file can be re-selected after a reset.
    e.target.value = ""
    if (!file) return

    // Anything that doesn't look like a processable photo falls straight
    // through to the fallback panel — never a raw error for this optional,
    // low-stakes feature.
    if (!file.type.startsWith("image/") || file.size > MAX_IMAGE_BYTES) {
      setStage("fallback")
      return
    }

    setStage("loading")
    try {
      const landmarks = await detectPoseLandmarks(file)
      const shape = landmarks ? classifyBodyShape(landmarks) : null
      if (shape) {
        setSuggestedSlug(shape)
        setStage("confident")
      } else {
        setStage("fallback")
      }
    } catch {
      // MediaPipe/browser failure (unsupported browser, corrupt image, WASM
      // load failure, ...) — fail silently into the same graceful fallback.
      setStage("fallback")
    }
  }

  function handleConfirmShape(slug: BodyShapeSlug): void {
    onSend(bodyShapeMessage(slug))
    reset()
  }

  return (
    <div className="relative shrink-0">
      {/* Hidden file input — stays mounted across all stages. */}
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_IMAGE_TYPES}
        className="sr-only"
        onChange={handleFileChange}
        aria-label="Upload a photo for a body-shape suggestion"
      />

      {/* Trigger button — distinct icon from the existing garment-photo upload
          (ImagePlus) so the two affordances are never confused. */}
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => setStage(stage === "idle" ? "intro" : "idle")}
        disabled={disabled}
        title="Body shape suggestion (optional)"
        aria-label="Body shape suggestion (optional)"
        className="shrink-0 h-9 px-2.5"
      >
        <PersonStanding className="h-4 w-4" />
      </Button>

      {/* Expanded panel — floats above the composer so it doesn't disturb the
          input row's layout. */}
      {stage !== "idle" && (
        <div
          className={cn(
            "absolute bottom-full right-0 mb-2 w-72 max-w-[85vw] z-10",
            "rounded-lg border bg-background shadow-md px-3 py-2.5 text-xs flex flex-col gap-2"
          )}
        >
          {stage === "intro" && (
            <>
              <p className="text-muted-foreground">
                For a body-shape suggestion: your photo is processed on your device and
                never uploaded.
              </p>
              <div className="flex gap-2">
                <Button
                  type="button"
                  size="sm"
                  onClick={handleChoosePhotoClick}
                  className="h-7 px-2.5 text-xs"
                >
                  Choose photo
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={reset}
                  className="h-7 px-2.5 text-xs"
                >
                  Cancel
                </Button>
              </div>
            </>
          )}

          {stage === "loading" && (
            <p className="text-muted-foreground flex items-center gap-2">
              <span
                className="inline-block h-3 w-3 shrink-0 rounded-full border-2 border-muted-foreground/40 border-t-transparent animate-spin"
                aria-hidden
              />
              Analyzing your photo in your browser — first time can take a moment…
            </p>
          )}

          {stage === "confident" && suggestedSlug && (
            <>
              <p>
                You might have a ~{shapeLabel(suggestedSlug)} silhouette — does that sound
                right?
              </p>
              <div className="flex flex-wrap items-center gap-1.5">
                <Button
                  type="button"
                  size="sm"
                  onClick={() => handleConfirmShape(suggestedSlug)}
                  className="h-7 px-2.5 text-xs"
                >
                  Yes, that&apos;s right
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setStage("picking")}
                  className="h-7 px-2.5 text-xs"
                >
                  Pick a different one
                </Button>
                <button
                  type="button"
                  onClick={reset}
                  aria-label="Dismiss"
                  className="ml-auto rounded p-1 hover:bg-accent"
                >
                  <X className="h-3.5 w-3.5 text-muted-foreground" />
                </button>
              </div>
            </>
          )}

          {(stage === "picking" || stage === "fallback") && (
            <>
              <p className="text-muted-foreground">
                {stage === "fallback"
                  ? "Prefer to just tell me? Tap a shape below or type it."
                  : "A few shapes people mention — tap whichever fits:"}
              </p>
              <div className="flex flex-wrap items-center gap-1.5">
                {SHAPE_OPTIONS.map((opt) => (
                  <button
                    key={opt.slug}
                    type="button"
                    onClick={() => handleConfirmShape(opt.slug)}
                    className="rounded-full border px-3 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                  >
                    {opt.label}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={reset}
                  aria-label="Dismiss"
                  className="rounded p-1 hover:bg-accent"
                >
                  <X className="h-3.5 w-3.5 text-muted-foreground" />
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
