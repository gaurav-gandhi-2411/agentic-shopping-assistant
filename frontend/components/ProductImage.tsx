"use client"

import { useState } from "react"
import Image from "next/image"
import { cn } from "@/lib/utils"

interface Props {
  src: string | null | undefined
  alt: string
  sizes?: string
  className?: string
}

/**
 * Product image with a loading skeleton and an honest error fallback.
 *
 * Store CDN images load slowly or intermittently fail hotlinking — without
 * this, a card rendered as a blank pink rectangle with no signal whether it
 * was loading or dead (defect sweep 2026-07-10, P2-11). While loading: a
 * pulsing skeleton. On error (or no URL): a neutral garment glyph.
 */
export function ProductImage({ src, alt, sizes, className }: Props) {
  const [status, setStatus] = useState<"loading" | "loaded" | "error">(
    src ? "loading" : "error"
  )

  if (!src || status === "error") {
    return (
      <div
        className={cn("w-full h-full flex items-center justify-center bg-muted", className)}
        aria-label={alt}
      >
        <span className="text-3xl select-none opacity-60" aria-hidden>
          👗
        </span>
      </div>
    )
  }

  return (
    <>
      {status === "loading" && (
        <div className="absolute inset-0 animate-pulse bg-muted-foreground/10" aria-hidden />
      )}
      {/* Blur-fill backdrop: a cover-cropped, blurred, zoomed copy of the same
          photo fills the frame behind the real image. Retailer photos are
          hotlinked raw (no resize/crop proxy) and often frame the garment in
          ~25% of the shot off-center — plain object-cover on a single image
          crops straight to blank backdrop. This backdrop replaces the empty
          letterbox with a soft echo of the photo instead (defect: photo crop,
          wave 7). */}
      <Image
        src={src}
        alt=""
        aria-hidden
        fill
        sizes={sizes}
        unoptimized
        className="object-cover scale-110 blur-xl"
      />
      {/* Foreground: the whole, undistorted photo — object-contain so the
          garment is never cropped, layered over the blurred backdrop. */}
      <Image
        src={src}
        alt={alt}
        fill
        sizes={sizes}
        unoptimized
        className={cn(
          "object-contain transition-transform duration-300 ease-out group-hover:scale-105",
          className
        )}
        onLoad={() => setStatus("loaded")}
        onError={() => setStatus("error")}
      />
    </>
  )
}
