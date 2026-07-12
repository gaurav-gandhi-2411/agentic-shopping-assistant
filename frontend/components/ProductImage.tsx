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
      <Image
        src={src}
        alt={alt}
        fill
        sizes={sizes}
        unoptimized
        className={cn("object-cover", className)}
        onLoad={() => setStatus("loaded")}
        onError={() => setStatus("error")}
      />
    </>
  )
}
