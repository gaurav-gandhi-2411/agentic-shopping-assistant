/**
 * OG image for shared look pages — /look/[id]
 *
 * Renders a polished static-brand card (cream background, Marigold Knot
 * mark) plus the look's occasion + item count when the
 * look can be fetched cheaply. Never throws: a failed fetch falls back to
 * the static brand card so the build/route never breaks.
 */

import { ImageResponse } from "next/og"
import type { SharedLook } from "@/lib/api/types"

export const runtime = "edge"
export const alt = "Style Maitri look"
export const size = { width: 1200, height: 630 }
export const contentType = "image/png"

const DEFAULT_BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  process.env.NEXT_PUBLIC_SNITCH_BACKEND_URL ??
  "http://localhost:8000"

function formatOccasion(slug: string): string {
  return slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

async function fetchSharedLook(id: string): Promise<SharedLook | null> {
  try {
    const res = await fetch(`${DEFAULT_BACKEND}/looks/${encodeURIComponent(id)}`, {
      next: { revalidate: 60 },
    })
    if (!res.ok) return null
    return (await res.json()) as SharedLook
  } catch {
    return null
  }
}

export default async function OpengraphImage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const look = await fetchSharedLook(id).catch(() => null)

  const occasion = look?.snapshot?.occasion ?? look?.occasion ?? null
  const itemCount = look?.snapshot?.items?.length ?? 0

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#FAF6F1",
          fontFamily: "sans-serif",
        }}
      >
        {/* Marigold Knot mark (brand/mark.svg, approved rebrand asset — the
            old "Smiling Hanger" here was a rejected concept that survived in
            OG images only; see defect sweep 2026-07-10 P1-8) */}
        <svg width="120" height="120" viewBox="0 0 48 48" fill="none">
          <g transform="translate(24,24)" stroke="#B99A5F" strokeWidth="3.8" strokeLinecap="round" strokeLinejoin="round" fill="none">
            <path d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z"/>
            <path d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z" transform="rotate(90)"/>
          </g>
        </svg>

        <div
          style={{
            display: "flex",
            marginTop: 32,
            fontSize: 64,
            fontWeight: 700,
            color: "#3B3230",
            letterSpacing: "-0.02em",
          }}
        >
          Style <span style={{ color: "#B99A5F" }}>Maitri</span>
        </div>

        {occasion ? (
          <div
            style={{
              display: "flex",
              marginTop: 20,
              fontSize: 32,
              color: "#6F6259",
            }}
          >
            {formatOccasion(occasion)}
            {itemCount > 0 ? ` · ${itemCount} item${itemCount === 1 ? "" : "s"}` : ""}
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              marginTop: 20,
              fontSize: 32,
              color: "#6F6259",
            }}
          >
            Your AI stylist for fashion discovery
          </div>
        )}
      </div>
    ),
    { ...size }
  )
}
