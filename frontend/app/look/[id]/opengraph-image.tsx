/**
 * OG image for shared look pages — /look/[id]
 *
 * Renders a polished static-brand card (plum background, marigold
 * "Smiling Hanger" mark) plus the look's occasion + item count when the
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
          backgroundColor: "#1B1722",
          fontFamily: "sans-serif",
        }}
      >
        {/* Smiling Hanger mark */}
        <svg width="120" height="120" viewBox="0 0 32 32" fill="none">
          <path
            d="M5.5 18.2L16 9.4l10.5 8.8"
            stroke="#E8A33D"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M5.5 18.2c3.2 4.6 17.8 4.6 21 0"
            stroke="#E8A33D"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M16 9.4V8.1c0-1.15.9-2.05 2.05-2.05"
            stroke="#E8A33D"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <div
          style={{
            display: "flex",
            marginTop: 32,
            fontSize: 64,
            fontWeight: 700,
            color: "#ffffff",
            letterSpacing: "-0.02em",
          }}
        >
          Style <span style={{ color: "#E8A33D" }}>Maitri</span>
        </div>

        {occasion ? (
          <div
            style={{
              display: "flex",
              marginTop: 20,
              fontSize: 32,
              color: "#e5e1e8",
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
              color: "#e5e1e8",
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
