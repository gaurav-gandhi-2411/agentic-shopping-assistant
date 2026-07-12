/**
 * Site-wide default OG image — applies to every route without its own
 * opengraph-image (notably /demo and /demo/chat, which shipped with NO OG
 * tags at all: a shared demo link rendered bare in WhatsApp — defect sweep
 * 2026-07-10, P1-8). /look/[id] keeps its dedicated dynamic card.
 *
 * Pure static brand card: cream background, Marigold Knot mark, wordmark.
 */

import { ImageResponse } from "next/og"

export const runtime = "edge"
export const alt = "Style Maitri — your AI stylist"
export const size = { width: 1200, height: 630 }
export const contentType = "image/png"

export default function OpengraphImage() {
  // exempt from design-token audit: this is a Satori/next-og render tree, which
  // runs outside the DOM/CSS runtime and cannot resolve CSS custom properties —
  // every hex literal below (cream/ink/champagne/muted-taupe) is intentional.
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
        {/* Marigold Knot mark (brand/mark.svg) */}
        <svg width="130" height="130" viewBox="0 0 48 48" fill="none">
          <g
            transform="translate(24,24)"
            stroke="#B99A5F"
            strokeWidth="3.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          >
            <path d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z" />
            <path
              d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z"
              transform="rotate(90)"
            />
          </g>
        </svg>

        <div
          style={{
            display: "flex",
            marginTop: 32,
            fontSize: 68,
            fontWeight: 700,
            color: "#3B3230",
            letterSpacing: "-0.02em",
          }}
        >
          Style <span style={{ color: "#B99A5F", marginLeft: 16 }}>Maitri</span>
        </div>

        <div style={{ display: "flex", marginTop: 20, fontSize: 32, color: "#6F6259" }}>
          Your AI stylist for weddings, sangeets &amp; every day
        </div>
      </div>
    ),
    size
  )
}
