/**
 * OG image for shared look pages — /look/[id]
 *
 * When the look and its hero item can be fetched, renders the actual product
 * (image, name, price, occasion) alongside the brand mark — so a shared link
 * previews the real look, not a generic card. Falls back to a polished
 * static-brand card (cream background, Marigold Knot mark) with just the
 * occasion + item count when there's no hero image, and to a fully generic
 * card when the look can't be fetched at all. Never throws: a failed fetch
 * falls back gracefully so the build/route never breaks.
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

// Satori's built-in fallback font has no glyph for ₹ (U+20B9) — it rendered
// as a tofu box in the live share preview (found 2026-07-16 live-proof of
// batch 3). Fetching a Noto Sans subset explicitly covering the glyphs this
// route actually emits (ASCII + ₹ + the ellipsis/middle-dot template chars)
// guarantees Satori always has the glyph, regardless of what product name
// text happens to come back from the catalogue.
const OG_FONT_CHARS =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" +
  " .,'&-:%()/₹…·"

type OgFontFace = { weight: 400 | 700; data: ArrayBuffer }

async function loadOgFonts(): Promise<OgFontFace[]> {
  try {
    const cssRes = await fetch(
      `https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&text=${encodeURIComponent(OG_FONT_CHARS)}`,
      { headers: { "User-Agent": "Mozilla/5.0 (compatible; Satori font fetch)" } }
    )
    const css = await cssRes.text()
    const faces: OgFontFace[] = []
    for (const block of css.split("@font-face").slice(1)) {
      const weightMatch = block.match(/font-weight:\s*(400|700)/)
      const urlMatch = block.match(/src: url\(([^)]+)\) format\('(?:opentype|truetype)'\)/)
      if (!weightMatch || !urlMatch) continue
      const fontRes = await fetch(urlMatch[1])
      if (!fontRes.ok) continue
      faces.push({
        weight: Number(weightMatch[1]) as 400 | 700,
        data: await fontRes.arrayBuffer(),
      })
    }
    return faces
  } catch {
    return []
  }
}

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

// Marigold Knot mark (brand/mark.svg, approved rebrand asset — the
// old "Smiling Hanger" here was a rejected concept that survived in
// OG images only; see defect sweep 2026-07-10 P1-8)
function MarigoldMark({ size: markSize }: { size: number }) {
  return (
    <svg width={markSize} height={markSize} viewBox="0 0 48 48" fill="none">
      <g transform="translate(24,24)" stroke="#B99A5F" strokeWidth="3.8" strokeLinecap="round" strokeLinejoin="round" fill="none">
        <path d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z"/>
        <path d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z" transform="rotate(90)"/>
      </g>
    </svg>
  )
}

export default async function OpengraphImage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const look = await fetchSharedLook(id).catch(() => null)

  const occasion = look?.snapshot?.occasion ?? look?.occasion ?? null
  const items = look?.snapshot?.items ?? []
  const itemCount = items.length
  // Prefer the "seed" (hero) item — the same item the page itself leads with;
  // fall back to the first item when no seed is tagged.
  const heroItem = items.find((it) => it.slot_role === "seed") ?? items[0] ?? null
  const heroName = heroItem?.prod_name ?? heroItem?.display_name ?? null

  const ogFontFaces = await loadOgFonts()
  const fonts = ogFontFaces.length
    ? ogFontFaces.map((f) => ({
        name: "Noto Sans",
        data: f.data,
        style: "normal" as const,
        weight: f.weight,
      }))
    : undefined
  const fontFamily = ogFontFaces.length ? "Noto Sans" : "sans-serif"

  // Generic fallback — no look, or a look with no items/image to show. Keeps
  // the static brand card so the route never renders a broken/empty image.
  // exempt from design-token audit: this is a Satori/next-og render tree, which
  // runs outside the DOM/CSS runtime and cannot resolve CSS custom properties —
  // every hex literal below (cream/ink/champagne/muted-taupe) is intentional.
  if (!heroItem?.image_url) {
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
            fontFamily,
          }}
        >
          <MarigoldMark size={120} />

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
            <div style={{ display: "flex", marginTop: 20, fontSize: 32, color: "#6F6259" }}>
              {formatOccasion(occasion)}
              {itemCount > 0 ? ` · ${itemCount} item${itemCount === 1 ? "" : "s"}` : ""}
            </div>
          ) : (
            <div style={{ display: "flex", marginTop: 20, fontSize: 32, color: "#6F6259" }}>
              Your AI stylist for fashion discovery
            </div>
          )}
        </div>
      ),
      { ...size, fonts }
    )
  }

  // Real-look card: hero product image on the left, brand + occasion + item
  // details on the right — a preview a link recipient can act on at a glance.
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
          backgroundColor: "#FAF6F1",
          fontFamily,
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- next/image
            doesn't render inside next/og's Satori-based ImageResponse; a
            plain <img> is the documented pattern for OG image routes. */}
        <img
          src={heroItem.image_url}
          alt=""
          width={470}
          height={630}
          style={{ objectFit: "cover" }}
        />

        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            padding: "0 56px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center" }}>
            <MarigoldMark size={56} />
            <div
              style={{
                display: "flex",
                marginLeft: 16,
                fontSize: 34,
                fontWeight: 700,
                color: "#3B3230",
                letterSpacing: "-0.02em",
              }}
            >
              Style <span style={{ color: "#B99A5F", marginLeft: 8 }}>Maitri</span>
            </div>
          </div>

          {occasion && (
            <div
              style={{
                display: "flex",
                marginTop: 28,
                fontSize: 24,
                fontWeight: 600,
                color: "#B99A5F",
                textTransform: "uppercase",
                letterSpacing: "0.04em",
              }}
            >
              {formatOccasion(occasion)}
            </div>
          )}

          {heroName && (
            <div
              style={{
                display: "flex",
                marginTop: 12,
                fontSize: 44,
                fontWeight: 700,
                color: "#3B3230",
                lineHeight: 1.15,
              }}
            >
              {heroName.length > 60 ? `${heroName.slice(0, 57)}…` : heroName}
            </div>
          )}

          <div style={{ display: "flex", marginTop: 20, fontSize: 30, color: "#6F6259" }}>
            {heroItem.price_inr != null ? `₹${heroItem.price_inr.toLocaleString("en-IN")}` : ""}
            {heroItem.price_inr != null && itemCount > 1 ? " · " : ""}
            {itemCount > 1 ? `${itemCount} item${itemCount === 1 ? "" : "s"} in this look` : ""}
          </div>
        </div>
      </div>
    ),
    { ...size, fonts }
  )
}
