/**
 * Read-only shared look page — /look/[id]
 *
 * Fetches GET /looks/{id} from the default backend (the looks table is shared
 * across all brand services so any backend can serve it). Renders the saved
 * snapshot using a lightweight read-only layout derived from OutfitBoard.
 *
 * Accessible by anonymous visitors; no demo session required.
 */

import type { Metadata } from "next"
import Link from "next/link"
import Image from "next/image"
import { ExternalLink, ShoppingBag, Sparkles } from "lucide-react"
import { Logo } from "@/components/Logo"
import { getStoreDisplayName } from "@/lib/stores"
import { formatSlotLabel } from "@/lib/displayFormat"
import type { ItemLink, LookSnapshot, SharedLook } from "@/lib/api/types"

// Default backend — the looks table is shared, any service can answer.
const DEFAULT_BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  process.env.NEXT_PUBLIC_SNITCH_BACKEND_URL ??
  "http://localhost:8000"

// Shopify brand ids — use cart URL if available.
const SHOPIFY_BRANDS = new Set(["snitch", "powerlook", "fashor", "virgio"])

// ---------------------------------------------------------------------------
// Data fetching — runs on the server
// ---------------------------------------------------------------------------

async function fetchSharedLook(id: string): Promise<SharedLook | null> {
  try {
    const res = await fetch(`${DEFAULT_BACKEND}/looks/${encodeURIComponent(id)}`, {
      // ISR: revalidate every 60 s; the look data itself is immutable after save.
      next: { revalidate: 60 },
    })
    if (res.status === 404) return null
    if (!res.ok) return null
    return (await res.json()) as SharedLook
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// Helper: format occasion slug for display
// ---------------------------------------------------------------------------

function formatOccasion(slug: string): string {
  return slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

// ---------------------------------------------------------------------------
// Page metadata — resilient to fetch failure (falls back to a static title)
// ---------------------------------------------------------------------------

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>
}): Promise<Metadata> {
  const { id } = await params
  const look = await fetchSharedLook(id)

  if (!look) {
    const title = "Style Maitri look"
    const description = "A saved outfit look from Style Maitri — your AI stylist for fashion discovery."
    return {
      title,
      description,
      // Explicit openGraph/twitter overrides — without these, Next's shallow
      // per-key metadata merge falls back to the ENTIRE parent layout object
      // (generic "Style Maitri" branding) instead of this page's own title/
      // description. See the populated-look branch below for the real case.
      openGraph: { title, description },
      twitter: { title, description },
    }
  }

  const occasion = look.snapshot?.occasion ?? look.occasion
  const itemCount = look.snapshot?.items?.length ?? 0
  const titlePrefix = occasion ? formatOccasion(occasion) : "Saved Look"
  const description = `A ${itemCount}-item ${occasion ? formatOccasion(occasion).toLowerCase() + " " : ""}look styled with Style Maitri.`
  const title = `${titlePrefix} — Style Maitri look`

  return {
    title,
    description,
    openGraph: { title, description },
    twitter: { title, description },
  }
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default async function SharedLookPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  const look = await fetchSharedLook(id)

  if (!look) {
    return (
      <main className="flex flex-col items-center justify-center min-h-screen p-8 text-center">
        <p className="text-4xl mb-4" aria-hidden>
          🪄
        </p>
        <h1 className="text-xl font-bold mb-2">Look not found</h1>
        <p className="text-sm text-muted-foreground mb-6">
          This look may have been removed or the link is incorrect.
        </p>
        <Link
          href="/demo"
          className="text-sm text-primary underline underline-offset-4 hover:text-primary/80 transition-colors"
        >
          Build your own look →
        </Link>
      </main>
    )
  }

  const snapshot: LookSnapshot = look.snapshot ?? {}
  const items = snapshot.items ?? []
  const rationale = snapshot.rationale ?? null
  const cartUrl = snapshot.cart_url ?? null
  const itemLinks: ItemLink[] = snapshot.item_links ?? []
  const brand = snapshot.brand ?? look.brand ?? null
  const budgetTotalInr = snapshot.budget_total_inr ?? look.look_total_inr ?? null
  const occasion = snapshot.occasion ?? look.occasion ?? null
  const lookGender = snapshot.look_gender ?? look.look_gender ?? null
  const variantLabel = snapshot.variant_label ?? null
  const isShopify = brand != null && SHOPIFY_BRANDS.has(brand)

  // Build item-level buy URL map for non-Shopify brands.
  const itemLinkMap = new Map<string, string>(itemLinks.map((l) => [l.article_id, l.buy_url]))

  const seed = items.find((it) => it.slot_role === "seed")
  const complements = items.filter((it) => it.slot_role === "complement")
  const allItems = seed ? [seed, ...complements] : complements

  return (
    <main className="min-h-screen bg-background flex items-center justify-center">
      <div className="w-full max-w-lg lg:max-w-3xl mx-auto px-4 py-8 space-y-6">
        {/* Brand */}
        <Logo className="mb-2" />

        {/* Header */}
        <div className="space-y-1">
          <h1 className="text-xl font-bold tracking-tight">Saved Look</h1>
          <div className="flex flex-wrap items-center gap-2">
            {occasion && (
              <span className="inline-block rounded-full bg-primary/10 text-primary text-xs font-semibold px-3 py-1">
                {formatOccasion(occasion)}
              </span>
            )}
            {lookGender && (
              <span className="inline-block rounded-full bg-secondary text-secondary-foreground text-xs px-2 py-0.5 capitalize">
                {lookGender}
              </span>
            )}
            {/* "Base" is the composer's internal tag for the primary look —
                meaningless to a share-link visitor; real variant labels
                ("Colour story", "Ethnic") still render. */}
            {variantLabel && variantLabel !== "Base" && (
              <span className="inline-block rounded-full border text-xs px-2 py-0.5 text-muted-foreground">
                {variantLabel}
              </span>
            )}
          </div>
        </div>

        {/* Item grid */}
        {allItems.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {allItems.map((item) => {
              if (!item.article_id) return null
              const isSeed = item.slot_role === "seed"
              const slotLabel = isSeed
                ? "Hero"
                : item.outfit_slot
                  ? formatSlotLabel(item.outfit_slot)
                  : "Item"
              // Note: item.pdp_handle is a bare handle, not a resolvable URL — never
              // use it as an href fallback. If no real URL is available, render
              // the item card without a link.
              const buyUrl = itemLinkMap.get(item.article_id) ?? item.buy_url ?? null
              // Per-item store label — mirrors ItemCard.tsx's "Buy at {store}" convention.
              // Unified cross-store mode mixes stores within one look, so the buy
              // button must read each item's own store, not the page-level brand
              // (which is "unified" — a composer tag, not a real store name).
              // Legacy single-brand saves have no item.store; fall back to the
              // page-level brand there (excluding the "unified" sentinel).
              const itemStoreDisplay =
                item.store_display ??
                getStoreDisplayName(item.store) ??
                (brand && brand !== "unified" ? getStoreDisplayName(brand) : null)
              const cardClassName =
                "rounded-lg border bg-card overflow-hidden hover:shadow-md transition-shadow"

              // The card itself is a plain div, not a link — a visible "Buy at
              // {store}" button below carries the buy action instead (matches
              // ItemCard.tsx's convention; also avoids nesting an <a> inside
              // an <a> now that the card has its own buy anchor).
              return (
                <div key={item.article_id} className={cardClassName}>
                  <div className="relative aspect-[4/5] bg-muted">
                    {item.image_url ? (
                      <Image
                        src={item.image_url}
                        alt={item.prod_name ?? item.display_name ?? "Item"}
                        fill
                        sizes="(max-width: 640px) 50vw, 33vw"
                        unoptimized
                        className="object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-3xl select-none">
                        👗
                      </div>
                    )}
                    <span className="absolute top-1.5 left-1.5 rounded-sm bg-background/90 text-foreground text-[10px] font-semibold px-1.5 py-0.5">
                      {slotLabel}
                    </span>
                  </div>
                  <div className="p-1.5 space-y-1">
                    <p className="text-xs font-medium leading-tight line-clamp-2">
                      {item.prod_name ?? item.display_name ?? ""}
                    </p>
                    {item.price_inr != null && (
                      <p className="text-xs text-muted-foreground">
                        ₹{item.price_inr.toLocaleString("en-IN")}
                      </p>
                    )}
                    {buyUrl && (
                      <a
                        href={buyUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex w-full items-center justify-center min-h-11 text-center text-[11px] font-medium px-2 py-1.5 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                      >
                        Buy at {itemStoreDisplay ?? "Shop"}
                      </a>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No items in this look.</p>
        )}

        {/* Stylist's note */}
        {rationale && (
          <div className="flex gap-2 rounded-lg bg-muted/60 px-3 py-2.5">
            <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary/70" />
            <p className="text-xs italic text-muted-foreground leading-relaxed">
              <span className="not-italic font-semibold text-foreground/70">
                Stylist&rsquo;s note&nbsp;
              </span>
              {rationale}
            </p>
          </div>
        )}

        {/* Buy action — enabled for anonymous visitors (viral → conversion loop) */}
        {budgetTotalInr != null && (
          <>
            {isShopify && cartUrl ? (
              <a
                href={cartUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold py-2.5 hover:bg-primary/90 transition-colors"
              >
                <ShoppingBag className="h-4 w-4" />
                Add the look to cart — ₹{Math.round(budgetTotalInr).toLocaleString("en-IN")}
              </a>
            ) : itemLinks.length > 0 ? (
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">Buy each item:</p>
                {itemLinks.map((link) => (
                  <a
                    key={link.article_id}
                    href={link.buy_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between min-h-11 rounded-lg border px-3 py-2 text-xs hover:bg-accent hover:text-accent-foreground transition-colors"
                  >
                    <span className="truncate mr-2">{link.name}</span>
                    <ExternalLink className="h-3 w-3 shrink-0" />
                  </a>
                ))}
              </div>
            ) : null}
          </>
        )}

        {/* Footer — link back to the full styler */}
        <div className="pt-2 border-t text-center">
          <p className="text-xs text-muted-foreground">
            {brand ? (
              <>
                Styled with{" "}
                <span className="font-semibold capitalize">
                  {brand === "unified" ? "Style Maitri" : brand}
                </span>
                {" · "}
              </>
            ) : null}
            <Link
              href="/demo"
              className="text-primary underline underline-offset-4 hover:text-primary/80 transition-colors"
            >
              Open the full styler
            </Link>
          </p>
        </div>
      </div>
    </main>
  )
}
