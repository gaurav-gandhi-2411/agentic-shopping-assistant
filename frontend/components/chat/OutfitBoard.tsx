"use client"

import { useEffect, useRef, useState } from "react"
import Image from "next/image"
import { Bookmark, Check, Copy, ExternalLink, ShoppingBag, Sparkles, Store } from "lucide-react"
import { useBrandConfig } from "@/hooks/useBrandConfig"
import { getStoreDisplayName } from "@/lib/stores"
import { cn } from "@/lib/utils"
import type {
  ItemLink,
  ItemSummary,
  LookSnapshot,
  OutfitVariant,
  SaveLookResponse,
  SuppressedSlot,
} from "@/lib/api/types"

// Marigold/amber brand accent for the "your item" owned-seed card — distinct from
// the primary theme so the user's own garment always reads as visually different
// from a catalogue product (no buy link, no price, no store badge). Used as a
// literal Tailwind arbitrary-value class (e.g. `border-[#E8A33D]`) below, not as
// a JS variable, so Tailwind's static class scanner picks it up at build time.

// ---------------------------------------------------------------------------
// Shopify store slugs — these stores support Shopify cart permalinks.
// In unified cross-store mode outfits may mix items from multiple stores, so
// we check per-item store rather than a single board-level brand.
// ---------------------------------------------------------------------------
const SHOPIFY_STORE_SLUGS = new Set(["snitch", "powerlook", "fashor", "virgio"])

const VARIANT_LABEL_MAP = {
  "Base": "Style 1",
  "Colour story": "Colour Palette",
  "Dressier": "Dressed Up",
  "Lighter": "Casual Edit",
} as const

function humanLabel(label: string): string {
  return (VARIANT_LABEL_MAP as Record<string, string>)[label] ?? label
}

// ---------------------------------------------------------------------------
// Default backend URL (used for look saves and reads — shared table).
// Falls back to demo session URL if available.
// ---------------------------------------------------------------------------
function getBackendUrl(): string {
  if (typeof window !== "undefined") {
    const demoUrl = sessionStorage.getItem("demo_backend_url")
    if (demoUrl) return demoUrl
  }
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"
}

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

interface OutfitBoardProps {
  items: ItemSummary[]
  lookId: string | null | undefined
  occasion: string | null | undefined
  lookGender: string | null | undefined
  budgetTotalInr: number | null | undefined
  outfitRationale?: string | null
  outfitVariants?: OutfitVariant[] | null
  /** Top-level cart URL for non-variant flow (Shopify brands). */
  cartUrl?: string | null
  /** Top-level per-item links for non-variant flow (non-Shopify). */
  itemLinks?: ItemLink[] | null
  /** Top-level suppressed slots for non-variant flow; variant-level takes precedence when present. */
  suppressedSlots?: SuppressedSlot[] | null
  /** "partner" marks this board as a partner-anchored look; absent/"primary" is the user's own look. */
  lookRole?: "primary" | "partner" | null
  /** Board heading override for partner looks — falls back to "Your partner's look" when absent. */
  lookTitle?: string | null
  /** One-line explanation of how a partner look was coordinated with the primary look. */
  coordinatedWith?: string | null
  /**
   * Local object URL for the user's own uploaded photo (set by sendImage() on the
   * assistant message). Rendered on the owned-seed card in place of the catalogue
   * image; falls back to the item's image_url when absent (e.g. restored sessions).
   */
  anchorImageUrl?: string | null
  sessionId: string
  anchorItemId: string
  anchorCategory: string
  brand: string | undefined
  onSend?: (text: string) => void
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format occasion slug for display — "sangeet_look" → "Sangeet Look". */
function formatOccasion(slug: string): string {
  return slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Format a suppressed slot name for display — "footwear" → "Footwear". */
function formatSlotLabel(slug: string): string {
  return slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Post a flywheel event — fire-and-forget, never throw. */
async function postEvent(backendUrl: string, token: string, payload: object): Promise<void> {
  try {
    await fetch(`${backendUrl}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload),
    })
  } catch {
    // Telemetry is best-effort; swallow errors silently.
  }
}

/** Retrieve demo session credentials from sessionStorage (browser-only). */
function getDemoSession(): { token: string | null; backendUrl: string } {
  if (typeof window === "undefined") {
    return { token: null, backendUrl: "http://localhost:8000" }
  }
  return {
    token: sessionStorage.getItem("demo_session_token"),
    backendUrl:
      sessionStorage.getItem("demo_backend_url") ??
      process.env.NEXT_PUBLIC_BACKEND_URL ??
      "http://localhost:8000",
  }
}

/** Persist a saved look id to localStorage so the user can revisit. */
function persistSavedLookId(id: string): void {
  try {
    const raw = localStorage.getItem("asa_saved_looks")
    const ids: string[] = raw ? (JSON.parse(raw) as string[]) : []
    if (!ids.includes(id)) {
      ids.unshift(id)
      localStorage.setItem("asa_saved_looks", JSON.stringify(ids.slice(0, 50)))
    }
  } catch {
    // localStorage unavailable or full; best-effort.
  }
}

// ---------------------------------------------------------------------------
// OutfitBoard
// ---------------------------------------------------------------------------

export function OutfitBoard({
  items,
  lookId,
  occasion,
  lookGender,
  budgetTotalInr,
  outfitRationale,
  outfitVariants,
  cartUrl: topLevelCartUrl,
  itemLinks: topLevelItemLinks,
  suppressedSlots: topLevelSuppressedSlots,
  lookRole,
  lookTitle,
  coordinatedWith,
  anchorImageUrl,
  sessionId,
  anchorItemId,
  anchorCategory,
  brand,
  onSend,
}: OutfitBoardProps) {
  const { data: brandConfig } = useBrandConfig()
  const loggedRef = useRef(false)

  // Determine the active variant — default to the first one (should be "Base").
  // When variants are absent or singular we fall back to the raw items prop.
  const hasMultipleVariants = outfitVariants != null && outfitVariants.length >= 2
  const [activeVariantId, setActiveVariantId] = useState<string | null>(
    hasMultipleVariants ? outfitVariants![0].variant_id : null,
  )

  // Save / share state.
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle")
  const [savedLookId, setSavedLookId] = useState<string | null>(null)
  const [shareUrl, setShareUrl] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Cross-store "open all items" inline panel — most browsers block every
  // window.open() after the first one within a single click handler, so a
  // forEach loop of window.open() silently drops all but one tab in real
  // browsers (Playwright's automation flags mask this). Instead we toggle an
  // inline panel of plain <a target="_blank"> links, each clicked individually
  // by the user — a real click per link is never popup-blocked.
  const [showOpenAllPanel, setShowOpenAllPanel] = useState(false)

  // Derive the displayed items, rationale, budget, and buy links from the active variant (if any).
  const activeVariant =
    hasMultipleVariants && activeVariantId != null
      ? (outfitVariants!.find((v) => v.variant_id === activeVariantId) ?? outfitVariants![0])
      : null

  const displayItems = activeVariant ? activeVariant.items : items
  const displayRationale = activeVariant ? activeVariant.rationale : (outfitRationale ?? null)
  const displayBudget = activeVariant ? activeVariant.budget_total_inr : budgetTotalInr

  // Buy link resolution: prefer variant-level, fall back to top-level message fields.
  const activeCartUrl = activeVariant?.cart_url ?? topLevelCartUrl ?? null
  const activeItemLinks = activeVariant?.item_links ?? topLevelItemLinks ?? null
  const activeSuppressedSlots = activeVariant?.suppressed_slots ?? topLevelSuppressedSlots ?? null

  // Partner-look presentation — non-partner boards render unchanged.
  const isPartnerLook = lookRole === "partner"

  // Cross-store Shopify check: only enable cart permalink when a single-store Shopify
  // cart_url is present.  In mixed-store outfits (unified mode) we degrade gracefully
  // to the open-all / per-item deep-link path regardless of the top-level brand prop.
  // Legacy per-brand mode still works: brand="snitch" + activeCartUrl → Shopify path.
  const isShopify =
    activeCartUrl != null &&
    (brand != null ? SHOPIFY_STORE_SLUGS.has(brand) : false)

  // ---------------------------------------------------------------------------
  // Slot derivation — self-consistent regardless of backend slot_role tagging.
  // Prefer explicit slot_role tags. If NONE of the items carry a slot_role
  // (a known backend gap — see spec.md B4b) fall back to treating the first
  // item as the hero and the rest as complements, so every returned item
  // still renders as a slot card instead of being silently dropped.
  // ---------------------------------------------------------------------------
  const hasAnySlotRole = displayItems.some((it) => it.slot_role != null)
  const seed = hasAnySlotRole
    ? displayItems.find((it) => it.slot_role === "seed")
    : displayItems[0]
  const complements = hasAnySlotRole
    ? displayItems.filter((it) => it.slot_role === "complement")
    : displayItems.slice(1)
  const allOutfitItems = seed ? [seed, ...complements] : complements

  // Per-item link map — resolves each rendered card's buy URL. Built once so
  // "Open all items" and the individual slot cards always agree.
  const itemLinkMap = new Map<string, string>(
    (activeItemLinks ?? []).map((l) => [l.article_id, l.buy_url]),
  )

  /**
   * Resolve a single item's buy URL: item_links → server pdp_url → legacy pdp_handle + template.
   * Owned items (the user's own uploaded garment) never have a buy link, regardless of
   * what the backend sends — this is a hard UI guarantee, not just a backend contract.
   */
  function resolveItemUrl(item: ItemSummary): string | null {
    if (item.is_owned) return null
    return (
      itemLinkMap.get(item.article_id) ??
      item.pdp_url ??
      (item.pdp_handle && brandConfig?.pdp_url_template
        ? brandConfig.pdp_url_template.replace("{handle}", item.pdp_handle)
        : null)
    )
  }

  // Owned items (the user's own garment) are never purchasable — exclude them from
  // the price total, "open all items" fan-out, and the flywheel item counts below.
  const priceableItems = allOutfitItems.filter((it) => !it.is_owned)

  // The displayed total MUST match the rendered (purchasable) cards. Sum the
  // rendered items' price_inr and prefer it; only fall back to the server-supplied
  // budget when a rendered item is missing a price (no authoritative sum
  // can be computed) — see spec.md B4b.
  const renderedPriceSum =
    priceableItems.length > 0 && priceableItems.every((it) => it.price_inr != null)
      ? priceableItems.reduce((sum, it) => sum + (it.price_inr as number), 0)
      : null
  const displayTotal = renderedPriceSum ?? displayBudget

  // Log look_shown once on mount — uses the base items/budget from props
  // (not variant-derived), as the look exposure event is for the whole look.
  useEffect(() => {
    if (loggedRef.current || !lookId) return
    loggedRef.current = true
    const { token, backendUrl } = getDemoSession()
    if (token) {
      const baseComplements = items.filter((it) => it.slot_role === "complement")
      postEvent(backendUrl, token, {
        event_type: "look_shown",
        session_id: sessionId,
        look_id: lookId,
        anchor_item_id: anchorItemId,
        anchor_category: anchorCategory,
        occasion: occasion ?? null,
        brand: brand ?? null,
        look_total_inr: budgetTotalInr ? Math.round(budgetTotalInr) : null,
        filled_slots: baseComplements.map((c) => ({
          slot: c.outfit_slot,
          item_id: c.article_id,
          brand: brand,
          price_inr: c.price_inr,
        })),
      })
    }
  }, [lookId]) // eslint-disable-line react-hooks/exhaustive-deps

  function handleVariantSwitch(variant: OutfitVariant) {
    if (variant.variant_id === activeVariantId) return
    setActiveVariantId(variant.variant_id)

    // Engagement log — only fires on user-initiated switches (not initial render).
    const { token, backendUrl } = getDemoSession()
    if (token && lookId) {
      postEvent(backendUrl, token, {
        event_type: "variant_selected",
        session_id: sessionId,
        look_id: lookId,
        variant_id: variant.variant_id,
        variant_label: variant.label,
        occasion: occasion ?? null,
        brand: brand ?? null,
      })
    }
  }

  /** Fire add_the_look flywheel event (shared by both Shopify and non-Shopify paths). */
  function fireAddTheLookEvent() {
    if (!lookId) return
    const { token, backendUrl } = getDemoSession()
    if (token) {
      postEvent(backendUrl, token, {
        event_type: "add_the_look",
        session_id: sessionId,
        look_id: lookId,
        anchor_item_id: anchorItemId,
        anchor_category: anchorCategory,
        occasion: occasion ?? null,
        brand: brand ?? null,
        look_total_inr: displayTotal ? Math.round(displayTotal) : null,
        filled_slots: complements.map((c) => ({
          slot: c.outfit_slot,
          item_id: c.article_id,
          brand: brand,
          price_inr: c.price_inr,
        })),
      })
    }
  }

  /** Shopify path: open the cart permalink in a new tab. */
  function handleShopifyCart() {
    fireAddTheLookEvent()
    if (activeCartUrl) {
      window.open(activeCartUrl, "_blank", "noopener,noreferrer")
    }
  }

  /**
   * Non-Shopify / cross-store path: toggle the inline "open all items" panel
   * listing every RENDERED, purchasable card's deep-link as a plain anchor.
   * We deliberately do NOT call window.open() in a loop here — browsers only
   * allow one popup per user gesture, so a forEach of window.open() drops all
   * but the first tab in real Chrome/Edge/Safari. The panel reuses
   * `priceableItems` / `resolveItemUrl` (not the raw server item_links, and
   * never the owned seed) so the links always match exactly what the user
   * sees on the board and can actually buy.
   */
  function handleOpenAllItems() {
    fireAddTheLookEvent()
    setShowOpenAllPanel((prev) => !prev)
  }

  /** Build a self-contained snapshot of the currently active variant/look. */
  function buildSnapshot(): LookSnapshot {
    const snapshotItems = displayItems.map((item) => {
      const buyUrl = resolveItemUrl(item)
      return {
        article_id: item.article_id,
        display_name: item.display_name ?? item.prod_name,
        prod_name: item.prod_name,
        colour: item.colour,
        product_type: item.product_type,
        outfit_slot: item.outfit_slot ?? null,
        slot_role: item.slot_role ?? null,
        image_url: item.image_url ?? null,
        price_inr: item.price_inr ?? null,
        pdp_handle: item.pdp_handle ?? null,
        buy_url: buyUrl ?? null,
      }
    })

    return {
      items: snapshotItems,
      rationale: displayRationale ?? null,
      cart_url: activeCartUrl ?? null,
      item_links: activeItemLinks ?? null,
      variant_label: activeVariant?.label ?? null,
      occasion: occasion ?? null,
      look_gender: lookGender ?? null,
      budget_total_inr: displayTotal ?? null,
      brand: brand ?? null,
    }
  }

  async function handleSaveLook() {
    if (saveState === "saving" || saveState === "saved") return
    setSaveState("saving")
    const { token, backendUrl } = getDemoSession()
    try {
      const snapshot = buildSnapshot()
      const body = {
        session_id: sessionId,
        brand: brand ?? "unified",
        look_id: lookId ?? null,
        occasion: occasion ?? null,
        look_gender: lookGender ?? null,
        anchor_item_id: anchorItemId,
        look_total_inr: displayTotal ? Math.round(displayTotal) : null,
        snapshot,
      }
      const res = await fetch(`${backendUrl}/looks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`POST /looks returned ${res.status}`)
      const data = (await res.json()) as SaveLookResponse
      persistSavedLookId(data.id)
      setSavedLookId(data.id)
      const url = `${window.location.origin}${data.share_path}`
      setShareUrl(url)
      setSaveState("saved")

      // Fire save_look flywheel event.
      if (token) {
        postEvent(backendUrl, token, {
          event_type: "save_look",
          session_id: sessionId,
          look_id: lookId ?? data.id,
          anchor_item_id: anchorItemId,
          anchor_category: anchorCategory,
          occasion: occasion ?? null,
          brand: brand ?? null,
          look_total_inr: displayTotal ? Math.round(displayTotal) : null,
        })
      }
    } catch (err) {
      console.error("save look failed", err)
      setSaveState("error")
      // Reset error state after 3 s so the user can retry.
      setTimeout(() => setSaveState("idle"), 3000)
    }
  }

  async function handleCopyShareUrl() {
    if (!shareUrl) return
    try {
      await navigator.clipboard.writeText(shareUrl)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard API blocked; fall through silently.
    }
  }

  if (allOutfitItems.length === 0) return null

  const filledSlotNames = complements
    .map((c) => c.outfit_slot)
    .filter(Boolean) as string[]

  return (
    <div className="w-full rounded-xl border bg-card p-4 space-y-3">
      {/* Partner-look heading — board-level badge + title + coordination note.
          Rendered only when the backend marks this board look_role === "partner";
          primary boards fall straight through to the occasion/gender header below. */}
      {isPartnerLook && (
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-full bg-[#E8A33D]/15 text-[#E8A33D] text-[10px] font-semibold px-2 py-0.5">
              Partner look
            </span>
            <h3 className="text-sm font-semibold text-foreground">
              {lookTitle || "Your partner's look"}
            </h3>
          </div>
          {coordinatedWith && (
            <p className="text-xs text-muted-foreground">{coordinatedWith}</p>
          )}
        </div>
      )}

      {/* Header — occasion + gender badges */}
      <div className="flex items-center gap-2">
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
      </div>

      {/* Variant switcher — shown only when ≥2 variants are present */}
      {hasMultipleVariants && (
        <div className="flex flex-wrap gap-1.5">
          {outfitVariants!.map((v) => (
            <button
              key={v.variant_id}
              onClick={() => handleVariantSwitch(v)}
              className={
                v.variant_id === activeVariantId
                  ? "rounded-full text-xs font-semibold px-3 py-1 bg-primary text-primary-foreground transition-colors"
                  : "rounded-full text-xs px-3 py-1 border text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
              }
            >
              {humanLabel(v.label)}
            </button>
          ))}
        </div>
      )}

      {/* Slot grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {allOutfitItems.map((item) => (
          <SlotCard
            key={item.article_id}
            item={item}
            pdpUrlTemplate={brandConfig?.pdp_url_template}
            isSeed={item.article_id === seed?.article_id}
            overrideBuyUrl={resolveItemUrl(item)}
            anchorImageUrl={item.article_id === seed?.article_id ? anchorImageUrl : undefined}
            onSend={onSend}
          />
        ))}
      </div>

      {/* Suppressed-slot notes — honest, unobtrusive disclosure of slots the
          stylist intentionally left empty rather than filling with a wrong-gender
          or off-vocabulary item. No error styling; silently absent when empty. */}
      {activeSuppressedSlots && activeSuppressedSlots.length > 0 && (
        <div className="space-y-0.5">
          {activeSuppressedSlots.map((s) => (
            <p key={s.slot} className="text-[11px] text-muted-foreground/70 px-1">
              {formatSlotLabel(s.slot)} — {s.reason}
            </p>
          ))}
        </div>
      )}

      {/* Stylist's note — rationale block */}
      {displayRationale && (
        <div className="flex gap-2 rounded-lg bg-muted/60 px-3 py-2.5">
          <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary/70" />
          <p className="text-xs italic text-muted-foreground leading-relaxed">
            <span className="not-italic font-semibold text-foreground/70">Stylist&rsquo;s note&nbsp;</span>
            {displayRationale}
          </p>
        </div>
      )}

      {/* Add the look — Shopify: single cart URL; non-Shopify: open-all.
          Amount is always displayTotal (sum of rendered cards) so the label
          never desyncs from what "Open all items" actually opens. */}
      {displayTotal != null && (
        <div className="space-y-2">
          {isShopify && activeCartUrl ? (
            <button
              onClick={handleShopifyCart}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold py-2.5 hover:bg-primary/90 transition-colors"
            >
              <ShoppingBag className="h-4 w-4" />
              Add the look to cart — ₹{Math.round(displayTotal).toLocaleString("en-IN")}
            </button>
          ) : (
            <>
              <button
                onClick={handleOpenAllItems}
                data-testid="open-all-items-button"
                aria-expanded={showOpenAllPanel}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold py-2.5 hover:bg-primary/90 transition-colors"
              >
                <ExternalLink className="h-4 w-4" />
                Open all items — ₹{Math.round(displayTotal).toLocaleString("en-IN")}
              </button>
              {showOpenAllPanel && (
                <div
                  data-testid="open-all-panel"
                  className="rounded-lg border bg-muted/40 p-2 space-y-1.5"
                >
                  <p className="text-[10px] text-muted-foreground px-1">
                    Opens each store in a new tab
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {priceableItems.map((item) => {
                      const url = resolveItemUrl(item)
                      if (!url) return null
                      const label =
                        item.store_display ??
                        getStoreDisplayName(item.store) ??
                        item.store ??
                        item.prod_name
                      return (
                        <a
                          key={item.article_id}
                          href={url}
                          target="_blank"
                          rel="noopener noreferrer"
                          data-testid="open-all-panel-item"
                          className="inline-flex items-center gap-1 rounded-full border bg-background text-xs px-3 py-1 text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                        >
                          {label}
                          {item.price_inr != null &&
                            ` — ₹${item.price_inr.toLocaleString("en-IN")}`}
                        </a>
                      )
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Save + share controls — interactive mode only */}
      {onSend && (
        <div className="space-y-2">
          {saveState !== "saved" && (
            <button
              onClick={() => void handleSaveLook()}
              disabled={saveState === "saving"}
              className="w-full flex items-center justify-center gap-2 rounded-lg border text-sm font-medium py-2 text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors disabled:opacity-60 disabled:cursor-wait"
            >
              <Bookmark className="h-4 w-4" />
              {saveState === "saving"
                ? "Saving…"
                : saveState === "error"
                  ? "Failed to save — tap to retry"
                  : "Save look"}
            </button>
          )}

          {saveState === "saved" && shareUrl && (
            <div className="rounded-lg border bg-muted/40 px-3 py-2.5 space-y-2">
              <div className="flex items-center gap-2">
                <Check className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                <span className="text-xs font-medium text-foreground/80">Look saved!</span>
              </div>
              <div className="flex items-center gap-1.5">
                <code className="flex-1 text-[10px] text-muted-foreground bg-background rounded px-2 py-1 border truncate select-all">
                  {shareUrl}
                </code>
                <button
                  onClick={() => void handleCopyShareUrl()}
                  className="shrink-0 rounded px-2 py-1 text-xs border hover:bg-accent transition-colors flex items-center gap-1"
                  aria-label="Copy share link"
                >
                  {copied ? <Check className="h-3 w-3 text-emerald-600" /> : <Copy className="h-3 w-3" />}
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Refinement chips */}
      {onSend && (
        <div className="flex flex-wrap gap-1.5">
          <RefinementChip
            label="More ethnic"
            onClick={() => onSend("Make this look more ethnic")}
          />
          <RefinementChip
            label="More formal"
            onClick={() => onSend("Make this look more formal")}
          />
          <RefinementChip
            label="Different colour"
            onClick={() => onSend("Show me a different colour palette")}
          />
          {filledSlotNames.slice(0, 2).map((slot) => (
            <RefinementChip
              key={slot}
              label={`Swap ${slot}`}
              onClick={() => onSend(`Swap the ${slot} in this look`)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// SlotCard
// ---------------------------------------------------------------------------

function SlotCard({
  item,
  pdpUrlTemplate,
  isSeed,
  overrideBuyUrl,
  anchorImageUrl,
  onSend,
}: {
  item: ItemSummary
  pdpUrlTemplate: string | undefined
  isSeed: boolean
  overrideBuyUrl: string | null
  /** The user's own uploaded photo — only meaningful when this card is the owned seed. */
  anchorImageUrl?: string | null
  /** Wired through so the owned card can offer "Where can I buy one like this?". */
  onSend?: (text: string) => void
}) {
  // The owned seed (user's own uploaded garment) is never purchasable: no buy link,
  // no price, no store badge — regardless of what the backend happens to send.
  const isOwned = isSeed && item.is_owned === true

  // Priority: override (from item_links) → server-built pdp_url → legacy pdp_handle template.
  const buyUrl = isOwned
    ? null
    : (overrideBuyUrl ??
      item.pdp_url ??
      (item.pdp_handle && pdpUrlTemplate
        ? pdpUrlTemplate.replace("{handle}", item.pdp_handle)
        : null))

  const slotLabel = isOwned
    ? "Your item"
    : isSeed
      ? "Hero"
      : item.outfit_slot
        ? item.outfit_slot.charAt(0).toUpperCase() + item.outfit_slot.slice(1)
        : "Item"

  // Testability attributes consumed by the Playwright proof suite — always present
  // (empty string when the underlying field is absent) so selectors can rely on them.
  const dataSlotValue = isSeed ? "seed" : (item.outfit_slot ?? "")
  const dataGenderValue = item.gender ?? ""

  // Prefer the user's actual uploaded photo; fall back to the catalogue image_url
  // (e.g. a restored session where the object URL no longer exists in memory).
  const imageSrc = isOwned ? (anchorImageUrl ?? item.image_url) : item.image_url

  const cardBody = (
    <>
      <div className="relative aspect-[4/5] bg-muted max-h-52 overflow-hidden">
        {imageSrc ? (
          <Image
            src={imageSrc}
            alt={isOwned ? "Your uploaded photo" : item.prod_name}
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
        <span
          className={cn(
            "absolute top-1.5 left-1.5 rounded-sm text-[10px] font-semibold px-1.5 py-0.5",
            isOwned ? "bg-[#E8A33D] text-white" : "bg-background/90 text-foreground",
          )}
        >
          {slotLabel}
        </span>
      </div>
      <div className="p-1.5">
        <p className="text-xs font-medium leading-tight line-clamp-2">{item.prod_name}</p>
        {!isOwned && item.price_inr != null && (
          <p className="text-xs text-muted-foreground mt-0.5">
            ₹{item.price_inr.toLocaleString("en-IN")}
          </p>
        )}
        {/* Store badge — shown on each (non-owned) card so mixed-store outfits are clear */}
        {!isOwned && (item.store_display ?? item.store) && (
          <p className="inline-flex items-center gap-0.5 mt-0.5 text-[9px] font-medium text-primary/80 bg-primary/10 rounded-sm px-1 py-0.5 leading-none">
            <Store className="h-2 w-2" aria-hidden />
            {item.store_display ?? getStoreDisplayName(item.store) ?? item.store}
          </p>
        )}
      </div>
    </>
  )

  if (isOwned) {
    return (
      <div
        className="rounded-lg border-2 border-[#E8A33D] bg-background overflow-hidden"
        data-slot={dataSlotValue}
        data-gender={dataGenderValue}
      >
        {cardBody}
        {onSend && (
          <div className="px-1.5 pb-1.5">
            <button
              type="button"
              onClick={() => onSend("Where can I buy one like this?")}
              className="w-full rounded-full border border-[#E8A33D] text-[#E8A33D] text-[10px] font-medium px-2 py-1 hover:bg-[#E8A33D]/10 transition-colors"
            >
              Where can I buy one like this?
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <a
      href={buyUrl ?? undefined}
      target={buyUrl ? "_blank" : undefined}
      rel="noopener noreferrer"
      className="group rounded-lg border bg-background overflow-hidden hover:shadow-md transition-shadow"
      data-slot={dataSlotValue}
      data-gender={dataGenderValue}
    >
      {cardBody}
    </a>
  )
}

// ---------------------------------------------------------------------------
// RefinementChip
// ---------------------------------------------------------------------------

function RefinementChip({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="rounded-full border text-xs px-3 py-1 text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
    >
      {label}
    </button>
  )
}
