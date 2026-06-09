"use client"

import { useEffect, useRef } from "react"
import Image from "next/image"
import { useBrandConfig } from "@/hooks/useBrandConfig"
import type { ItemSummary } from "@/lib/api/types"

interface OutfitBoardProps {
  items: ItemSummary[]
  lookId: string | null | undefined
  occasion: string | null | undefined
  lookGender: string | null | undefined
  budgetTotalInr: number | null | undefined
  sessionId: string
  anchorItemId: string
  anchorCategory: string
  brand: string | undefined
  onSend?: (text: string) => void
}

// Format occasion slug for display (e.g. "sangeet_look" -> "Sangeet Look")
function formatOccasion(slug: string): string {
  return slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

// Post a flywheel event — fire-and-forget, never throw
async function postEvent(backendUrl: string, token: string, payload: object): Promise<void> {
  try {
    await fetch(`${backendUrl}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(payload),
    })
  } catch {
    // Telemetry is best-effort; swallow errors silently
  }
}

export function OutfitBoard({
  items,
  lookId,
  occasion,
  lookGender,
  budgetTotalInr,
  sessionId,
  anchorItemId,
  anchorCategory,
  brand,
  onSend,
}: OutfitBoardProps) {
  const { data: brandConfig } = useBrandConfig()
  const loggedRef = useRef(false)

  const seed = items.find((it) => it.slot_role === "seed")
  const complements = items.filter((it) => it.slot_role === "complement")
  const allOutfitItems = seed ? [seed, ...complements] : complements

  // Log look_shown once on mount
  useEffect(() => {
    if (loggedRef.current || !lookId) return
    loggedRef.current = true
    // Get auth token from sessionStorage (demo) or skip (token fetching is async; best-effort)
    const demoToken =
      typeof window !== "undefined" ? sessionStorage.getItem("demo_session_token") : null
    const backendUrl =
      typeof window !== "undefined"
        ? (sessionStorage.getItem("demo_backend_url") ??
          process.env.NEXT_PUBLIC_BACKEND_URL ??
          "http://localhost:8000")
        : "http://localhost:8000"
    if (demoToken) {
      postEvent(backendUrl, demoToken, {
        event_type: "look_shown",
        session_id: sessionId,
        look_id: lookId,
        anchor_item_id: anchorItemId,
        anchor_category: anchorCategory,
        occasion: occasion ?? null,
        brand: brand ?? null,
        look_total_inr: budgetTotalInr ? Math.round(budgetTotalInr) : null,
        filled_slots: complements.map((c) => ({
          slot: c.outfit_slot,
          item_id: c.article_id,
          brand: brand,
          price_inr: c.price_inr,
        })),
      })
    }
  }, [lookId]) // eslint-disable-line react-hooks/exhaustive-deps

  function handleBuyTheLook() {
    if (!lookId) return
    const demoToken =
      typeof window !== "undefined" ? sessionStorage.getItem("demo_session_token") : null
    const backendUrl =
      typeof window !== "undefined"
        ? (sessionStorage.getItem("demo_backend_url") ??
          process.env.NEXT_PUBLIC_BACKEND_URL ??
          "http://localhost:8000")
        : "http://localhost:8000"
    if (demoToken) {
      postEvent(backendUrl, demoToken, {
        event_type: "add_the_look",
        session_id: sessionId,
        look_id: lookId,
        anchor_item_id: anchorItemId,
        anchor_category: anchorCategory,
        occasion: occasion ?? null,
        brand: brand ?? null,
        look_total_inr: budgetTotalInr ? Math.round(budgetTotalInr) : null,
        filled_slots: complements.map((c) => ({
          slot: c.outfit_slot,
          item_id: c.article_id,
          brand: brand,
          price_inr: c.price_inr,
        })),
      })
    }
    // Open each item's PDP in a new tab
    allOutfitItems.forEach((item) => {
      if (item.pdp_handle && brandConfig?.pdp_url_template) {
        window.open(
          brandConfig.pdp_url_template.replace("{handle}", item.pdp_handle),
          "_blank",
        )
      }
    })
  }

  if (allOutfitItems.length === 0) return null

  const filledSlotNames = complements
    .map((c) => c.outfit_slot)
    .filter(Boolean) as string[]

  return (
    <div className="w-full rounded-xl border bg-card p-4 space-y-3">
      {/* Header */}
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

      {/* Slot grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {allOutfitItems.map((item) => (
          <SlotCard
            key={item.article_id}
            item={item}
            pdpUrlTemplate={brandConfig?.pdp_url_template}
            isSeed={item.slot_role === "seed"}
          />
        ))}
      </div>

      {/* Buy the look */}
      {budgetTotalInr != null && (
        <button
          onClick={handleBuyTheLook}
          className="w-full rounded-lg bg-primary text-primary-foreground text-sm font-semibold py-2.5 hover:bg-primary/90 transition-colors"
        >
          Buy the look — ₹{Math.round(budgetTotalInr).toLocaleString("en-IN")}
        </button>
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

function SlotCard({
  item,
  pdpUrlTemplate,
  isSeed,
}: {
  item: ItemSummary
  pdpUrlTemplate: string | undefined
  isSeed: boolean
}) {
  const buyUrl =
    item.pdp_handle && pdpUrlTemplate
      ? pdpUrlTemplate.replace("{handle}", item.pdp_handle)
      : null
  const slotLabel = isSeed
    ? "Hero"
    : item.outfit_slot
      ? item.outfit_slot.charAt(0).toUpperCase() + item.outfit_slot.slice(1)
      : "Item"

  return (
    <a
      href={buyUrl ?? undefined}
      target={buyUrl ? "_blank" : undefined}
      rel="noopener noreferrer"
      className="group rounded-lg border bg-background overflow-hidden hover:shadow-md transition-shadow"
    >
      <div className="relative aspect-[4/5] bg-muted">
        {item.image_url ? (
          <Image
            src={item.image_url}
            alt={item.prod_name}
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
      <div className="p-1.5">
        <p className="text-xs font-medium leading-tight line-clamp-2">{item.prod_name}</p>
        {item.price_inr != null && (
          <p className="text-xs text-muted-foreground mt-0.5">
            ₹{item.price_inr.toLocaleString("en-IN")}
          </p>
        )}
      </div>
    </a>
  )
}

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
