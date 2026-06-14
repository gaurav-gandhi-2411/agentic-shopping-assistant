"use client"

import { useState } from "react"
import Image from "next/image"
import { useQuery } from "@tanstack/react-query"
import { Store } from "lucide-react"
import { api } from "@/lib/api/client"
import type { ItemSummary } from "@/lib/api/types"
import { useBrandConfig } from "@/hooks/useBrandConfig"
import { getStoreDisplayName } from "@/lib/stores"
import { cn } from "@/lib/utils"

interface Props {
  item: ItemSummary
  onSend?: (text: string) => void
}

export function ItemCard({ item, onSend }: Props) {
  const [showSimilar, setShowSimilar] = useState(false)
  const score = item.score !== null ? Math.round(item.score * 100) : null
  const { data: brand } = useBrandConfig()

  // Cross-store buy URL: prefer server-built pdp_url; fall back to legacy template expansion.
  const buyUrl =
    item.pdp_url ??
    (item.pdp_handle && brand?.pdp_url_template
      ? brand.pdp_url_template.replace("{handle}", item.pdp_handle)
      : null)

  // Store display name: prefer server-supplied store_display; fall back to client map.
  const storeDisplay =
    item.store_display ?? getStoreDisplayName(item.store) ?? brand?.display_name ?? null

  return (
    <div className="rounded-lg border bg-card overflow-hidden">
      <div className="flex gap-3 p-3 hover:bg-accent/30 transition-colors">
        {/* Image or placeholder */}
        <div className="w-16 h-20 shrink-0 rounded-md overflow-hidden bg-muted flex items-center justify-center">
          {item.image_url ? (
            <Image
              src={item.image_url}
              alt={item.prod_name}
              width={64}
              height={80}
              sizes="64px"
              unoptimized
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-2xl select-none" aria-hidden>
              👗
            </span>
          )}
        </div>

        {/* Info */}
        <div className="flex flex-col justify-between min-w-0 flex-1">
          <div>
            <p className="text-sm font-medium leading-tight line-clamp-2">
              {item.prod_name}
            </p>
            <div className="flex flex-wrap gap-1 mt-1">
              <Badge>{item.product_type}</Badge>
              {item.colour && <Badge>{item.colour}</Badge>}
              {/* Store badge — always shown when store info is available */}
              {storeDisplay && (
                <Badge variant="store">
                  <Store className="inline h-2.5 w-2.5 mr-0.5 -mt-px" aria-hidden />
                  {storeDisplay}
                </Badge>
              )}
            </div>
            {item.price_inr != null && (
              <p className="text-xs font-semibold text-foreground mt-1">
                ₹{item.price_inr.toLocaleString("en-IN")}
              </p>
            )}
          </div>
          <div className="flex items-center justify-between mt-1 gap-1 flex-wrap">
            {score !== null && (
              <p className="text-xs text-muted-foreground">{score}% match</p>
            )}
            <div className={cn("flex gap-2 items-center flex-wrap", score === null && "ml-auto")}>
              {onSend && (
                <button
                  className="text-[10px] text-primary hover:text-primary/80 underline underline-offset-2 transition-colors"
                  onClick={() => onSend(`Style this ${item.prod_name}`)}
                >
                  Style this
                </button>
              )}
              <button
                className="text-[10px] text-muted-foreground hover:text-foreground underline underline-offset-2 transition-colors"
                onClick={() => setShowSimilar((v) => !v)}
                aria-expanded={showSimilar}
              >
                {showSimilar ? "Hide similar" : "More like this"}
              </button>
              {buyUrl && (
                <a
                  href={buyUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block text-[10px] font-medium px-2 py-0.5 rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors shrink-0"
                >
                  Buy at {storeDisplay ?? "Shop"}
                </a>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Similar items panel */}
      {showSimilar && (
        <SimilarItemsPanel articleId={item.article_id} onSend={onSend} />
      )}
    </div>
  )
}

function SimilarItemsPanel({
  articleId,
  onSend,
}: {
  articleId: string
  onSend?: (text: string) => void
}) {
  const { data, isLoading, isError } = useQuery<ItemSummary[]>({
    queryKey: ["similar", articleId],
    queryFn: () => api.catalogue.similar(articleId),
    staleTime: 5 * 60_000,
  })

  return (
    <div className="border-t bg-muted/30 px-3 py-2">
      <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mb-2">
        Similar items
      </p>
      {isLoading && (
        <p className="text-xs text-muted-foreground">Loading…</p>
      )}
      {isError && (
        <p className="text-xs text-destructive">Could not load similar items.</p>
      )}
      {data && data.length === 0 && (
        <p className="text-xs text-muted-foreground">No similar items found.</p>
      )}
      {data && data.length > 0 && (
        <div className="flex flex-col gap-2">
          {data.map((sim) => (
            <SimilarItemRow key={sim.article_id} item={sim} onSend={onSend} />
          ))}
        </div>
      )}
    </div>
  )
}

function SimilarItemRow({
  item,
  onSend,
}: {
  item: ItemSummary
  onSend?: (text: string) => void
}) {
  const score = item.score !== null ? Math.round(item.score * 100) : null
  return (
    <div className="flex items-center gap-2">
      <div className="w-8 h-10 shrink-0 rounded overflow-hidden bg-muted flex items-center justify-center">
        {item.image_url ? (
          <Image
            src={item.image_url}
            alt={item.prod_name}
            width={32}
            height={40}
            sizes="32px"
            unoptimized
            className="w-full h-full object-cover"
          />
        ) : (
          <span className="text-sm select-none" aria-hidden>
            👗
          </span>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium truncate leading-tight">
          {item.prod_name}
        </p>
        <p className="text-[10px] text-muted-foreground truncate">
          {item.product_type}
          {item.colour ? ` · ${item.colour}` : ""}
          {item.store_display ?? (item.store ? ` · ${item.store}` : "")}
          {score !== null ? ` · ${score}%` : ""}
        </p>
      </div>
      {onSend && (
        <button
          className="text-[10px] text-primary hover:text-primary/80 underline underline-offset-2 transition-colors shrink-0"
          onClick={() => onSend(`Style this ${item.prod_name}`)}
        >
          Style this
        </button>
      )}
    </div>
  )
}

interface BadgeProps {
  children: React.ReactNode
  variant?: "default" | "store"
}

function Badge({ children, variant = "default" }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm px-1.5 py-0.5 text-[10px] font-medium leading-none",
        variant === "store"
          ? "bg-primary/10 text-primary"
          : "bg-secondary text-secondary-foreground",
      )}
    >
      {children}
    </span>
  )
}
