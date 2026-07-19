"use client"

import { useState } from "react"
import Image from "next/image"
import { useQuery } from "@tanstack/react-query"
import { ShoppingBag, Tag } from "lucide-react"
import { api } from "@/lib/api/client"
import type { ItemSummary, PriceMatch } from "@/lib/api/types"
import { ProductImage } from "@/components/ProductImage"
import { Button } from "@/components/ui/button"
import { useBrandConfig } from "@/hooks/useBrandConfig"
import { getStoreDisplayName } from "@/lib/stores"

interface Props {
  item: ItemSummary
  onSend?: (text: string) => void
}

export function ItemCard({ item, onSend }: Props) {
  const [showSimilar, setShowSimilar] = useState(false)
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
    <div
      className="rounded-lg border bg-card overflow-hidden flex flex-col transition-shadow hover:shadow-md"
      data-gender={item.gender}
    >
      {/* Image — the hero. Tall editorial aspect ratio, subtly rounded top
          (inherited from the card's own overflow-hidden + rounded-lg). */}
      <div className="relative w-full aspect-[4/5] shrink-0 bg-muted overflow-hidden group">
        <ProductImage
          src={item.image_url}
          alt={item.prod_name}
          sizes="(max-width: 640px) 50vw, 320px"
        />
      </div>

      {/* Minimal chrome below the image. flex-1 + mt-auto on the button row
          below keeps that row bottom-aligned across cards in the same grid
          row even when the title wraps to 1 vs 2 lines (grid stretches every
          card to the tallest sibling; without this the row would float at
          different heights). */}
      <div className="flex flex-col gap-1 p-3 flex-1">
        <p className="text-sm font-medium leading-tight line-clamp-2 text-foreground">
          {item.prod_name}
        </p>
        <p className="text-xs text-muted-foreground truncate">
          {storeDisplay && (
            <>
              <ShoppingBag className="inline h-3 w-3 mr-1 -mt-0.5" aria-hidden />
              {storeDisplay}
              {" · "}
            </>
          )}
          {item.product_type}
          {item.colour ? ` · ${item.colour}` : ""}
        </p>

        {/* item.score is a raw hybrid/RRF ranking value (~0.01-0.03 by construction),
            not a 0-1 relevance probability — rendering it as "N% match" showed "1%
            match" on good results and made healthy retrieval look broken. Ranking
            scores are for ordering, never for display. */}
        {item.price_inr != null && (
          <p className="text-sm font-semibold text-foreground mt-0.5">
            ₹{item.price_inr.toLocaleString("en-IN")}
          </p>
        )}

        <div className="flex items-center gap-1.5 flex-wrap mt-auto pt-1.5">
          {onSend && (
            <Button
              variant="ghost"
              size="chip"
              className="border border-border text-muted-foreground"
              onClick={() => onSend(`Style this ${item.prod_name}`)}
            >
              Style this
            </Button>
          )}
          <Button
            variant="ghost"
            size="chip"
            className="border border-border text-muted-foreground"
            onClick={() => setShowSimilar((v) => !v)}
            aria-expanded={showSimilar}
          >
            {showSimilar ? "Hide similar" : "More like this"}
          </Button>
          {buyUrl && (
            <Button asChild variant="default" size="chip" className="shrink-0 ml-auto">
              <a href={buyUrl} target="_blank" rel="noopener noreferrer">
                Buy at {storeDisplay ?? "Shop"}
              </a>
            </Button>
          )}
        </div>
      </div>

      {/* Similar items panel */}
      {showSimilar && (
        <SimilarItemsPanel articleId={item.article_id} onSend={onSend} />
      )}

      {/* Cross-store price matches — only rendered when matches exist */}
      {item.price_matches && item.price_matches.length > 0 && (
        <PriceMatchPanel matches={item.price_matches} />
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
        <p className="text-xs font-medium truncate leading-tight" title={item.prod_name}>
          {item.prod_name}
        </p>
        <p className="text-[10px] text-muted-foreground truncate">
          {item.product_type}
          {item.colour ? ` · ${item.colour}` : ""}
          {item.store_display ?? (item.store ? ` · ${item.store}` : "")}
        </p>
      </div>
      {onSend && (
        <Button
          variant="link"
          size="chip"
          className="text-[10px] text-primary hover:text-primary/80 underline underline-offset-2 shrink-0"
          onClick={() => onSend(`Style this ${item.prod_name}`)}
        >
          Style this
        </Button>
      )}
    </div>
  )
}

/**
 * Cross-store price-match panel — shows same product in other stores, lowest price first.
 * Prices are catalogue snapshots; the panel explicitly labels them as such.
 * Rendered only when price_matches is non-empty (current reality: fires ~never).
 */
function PriceMatchPanel({ matches }: { matches: PriceMatch[] }) {
  return (
    <div className="border-t bg-muted/20 px-3 py-2">
      <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mb-1.5 flex items-center gap-1">
        <Tag className="h-2.5 w-2.5" aria-hidden />
        Also available at
      </p>
      <div className="flex flex-col gap-1">
        {matches.map((m) => (
          <div key={m.store} className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="text-[10px] font-medium text-foreground truncate">
                {m.store_display}
              </span>
              {m.price_inr != null && (
                <span className="text-[10px] text-muted-foreground">
                  &#8377;{m.price_inr.toLocaleString("en-IN")}
                  {" "}
                  <span className="italic">(snapshot price)</span>
                </span>
              )}
            </div>
            {m.pdp_url && (
              <Button asChild variant="secondary" size="chip" className="text-[10px] shrink-0">
                <a href={m.pdp_url} target="_blank" rel="noopener noreferrer">
                  View
                </a>
              </Button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
