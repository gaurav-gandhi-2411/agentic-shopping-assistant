// Mirror types from api/schemas.py and api/routes/conversations.py
// OutfitVariant mirrors the OutfitVariant pydantic model in the backend ChatResponse.

// ---------------------------------------------------------------------------
// Shared-look types — mirrors POST /looks and GET /looks/{id}
// ---------------------------------------------------------------------------

/** Per-item buy link returned by non-Shopify brand backends. */
export interface ItemLink {
  article_id: string
  name: string
  buy_url: string
}

/**
 * The self-contained snapshot stored when a look is saved.
 * Typed loosely so partial-data (older saves) never crash the reader.
 */
export interface LookSnapshot {
  items: Array<{
    article_id: string
    display_name?: string | null
    prod_name?: string | null
    colour?: string | null
    product_type?: string | null
    outfit_slot?: string | null
    slot_role?: string | null
    image_url?: string | null
    price_inr?: number | null
    pdp_handle?: string | null
    buy_url?: string | null
  }>
  rationale?: string | null
  cart_url?: string | null
  item_links?: ItemLink[] | null
  variant_label?: string | null
  occasion?: string | null
  look_gender?: string | null
  budget_total_inr?: number | null
  brand?: string | null
}

/** Response from POST /looks */
export interface SaveLookResponse {
  id: string
  share_path: string
}

/** Response from GET /looks/{id} */
export interface SharedLook {
  id: string
  brand: string | null
  occasion: string | null
  look_gender: string | null
  look_total_inr: number | null
  snapshot: LookSnapshot
  created_at: string
}

// ---------------------------------------------------------------------------
// Dashboard types — mirrors api/routes/dashboard.py response shape
// ---------------------------------------------------------------------------

export interface PairingStat {
  anchor_category: string
  fill_category: string
  occasion: string
  add_the_look: number
  total_signals: number
}

export interface OccasionStat {
  occasion: string
  looks_shown: number
  add_the_look_rate: number
  basket_delta_inr: number | null
}

export interface BrandStat {
  brand: string
  looks_shown: number
  add_the_look_rate: number
  basket_delta_inr: number | null
}

export interface DashboardData {
  looks_shown: number
  add_the_look_rate: number
  add_single_rate: number
  basket_size: {
    look_avg_inr: number | null
    single_avg_inr: number | null
    delta_inr: number | null
    caveat: string
  }
  top_pairings: PairingStat[]
  by_occasion: OccasionStat[]
  by_brand: BrandStat[]
}

// ---------------------------------------------------------------------------
// Catalogue / conversation types
// ---------------------------------------------------------------------------

export interface OutfitVariant {
  variant_id: string
  label: "Base" | "Colour story" | "Dressier" | "Lighter"
  rationale: string
  items: ItemSummary[]
  occasion: string | null
  budget_total_inr: number | null
  /** Shopify cart permalink pre-filled with all variant items; null for non-Shopify brands. */
  cart_url?: string | null
  /** Per-item buy links; present when cart_url is null (non-Shopify). */
  item_links?: ItemLink[] | null
}

export interface ItemSummary {
  article_id: string
  prod_name: string
  display_name: string
  colour: string
  product_type: string
  department: string
  image_url: string | null
  detail_desc: string | null
  score: number | null
  price_inr?: number | null
  pdp_handle?: string | null
  outfit_slot?: string | null
  slot_role?: string | null
}

export interface ConversationSummary {
  conversation_id: string
  title: string
  message_count: number
  last_message: string | null
  filters: Record<string, unknown>
  is_public: boolean
}

export interface ConversationDetail extends ConversationSummary {
  messages: Array<{ id: string | null; role: "user" | "assistant"; content: string }>
  retrieved_items: ItemSummary[]
}

// In-progress or completed message in the chat UI.
export interface ChatMessage {
  id: string       // Stable React key (always present — DB UUID or crypto.randomUUID())
  dbId: string | null  // Persisted DB message UUID; null when no DB backing (in-memory mode)
  role: "user" | "assistant"
  content: string
  items: ItemSummary[]
  isStreaming: boolean
  lookId?: string | null
  occasion?: string | null
  lookGender?: string | null
  budgetTotalInr?: number | null
  outfitRationale?: string | null
  outfitVariants?: OutfitVariant[] | null
  /** Shopify cart URL for the base (first) look; null for non-Shopify brands. */
  cartUrl?: string | null
  /** Per-item buy links for non-Shopify brands. */
  itemLinks?: ItemLink[] | null
}

// Discriminated union of every frame the WS server can send.
export type WsFrame =
  | { type: "session"; conversation_id: string }
  | { type: "routing"; decision: { action: string; query?: string } }
  | { type: "tool_start"; tool: string }
  | { type: "items"; items: ItemSummary[] }
  | { type: "token"; text: string }
  | {
      type: "done"
      message_id: string | null
      final_state: {
        filters: Record<string, unknown>
        out_of_catalogue: boolean
        new_items_this_turn: boolean
        response: string
        look_id?: string | null
        occasion?: string | null
        look_gender?: string | null
        budget_total_inr?: number | null
        outfit_rationale?: string | null
        outfit_variants?: OutfitVariant[] | null
        /** Shopify cart permalink for the look; null for non-Shopify brands. */
        cart_url?: string | null
        /** Per-item buy links for non-Shopify brands. */
        item_links?: ItemLink[] | null
      }
    }
  | { type: "cancelled" }
  | { type: "error"; message: string; code: string }
