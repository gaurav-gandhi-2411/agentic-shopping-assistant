// Mirror types from api/schemas.py and api/routes/conversations.py

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
      }
    }
  | { type: "cancelled" }
  | { type: "error"; message: string; code: string }
