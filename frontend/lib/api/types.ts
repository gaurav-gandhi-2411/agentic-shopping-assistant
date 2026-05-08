// Mirror types from api/schemas.py and api/routes/conversations.py

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
  messages: Array<{ role: "user" | "assistant"; content: string }>
  retrieved_items: ItemSummary[]
}

// In-progress or completed message in the chat UI.
export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  items: ItemSummary[]
  isStreaming: boolean
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
      final_state: {
        filters: Record<string, unknown>
        out_of_catalogue: boolean
        new_items_this_turn: boolean
        response: string
      }
    }
  | { type: "cancelled" }
  | { type: "error"; message: string; code: string }
