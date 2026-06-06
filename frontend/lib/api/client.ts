import { createClient as createSupabaseClient } from "@/lib/supabase/client"
import type { ConversationDetail, ConversationSummary, ItemSummary } from "./types"

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

async function getToken(): Promise<string> {
  const supabase = createSupabaseClient()
  const {
    data: { session },
  } = await supabase.auth.getSession()
  if (!session?.access_token) throw new Error("Not authenticated")
  return session.access_token
}

async function authHeaders(): Promise<Record<string, string>> {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${await getToken()}`,
  }
}

// ---------------------------------------------------------------------------
// Generic fetch wrapper
// ---------------------------------------------------------------------------

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    ...init,
    headers: { ...(await authHeaders()), ...init.headers },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Conversations REST API
// ---------------------------------------------------------------------------

export const api = {
  conversations: {
    list: (): Promise<ConversationSummary[]> => request("/conversations"),

    get: (id: string): Promise<ConversationDetail> =>
      request(`/conversations/${id}`),

    create: (): Promise<ConversationSummary> =>
      request("/conversations", { method: "POST" }),

    delete: (id: string): Promise<void> =>
      request(`/conversations/${id}`, { method: "DELETE" }),

    patch: (
      id: string,
      body: { title?: string; is_public?: boolean }
    ): Promise<ConversationSummary> =>
      request(`/conversations/${id}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
  },

  catalogue: {
    similar: (articleId: string): Promise<ItemSummary[]> =>
      request(`/catalogue/${encodeURIComponent(articleId)}/similar`),
  },

  feedback: {
    submit: (
      messageId: string,
      rating: 1 | -1,
      comment?: string
    ): Promise<void> =>
      request(`/messages/${encodeURIComponent(messageId)}/feedback`, {
        method: "POST",
        body: JSON.stringify({ rating, comment: comment ?? null }),
      }),
  },
}

// ---------------------------------------------------------------------------
// WebSocket URL builder
// ---------------------------------------------------------------------------

// Browsers cannot send custom Authorization headers on WebSocket connections.
// We pass the JWT as a query parameter instead — this is the pattern baked into
// the backend's WS auth handler (get_current_user_id_ws).
// Tokens are valid for one hour; mid-session refresh is out of scope.
// conversation_id is NOT passed as a query param — the backend reads it from
// the user_message frame body. Tokens must be percent-encoded because JWTs
// contain base64url characters that are safe, but the + padding variant and
// any future header changes could break raw interpolation.
export async function getWsUrl(): Promise<string> {
  const token = await getToken()
  const wsBase = BACKEND_URL.replace(/^http/, "ws")
  return `${wsBase}/chat/stream?token=${encodeURIComponent(token)}`
}
