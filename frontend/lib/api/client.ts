import { createClient as createSupabaseClient } from "@/lib/supabase/client"
import type { ConversationDetail, ConversationSummary, DashboardData, ItemSummary } from "./types"
import {
  resolveSubprotocolTarget,
  resolveUrlTicketTarget,
  type WsTarget,
  type WsTargetDeps,
} from "./wsConnect"

function getBackendUrl(): string {
  if (typeof window !== "undefined") {
    const demoUrl = sessionStorage.getItem("demo_backend_url")
    if (demoUrl) return demoUrl
  }
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000"
}

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

async function getToken(): Promise<string> {
  if (typeof window !== "undefined") {
    const demoToken = sessionStorage.getItem("demo_session_token")
    if (demoToken) return demoToken
  }
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
  const res = await fetch(`${getBackendUrl()}${path}`, {
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

  dashboard: {
    get: (): Promise<DashboardData> => request("/dashboard"),
  },
}

// ---------------------------------------------------------------------------
// WebSocket URL builder
// ---------------------------------------------------------------------------

// Browsers cannot send custom Authorization headers on WebSocket connections.
// Preferred path: exchange a short-lived single-use ticket via POST /auth/ws-ticket and
// offer it as a Sec-WebSocket-Protocol entry (see wsConnect.ts), so it never appears in the
// URL that Cloud Run's request log records.
// Fallbacks (graceful degradation): a pre-open failure retries once with ?ticket=<fresh
// ticket> (backend that predates the subprotocol path); if the ticket endpoint itself fails
// we fall back to ?token=<jwt> so the user is never silently blocked.
async function mintWsTicket(): Promise<string> {
  const res = await fetch(`${getBackendUrl()}/auth/ws-ticket`, {
    method: "POST",
    headers: await authHeaders(),
  })
  if (!res.ok) throw new Error(`ws-ticket endpoint returned ${res.status}`)
  const { ticket } = (await res.json()) as { ticket: string }
  return ticket
}

function wsTargetDeps(): WsTargetDeps {
  return {
    wsBase: getBackendUrl().replace(/^http/, "ws"),
    mintTicket: mintWsTicket,
    getToken,
  }
}

export function getWsTarget(): Promise<WsTarget> {
  return resolveSubprotocolTarget(wsTargetDeps())
}

export function getWsFallbackTarget(): Promise<WsTarget> {
  return resolveUrlTicketTarget(wsTargetDeps())
}
