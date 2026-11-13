// WebSocket connection helpers for /chat/stream, kept free of Supabase/Next imports so they
// can be unit-tested in plain node (see wsConnect.test.ts).
//
// Why the ticket travels as a subprotocol: Cloud Run's platform request log records the full
// request URL (httpRequest.requestUrl), so a ?ticket= query param is persisted. The
// Sec-WebSocket-Protocol header is not logged there. The browser offers
// ["stylemaitri.ticket.v1", "<ticket>"]; the server selects only the marker.

export const WS_TICKET_SUBPROTOCOL = "stylemaitri.ticket.v1"

export interface WsTarget {
  url: string
  /** Present only for the subprotocol first attempt. */
  protocols?: string[]
}

export interface WsTargetDeps {
  /** ws(s):// base URL of the backend (no trailing path). */
  wsBase: string
  /** Mint a fresh single-use ticket via POST /auth/ws-ticket; throws on any failure. */
  mintTicket: () => Promise<string>
  /** Return the bearer credential for the legacy ?token= fallback. */
  getToken: () => Promise<string>
}

async function legacyTokenTarget(deps: WsTargetDeps): Promise<WsTarget> {
  // Legacy fallback: the ticket endpoint is unavailable. conversation_id is NOT a query
  // param (the backend reads it from the user_message frame). JWTs are percent-encoded.
  const token = await deps.getToken()
  return { url: `${deps.wsBase}/chat/stream?token=${encodeURIComponent(token)}` }
}

/** First attempt: ticket in the Sec-WebSocket-Protocol header, never in the URL. */
export async function resolveSubprotocolTarget(deps: WsTargetDeps): Promise<WsTarget> {
  try {
    const ticket = await deps.mintTicket()
    return {
      url: `${deps.wsBase}/chat/stream`,
      protocols: [WS_TICKET_SUBPROTOCOL, ticket],
    }
  } catch (err) {
    console.warn("[ws] ws-ticket request failed, falling back to ?token=", err)
    return legacyTokenTarget(deps)
  }
}

/**
 * Retry target after a pre-open failure (e.g. an old backend that ignores subprotocols so
 * the browser aborts the handshake). Always mints a FRESH ticket: tickets are single-use
 * and a newer backend may already have consumed the first one.
 */
export async function resolveUrlTicketTarget(deps: WsTargetDeps): Promise<WsTarget> {
  try {
    const ticket = await deps.mintTicket()
    return { url: `${deps.wsBase}/chat/stream?ticket=${encodeURIComponent(ticket)}` }
  } catch (err) {
    console.warn("[ws] ws-ticket retry failed, falling back to ?token=", err)
    return legacyTokenTarget(deps)
  }
}

export interface ChatSocketHandlers {
  /** Called with every socket created (the first attempt and the fallback). */
  onSocket: (ws: WebSocket) => void
  /** Called when the fallback retry replaces the target, so later reconnects reuse it. */
  onTarget: (target: WsTarget) => void
  /** False when the caller already replaced/closed this socket (skips the fallback). */
  isCurrent?: (ws: WebSocket) => boolean
  onopen: (ws: WebSocket) => void
  onmessage: (event: MessageEvent) => void
  /** Called for every close that is NOT swallowed by the one-time pre-open fallback. */
  onclose: (event: CloseEvent) => void
}

type SocketCtor = new (url: string, protocols?: string | string[]) => WebSocket

/**
 * Open a chat socket for `target`. If a subprotocol attempt closes before it ever opened,
 * retry exactly once via `getFallbackTarget()` (URL ticket). Never retries after open.
 */
export function openChatSocket(
  target: WsTarget,
  getFallbackTarget: () => Promise<WsTarget>,
  handlers: ChatSocketHandlers,
  Ctor: SocketCtor = WebSocket
): void {
  let fellBack = false

  function start(t: WsTarget): void {
    const ws = t.protocols ? new Ctor(t.url, t.protocols) : new Ctor(t.url)
    let opened = false
    handlers.onSocket(ws)

    ws.onopen = () => {
      opened = true
      handlers.onopen(ws)
    }
    ws.onmessage = (event: MessageEvent) => handlers.onmessage(event)
    ws.onerror = () => {
      // onclose always fires after onerror; let onclose drive retry/fallback.
    }
    ws.onclose = (event: CloseEvent) => {
      const isCurrent = handlers.isCurrent ? handlers.isCurrent(ws) : true
      if (!opened && t.protocols && !fellBack && isCurrent) {
        fellBack = true
        getFallbackTarget().then(
          (fallback) => {
            handlers.onTarget(fallback)
            start(fallback)
          },
          () => handlers.onclose(event)
        )
        return
      }
      handlers.onclose(event)
    }
  }

  start(target)
}
