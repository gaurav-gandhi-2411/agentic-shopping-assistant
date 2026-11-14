import { describe, expect, it, vi } from "vitest"
import {
  openChatSocket,
  resolveSubprotocolTarget,
  resolveUrlTicketTarget,
  WS_TICKET_SUBPROTOCOL,
  type ChatSocketHandlers,
  type WsTargetDeps,
} from "./wsConnect"

// Scripted stand-in for the browser WebSocket: tests drive open/close by hand.
class FakeSocket {
  static instances: FakeSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  onclose: ((e: CloseEvent) => void) | null = null
  constructor(
    public url: string,
    public protocols?: string | string[]
  ) {
    FakeSocket.instances.push(this)
  }
  open(): void {
    this.onopen?.()
  }
  close(code = 1006): void {
    this.onclose?.({ code } as CloseEvent)
  }
}

const Ctor = FakeSocket as unknown as new (u: string, p?: string | string[]) => WebSocket

function deps(overrides: Partial<WsTargetDeps> = {}): WsTargetDeps {
  let n = 0
  return {
    wsBase: "wss://api.example.test",
    mintTicket: async () => `ticket-${++n}`,
    getToken: async () => "jwt.value.here",
    ...overrides,
  }
}

function handlers(over: Partial<ChatSocketHandlers> = {}): ChatSocketHandlers {
  return {
    onSocket: vi.fn(),
    onTarget: vi.fn(),
    onopen: vi.fn(),
    onmessage: vi.fn(),
    onclose: vi.fn(),
    ...over,
  }
}

async function flush(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0))
}

describe("resolveSubprotocolTarget", () => {
  it("offers the marker + ticket as subprotocols and keeps the ticket out of the URL", async () => {
    const target = await resolveSubprotocolTarget(deps())
    expect(target.protocols).toEqual([WS_TICKET_SUBPROTOCOL, "ticket-1"])
    expect(target.url).toBe("wss://api.example.test/chat/stream")
    expect(target.url).not.toContain("ticket-1")
  })

  it("falls back to the legacy ?token= URL when the ticket endpoint fails", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    const target = await resolveSubprotocolTarget(
      deps({ mintTicket: async () => Promise.reject(new Error("503")) })
    )
    warn.mockRestore()
    expect(target.protocols).toBeUndefined()
    expect(target.url).toBe("wss://api.example.test/chat/stream?token=jwt.value.here")
  })
})

describe("resolveUrlTicketTarget", () => {
  it("mints a fresh ticket and puts it in the URL", async () => {
    const d = deps()
    await resolveSubprotocolTarget(d) // consumes ticket-1
    const target = await resolveUrlTicketTarget(d)
    expect(target.url).toBe("wss://api.example.test/chat/stream?ticket=ticket-2")
    expect(target.protocols).toBeUndefined()
  })
})

describe("openChatSocket", () => {
  it("first attempt uses subprotocols and no ticket in the URL", async () => {
    FakeSocket.instances = []
    const target = await resolveSubprotocolTarget(deps())
    openChatSocket(target, async () => ({ url: "unused" }), handlers(), Ctor)
    expect(FakeSocket.instances).toHaveLength(1)
    expect(FakeSocket.instances[0].protocols).toEqual([WS_TICKET_SUBPROTOCOL, "ticket-1"])
    expect(FakeSocket.instances[0].url).not.toContain("ticket")
  })

  it("a pre-open failure triggers exactly one URL-ticket retry with a fresh ticket", async () => {
    FakeSocket.instances = []
    const d = deps()
    const target = await resolveSubprotocolTarget(d)
    const h = handlers()
    openChatSocket(target, () => resolveUrlTicketTarget(d), h, Ctor)

    FakeSocket.instances[0].close(1006) // handshake aborted, never opened
    await flush()
    expect(FakeSocket.instances).toHaveLength(2)
    expect(FakeSocket.instances[1].url).toBe(
      "wss://api.example.test/chat/stream?ticket=ticket-2"
    )
    expect(FakeSocket.instances[1].protocols).toBeUndefined()
    expect(h.onclose).not.toHaveBeenCalled() // the first failure was swallowed by the retry
    expect(h.onTarget).toHaveBeenCalledTimes(1)

    FakeSocket.instances[1].close(1006) // the retry fails too: no third attempt
    await flush()
    expect(FakeSocket.instances).toHaveLength(2)
    expect(h.onclose).toHaveBeenCalledTimes(1)
  })

  it("never retries after the socket opened", async () => {
    FakeSocket.instances = []
    const d = deps()
    const target = await resolveSubprotocolTarget(d)
    const h = handlers()
    openChatSocket(target, () => resolveUrlTicketTarget(d), h, Ctor)

    FakeSocket.instances[0].open()
    FakeSocket.instances[0].close(1006)
    await flush()
    expect(FakeSocket.instances).toHaveLength(1)
    expect(h.onopen).toHaveBeenCalledTimes(1)
    expect(h.onclose).toHaveBeenCalledTimes(1)
  })

  it("does not fall back for a socket the caller already replaced", async () => {
    FakeSocket.instances = []
    const d = deps()
    const target = await resolveSubprotocolTarget(d)
    const h = handlers({ isCurrent: () => false })
    openChatSocket(target, () => resolveUrlTicketTarget(d), h, Ctor)

    FakeSocket.instances[0].close(1006)
    await flush()
    expect(FakeSocket.instances).toHaveLength(1)
    expect(h.onclose).toHaveBeenCalledTimes(1)
  })

  it("a legacy ?token= first target is never retried", async () => {
    FakeSocket.instances = []
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    const d = deps({ mintTicket: async () => Promise.reject(new Error("down")) })
    const target = await resolveSubprotocolTarget(d)
    warn.mockRestore()
    const h = handlers()
    openChatSocket(target, () => resolveUrlTicketTarget(d), h, Ctor)

    FakeSocket.instances[0].close(1006)
    await flush()
    expect(FakeSocket.instances).toHaveLength(1)
    expect(h.onclose).toHaveBeenCalledTimes(1)
  })

  it("treats a throw while opening the retry socket as a failed retry (no unhandled rejection)", async () => {
    FakeSocket.instances = []
    const d = deps()
    const target = await resolveSubprotocolTarget(d)
    class ThrowOnRetry extends FakeSocket {
      constructor(url: string, protocols?: string | string[]) {
        if (FakeSocket.instances.length >= 1) throw new SyntaxError("bad url")
        super(url, protocols)
      }
    }
    const unhandled = vi.fn()
    process.on("unhandledRejection", unhandled)
    const h = handlers()
    openChatSocket(
      target,
      () => resolveUrlTicketTarget(d),
      h,
      ThrowOnRetry as unknown as new (u: string, p?: string | string[]) => WebSocket
    )

    FakeSocket.instances[0].close(1006)
    await flush()
    await flush()
    process.off("unhandledRejection", unhandled)
    expect(unhandled).not.toHaveBeenCalled()
    expect(h.onclose).toHaveBeenCalledTimes(1)
    expect(FakeSocket.instances).toHaveLength(1)
  })

  it("opens no retry socket when the socket was replaced while the ticket was being minted", async () => {
    FakeSocket.instances = []
    const d = deps()
    const target = await resolveSubprotocolTarget(d)
    let current = true
    let release: (t: { url: string }) => void = () => {}
    const pending = new Promise<{ url: string }>((resolve) => {
      release = resolve
    })
    const h = handlers({ isCurrent: () => current })
    openChatSocket(target, () => pending, h, Ctor)

    FakeSocket.instances[0].close(1006) // still current: the fallback starts minting
    current = false // the caller starts a new send during the await
    release({ url: "wss://api.example.test/chat/stream?ticket=late" })
    await flush()
    expect(FakeSocket.instances).toHaveLength(1) // no orphan socket
    expect(h.onTarget).not.toHaveBeenCalled()
    expect(h.onclose).not.toHaveBeenCalled()
  })
})
