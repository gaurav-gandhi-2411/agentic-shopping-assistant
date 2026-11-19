// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatMessage } from "@/lib/api/types"
import type { LimitNotice } from "@/lib/limits"
import type { DemoSessionStatus } from "@/hooks/useDemoSession"

// --- Controllable stand-ins for the network-facing hooks and heavy children ------------
const sendMessage = vi.fn()
const sendImage = vi.fn()
const resetMessages = vi.fn()
const sessionRetry = vi.fn()
const sessionRecreate = vi.fn()

let sessionStatus: DemoSessionStatus = "ready"
let chatState: {
  messages: ChatMessage[]
  isSending: boolean
  connectionLost: boolean
  limitNotice: LimitNotice | null
}

vi.mock("@/hooks/useDemoSession", () => ({
  useDemoSession: () => ({ status: sessionStatus, retry: sessionRetry, recreate: sessionRecreate }),
}))
vi.mock("@/hooks/useChatStream", () => ({
  useChatStream: () => ({
    ...chatState,
    sendMessage,
    sendImage,
    resetMessages,
    cancel: vi.fn(),
  }),
}))
// MessageList pulls in markdown/outfit boards and a react-query hook; the landing's own
// logic is what is under test, so record what it is given instead.
vi.mock("@/components/chat/MessageList", () => ({
  MessageList: (props: { messages: ChatMessage[] }) => (
    <div data-testid="message-list">{props.messages.map((m) => m.content).join("|")}</div>
  ),
}))
// ChatInput imports the on-device pose model; a one-button stand-in is enough here.
vi.mock("@/components/chat/ChatInput", () => ({
  ChatInput: (props: {
    onSend: (t: string) => void
    onSendImage?: (f: File, t?: string) => void
  }) => (
    <>
      <button type="button" onClick={() => props.onSend("typed query")}>
        send typed
      </button>
      <button
        type="button"
        onClick={() => props.onSendImage?.(new File(["x"], "look.jpg", { type: "image/jpeg" }), "my note")}
      >
        send photo
      </button>
    </>
  ),
}))

import { AssistantLanding } from "./AssistantLanding"

function msg(role: "user" | "assistant", content: string): ChatMessage {
  return { id: `${role}-${content}`, dbId: null, role, content, items: [], isStreaming: false }
}

beforeEach(() => {
  vi.useFakeTimers()
  sessionStatus = "ready"
  chatState = { messages: [], isSending: false, connectionLost: false, limitNotice: null }
  sendMessage.mockReset()
  sendImage.mockReset()
  resetMessages.mockReset()
  sessionRetry.mockReset()
  // The real recreate() flips the session back to "creating" synchronously.
  sessionRecreate.mockReset()
  sessionRecreate.mockImplementation(() => {
    sessionStatus = "creating"
  })
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("landing: chips run a real query", () => {
  it("shows the promise, the chips and the input on first paint, with no redirect", () => {
    render(<AssistantLanding />)
    expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/AI stylist/)
    expect(screen.getAllByRole("button", { name: /Sangeet outfit under ₹3,000/ })).toHaveLength(1)
    expect(screen.getByRole("button", { name: "send typed" })).toBeTruthy()
  })

  it("a chip tap on a ready session sends that exact query immediately (one tap)", () => {
    render(<AssistantLanding />)
    fireEvent.click(screen.getByRole("button", { name: "Wedding guest saree" }))
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith("Wedding guest saree")
  })
})

describe("landing: slow start (cold API)", () => {
  it("is silent at 3.9 s, shows skeleton + 'waking up' at 4 s, and is still there at 70 s", () => {
    sessionStatus = "creating"
    render(<AssistantLanding />)
    act(() => {
      vi.advanceTimersByTime(3900)
    })
    expect(screen.queryByTestId("waking-skeleton")).toBeNull()
    act(() => {
      vi.advanceTimersByTime(100)
    })
    const status = screen.getByTestId("waking-skeleton")
    expect(status.textContent).toMatch(/waking up/i)
    expect(status.textContent).toMatch(/about a minute/i)
    act(() => {
      vi.advanceTimersByTime(66_000)
    })
    expect(screen.getByTestId("waking-skeleton")).toBeTruthy()
  })

  it("swaps the skeleton away as soon as the session is ready", () => {
    sessionStatus = "creating"
    const { rerender } = render(<AssistantLanding />)
    act(() => {
      vi.advanceTimersByTime(5000)
    })
    expect(screen.getByTestId("waking-skeleton")).toBeTruthy()
    sessionStatus = "ready"
    rerender(<AssistantLanding />)
    expect(screen.queryByTestId("waking-skeleton")).toBeNull()
  })

  it("queues a chip tap made during the cold start and sends it once the session is ready", () => {
    sessionStatus = "creating"
    const { rerender } = render(<AssistantLanding />)
    fireEvent.click(screen.getByRole("button", { name: "Haldi kurta set" }))
    // Not sent yet, but the visitor sees their query and never has to tap twice.
    expect(sendMessage).not.toHaveBeenCalled()
    expect(screen.getByText("Haldi kurta set")).toBeTruthy()

    sessionStatus = "ready"
    rerender(<AssistantLanding />)
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith("Haldi kurta set")
    // A later re-render must not send it again.
    rerender(<AssistantLanding />)
    expect(sendMessage).toHaveBeenCalledTimes(1)
  })

  it("shows a friendly retry state when the session cannot be created", () => {
    sessionStatus = "error"
    render(<AssistantLanding />)
    const alert = screen.getByRole("alert")
    expect(alert.textContent).toMatch(/couldn't reach the stylist/i)
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    expect(sessionRetry).toHaveBeenCalledTimes(1)
  })
})

describe("landing: limit and busy states", () => {
  it("rate_limited: friendly copy with the wait in human terms, retry resends the question once", () => {
    chatState.messages = [msg("user", "Haldi kurta set")]
    chatState.limitNotice = { kind: "rate_limited", retryAfterSeconds: 720 }
    render(<AssistantLanding />)
    const alert = screen.getByRole("alert")
    expect(alert.textContent).toContain("try again in about 12 minutes")
    expect(alert.textContent).not.toMatch(/rate_limited|720/)
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    // The failed turn is dropped first so the question is not shown twice.
    expect(resetMessages).toHaveBeenCalledWith([])
    expect(sendMessage).toHaveBeenCalledWith("Haldi kurta set")
  })

  it("demo_limit from the session endpoint: try-tomorrow copy, no retry button", () => {
    sessionStatus = "limit"
    render(<AssistantLanding />)
    expect(screen.getByRole("alert").textContent).toMatch(/try again tomorrow/i)
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull()
  })

  it("demo_limit from the chat socket shows the same friendly state", () => {
    chatState.messages = [msg("user", "Haldi kurta set")]
    chatState.limitNotice = { kind: "demo_limit", retryAfterSeconds: null }
    render(<AssistantLanding />)
    expect(screen.getByRole("alert").textContent).toMatch(/try again tomorrow/i)
  })

  it("busy: replaces the backend's apology bubble with a notice that says when to retry", async () => {
    // Warm the module cache so the lazy MessageList resolves on a microtask under fake timers.
    await import("@/components/chat/MessageList")
    chatState.messages = [
      msg("user", "Haldi kurta set"),
      msg(
        "assistant",
        "I'm getting a lot of requests right now and couldn't put together a reply. Please try again in a minute.",
      ),
    ]
    render(<AssistantLanding />)
    await act(async () => {})
    expect(screen.getByRole("alert").textContent).toMatch(/busy.*about a minute/i)
    // The raw apology text is not rendered a second time in the message list.
    expect(screen.getByTestId("message-list").textContent).toBe("Haldi kurta set")
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    expect(sendMessage).toHaveBeenCalledWith("Haldi kurta set")
  })
})

describe("landing: queued photo during the cold start", () => {
  it("sends a photo chosen while the session is being created once, when it is ready", () => {
    sessionStatus = "creating"
    const { rerender } = render(<AssistantLanding />)
    fireEvent.click(screen.getByRole("button", { name: "send photo" }))
    expect(sendImage).not.toHaveBeenCalled()
    expect(screen.getByText("my note")).toBeTruthy() // the queued request is visible

    sessionStatus = "ready"
    rerender(<AssistantLanding />)
    expect(sendImage).toHaveBeenCalledTimes(1)
    const [file, note] = sendImage.mock.calls[0]
    expect((file as File).name).toBe("look.jpg")
    expect(note).toBe("my note")
    rerender(<AssistantLanding />)
    expect(sendImage).toHaveBeenCalledTimes(1)
  })
})

describe("landing: expired demo session (1008 invalid or missing token)", () => {
  const expired: LimitNotice = { kind: "session_expired", retryAfterSeconds: null }

  it("shows no rate-limit copy, recreates the session once and resends the turn once", () => {
    chatState.messages = [msg("user", "Haldi kurta set")]
    chatState.limitNotice = expired
    const { rerender } = render(<AssistantLanding />)

    expect(sessionRecreate).toHaveBeenCalledTimes(1)
    expect(resetMessages).toHaveBeenCalledWith([]) // failed turn dropped: no duplicate bubble
    expect(screen.queryByText(/sent a lot of requests/i)).toBeNull()
    expect(sendMessage).not.toHaveBeenCalled() // waits for the fresh session

    // The real hook clears the notice on reset; the new session then becomes ready.
    chatState = { ...chatState, messages: [], limitNotice: null }
    sessionStatus = "ready"
    rerender(<AssistantLanding />)
    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith("Haldi kurta set")
    expect(sessionRecreate).toHaveBeenCalledTimes(1)
  })

  it("a second failure shows the generic retry notice and does not loop", () => {
    chatState.messages = [msg("user", "Haldi kurta set")]
    chatState.limitNotice = expired
    const { rerender } = render(<AssistantLanding />)
    expect(sessionRecreate).toHaveBeenCalledTimes(1)

    chatState = { ...chatState, messages: [msg("user", "Haldi kurta set")], limitNotice: expired }
    sessionStatus = "ready"
    rerender(<AssistantLanding />)
    rerender(<AssistantLanding />)
    expect(sessionRecreate).toHaveBeenCalledTimes(1) // no second automatic recovery
    expect(screen.getByRole("alert").textContent).toMatch(/couldn't reach the stylist/i)
    expect(screen.getByRole("alert").textContent).not.toMatch(/sent a lot of requests/i)
  })

  it("a real rate_limited notice still shows the rate-limit copy and never recreates", () => {
    chatState.messages = [msg("user", "Haldi kurta set")]
    chatState.limitNotice = { kind: "rate_limited", retryAfterSeconds: 90 }
    render(<AssistantLanding />)
    expect(screen.getByRole("alert").textContent).toContain("try again in about 2 minutes")
    expect(sessionRecreate).not.toHaveBeenCalled()
  })

  it("keeps a single <main> landmark in both the hero and the chat state", () => {
    const { container, unmount } = render(<AssistantLanding />)
    expect(container.querySelectorAll("main")).toHaveLength(1)
    unmount()
    chatState.messages = [msg("user", "Haldi kurta set")]
    const second = render(<AssistantLanding />)
    expect(second.container.querySelectorAll("main")).toHaveLength(1)
  })
})
