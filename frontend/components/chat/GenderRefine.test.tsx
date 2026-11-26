// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { COPY } from "@/lib/copy"
import type { ChatMessage, GenderRefineOption } from "@/lib/api/types"
import { GenderRefine } from "./GenderRefine"
import { MessageBubble } from "./MessageBubble"

// ItemCard (rendered inside MessageBubble) reads the brand config through react-query; this test
// is about the refine control only, so the brand query is stubbed out.
vi.mock("@/hooks/useBrandConfig", () => ({ useBrandConfig: () => ({ data: undefined }) }))

afterEach(cleanup)

const OPTIONS: GenderRefineOption[] = [
  { gender: "women", query: "Haldi kurta set for women" },
  { gender: "men", query: "Haldi kurta set for men" },
]

function message(over: Partial<ChatMessage>): ChatMessage {
  return {
    id: "m1",
    dbId: null,
    role: "assistant",
    content: "Here are some options.",
    items: [
      {
        article_id: "1",
        prod_name: "Kurta Set",
        display_name: "Kurta Set",
        colour: "Yellow",
        product_type: "kurta set",
        department: "",
        gender: "women",
      },
    ],
    isStreaming: false,
    ...over,
  } as ChatMessage
}

describe("GenderRefine", () => {
  it("uses the approved Women / Men labels from copy.ts", () => {
    expect(COPY.refine.women).toBe("Women")
    expect(COPY.refine.men).toBe("Men")
  })

  it("renders one real button per option, named by its label, with a 44 px tap height", () => {
    render(<GenderRefine options={OPTIONS} onSelect={() => {}} />)
    const women = screen.getByRole("button", { name: "Women" })
    const men = screen.getByRole("button", { name: "Men" })
    for (const b of [women, men]) {
      expect(b.className).toContain("min-h-11")
      expect(b.className).toContain("focus-visible:ring-2")
    }
  })

  it("tapping an option sends that option's query once and disables both (loading)", () => {
    const onSelect = vi.fn()
    render(<GenderRefine options={OPTIONS} onSelect={onSelect} />)
    fireEvent.click(screen.getByRole("button", { name: "Men" }))
    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(onSelect).toHaveBeenCalledWith("Haldi kurta set for men")
    expect((screen.getByRole("button", { name: "Women" }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole("button", { name: "Men" }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: "Women" }))
    expect(onSelect).toHaveBeenCalledTimes(1)
  })

  it("renders nothing (empty state) with no options", () => {
    const { container, rerender } = render(<GenderRefine options={null} onSelect={() => {}} />)
    expect(container.firstChild).toBeNull()
    rerender(<GenderRefine options={[]} onSelect={() => {}} />)
    expect(container.firstChild).toBeNull()
    rerender(<GenderRefine options={undefined} onSelect={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
})

describe("GenderRefine in MessageBubble (visibility rules)", () => {
  it("shows for the latest finished assistant message that carries options", () => {
    render(
      <MessageBubble message={message({ genderRefine: OPTIONS })} onSend={() => {}} isLatestAssistant />,
    )
    expect(screen.getByRole("button", { name: "Women" })).toBeTruthy()
    expect(screen.getByRole("button", { name: "Men" })).toBeTruthy()
  })

  it("is absent when the backend sent none (gendered query such as Men's kurta, or saree)", () => {
    render(<MessageBubble message={message({ genderRefine: null })} onSend={() => {}} isLatestAssistant />)
    expect(screen.queryByRole("button", { name: "Women" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Men" })).toBeNull()
  })

  it("is absent on an older assistant message, while streaming, and with no onSend", () => {
    const m = message({ genderRefine: OPTIONS })
    const { rerender } = render(<MessageBubble message={m} onSend={() => {}} isLatestAssistant={false} />)
    expect(screen.queryByRole("button", { name: "Women" })).toBeNull()
    rerender(<MessageBubble message={{ ...m, isStreaming: true }} onSend={() => {}} isLatestAssistant />)
    expect(screen.queryByRole("button", { name: "Women" })).toBeNull()
    rerender(<MessageBubble message={m} isLatestAssistant />)
    expect(screen.queryByRole("button", { name: "Women" })).toBeNull()
  })

  it("is absent when the message has no cards", () => {
    render(
      <MessageBubble
        message={message({ items: [], genderRefine: OPTIONS })}
        onSend={() => {}}
        isLatestAssistant
      />,
    )
    expect(screen.queryByRole("button", { name: "Women" })).toBeNull()
  })

  it("tapping sends the backend's query as the next message", () => {
    const onSend = vi.fn()
    render(<MessageBubble message={message({ genderRefine: OPTIONS })} onSend={onSend} isLatestAssistant />)
    fireEvent.click(screen.getByRole("button", { name: "Women" }))
    expect(onSend).toHaveBeenCalledWith("Haldi kurta set for women")
  })
})
