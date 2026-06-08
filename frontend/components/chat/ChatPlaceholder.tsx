"use client"

import { useBrandConfig } from "@/hooks/useBrandConfig"

const FALLBACK_CHIPS = [
  "Show me something to wear",
  "Help me find an outfit",
  "What's trending?",
]

interface Props {
  userEmail: string
  onSend?: (text: string) => void
}

export function ChatPlaceholder({ userEmail, onSend }: Props) {
  const { data: brand } = useBrandConfig()

  const chips = brand?.suggestion_chips?.length ? brand.suggestion_chips : FALLBACK_CHIPS
  const displayName = brand?.display_name ?? "Shopping Assistant"

  return (
    <div className="flex flex-col items-center justify-center flex-1 min-h-[60vh] gap-4 text-center px-4">
      <div className="text-5xl select-none" aria-hidden>
        🛍️
      </div>
      <h1 className="text-2xl font-semibold tracking-tight">
        {displayName}
      </h1>
      <p className="text-muted-foreground text-sm max-w-xs">
        Signed in as <span className="font-medium text-foreground">{userEmail}</span>.
        Ask me anything about fashion.
      </p>
      {onSend && (
        <div className="flex flex-wrap justify-center gap-2 max-w-sm mt-2">
          {chips.map((chip) => (
            <button
              key={chip}
              onClick={() => onSend(chip)}
              className="text-xs px-3 py-1.5 rounded-full border border-border bg-background hover:bg-accent transition-colors"
            >
              {chip}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
