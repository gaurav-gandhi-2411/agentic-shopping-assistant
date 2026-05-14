"use client"

import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const MAX_LENGTH = 2000
const WARN_LENGTH = 1500
const ERROR_LENGTH = 1900

interface Props {
  onSend: (message: string) => void
  onCancel: () => void
  isSending: boolean
  disabled?: boolean
}

export function ChatInput({ onSend, onCancel, isSending, disabled }: Props) {
  const [text, setText] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const charCount = text.length
  const overLimit = charCount > MAX_LENGTH

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || isSending || overLimit) return
    onSend(trimmed)
    setText("")
    if (textareaRef.current) textareaRef.current.style.height = "auto"
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setText(e.target.value)
    const el = e.target
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`
  }

  const showCounter = charCount >= WARN_LENGTH
  const counterColor = overLimit
    ? "text-destructive"
    : charCount >= ERROR_LENGTH
    ? "text-destructive/80"
    : "text-amber-500"

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t bg-background px-4 py-3 flex gap-2 items-end shrink-0"
    >
      <div className="flex-1 flex flex-col gap-1">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Ask about clothing, style, outfits…"
          disabled={disabled}
          rows={1}
          className={cn(
            "w-full resize-none rounded-xl border bg-background px-3 py-2 text-sm",
            "ring-offset-background placeholder:text-muted-foreground",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
            "disabled:cursor-not-allowed disabled:opacity-50 leading-relaxed overflow-hidden",
            overLimit ? "border-destructive" : "border-input"
          )}
        />
        {showCounter && (
          <p className={cn("text-[10px] text-right pr-1", counterColor)}>
            {charCount}/{MAX_LENGTH}
          </p>
        )}
      </div>
      {isSending ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={onCancel}
          className="shrink-0 h-9 px-3"
        >
          Stop
        </Button>
      ) : (
        <Button
          type="submit"
          size="sm"
          disabled={!text.trim() || disabled || overLimit}
          className="shrink-0 h-9 px-3"
        >
          Send
        </Button>
      )}
    </form>
  )
}
