"use client"

import { useRef, useState } from "react"
import { ImagePlus, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const MAX_LENGTH = 2000
const WARN_LENGTH = 1500
const ERROR_LENGTH = 1900

/** 15 MB in bytes — mirrors the backend limit on /style/from-image. */
const MAX_IMAGE_BYTES = 15 * 1024 * 1024

const ACCEPTED_IMAGE_TYPES = "image/jpeg,image/png,image/webp,image/heic,image/heif"

interface Props {
  onSend: (message: string) => void
  onCancel: () => void
  isSending: boolean
  disabled?: boolean
  /** Called when the user picks and confirms an image for /style/from-image.
   *  text is the textarea content at submission time (may be empty). */
  onSendImage?: (file: File, text?: string) => void
}

/** Small thumbnail preview chip shown after the user picks an image. */
function ImageChip({
  file,
  onRemove,
}: {
  file: File
  onRemove: () => void
}) {
  const objectUrl = useRef(URL.createObjectURL(file))
  return (
    <div className="flex items-center gap-1.5 rounded-lg border bg-muted/60 px-2 py-1 text-xs">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={objectUrl.current}
        alt="Upload preview"
        className="h-8 w-8 rounded object-cover shrink-0"
      />
      <span className="max-w-[120px] truncate text-muted-foreground">{file.name}</span>
      <button
        type="button"
        onClick={onRemove}
        aria-label="Remove image"
        className="rounded p-0.5 hover:bg-destructive/10 hover:text-destructive transition-colors"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  )
}

export function ChatInput({ onSend, onCancel, isSending, disabled, onSendImage }: Props) {
  const [text, setText] = useState("")
  const [pendingImage, setPendingImage] = useState<File | null>(null)
  const [imageError, setImageError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const charCount = text.length
  const overLimit = charCount > MAX_LENGTH

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    // If there is a pending image, send it via the image path (include any typed text).
    if (pendingImage && onSendImage) {
      onSendImage(pendingImage, text.trim() || undefined)
      setPendingImage(null)
      setText("")
      setImageError(null)
      if (textareaRef.current) textareaRef.current.style.height = "auto"
      return
    }
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

  function handleImagePickerClick() {
    fileInputRef.current?.click()
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    // Reset the input so the same file can be re-selected after removal.
    e.target.value = ""
    setImageError(null)
    if (!file) return

    if (!file.type.startsWith("image/")) {
      setImageError("That doesn't look like a supported image (JPEG, PNG, WebP, HEIC).")
      return
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setImageError("Image too large (max 15 MB). Please choose a smaller file.")
      return
    }

    setPendingImage(file)
  }

  function handleRemoveImage() {
    setPendingImage(null)
    setImageError(null)
  }

  const showCounter = charCount >= WARN_LENGTH
  const counterColor = overLimit
    ? "text-destructive"
    : charCount >= ERROR_LENGTH
    ? "text-destructive/80"
    : "text-amber-500"

  // Send is valid when: pending image present, OR non-empty text within limit.
  const canSend = pendingImage != null || (text.trim().length > 0 && !overLimit)

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t bg-background px-4 py-3 flex flex-col gap-2 shrink-0"
    >
      {/* Pending image chip */}
      {pendingImage && (
        <ImageChip file={pendingImage} onRemove={handleRemoveImage} />
      )}

      {/* Inline image error */}
      {imageError && (
        <p className="text-xs text-destructive px-1">{imageError}</p>
      )}

      <div className="flex gap-2 items-end">
        <div className="flex-1 flex flex-col gap-1">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={pendingImage ? "Add a note or just press Send…" : "Ask about clothing, style, outfits…"}
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

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_IMAGE_TYPES}
          className="sr-only"
          onChange={handleFileChange}
          aria-label="Upload garment or inspiration photo"
        />

        {/* Image upload button — only shown when onSendImage is wired */}
        {onSendImage && !isSending && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleImagePickerClick}
            disabled={disabled}
            title="Style what you own"
            aria-label="Style what you own"
            className="shrink-0 h-9 px-2.5"
          >
            <ImagePlus className="h-4 w-4" />
          </Button>
        )}

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
            disabled={!canSend || disabled}
            className="shrink-0 h-9 px-3"
          >
            Send
          </Button>
        )}
      </div>

      {/* Privacy microcopy — only shown when image upload is available */}
      {onSendImage && (
        <p className="text-[10px] text-muted-foreground text-center px-1">
          Your photo is used once to find a match and is not stored.
        </p>
      )}
    </form>
  )
}
