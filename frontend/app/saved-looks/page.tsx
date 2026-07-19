/**
 * Saved looks — /saved-looks
 *
 * Durable retrieval surface for the "Save look" feature (OutfitBoard). Save
 * only ever produced a one-time inline share link with no way to revisit it
 * later — this page reads the `asa_saved_looks` localStorage key that
 * OutfitBoard writes to and lists every look saved on this device.
 *
 * Client-only (localStorage has no server-side equivalent); no backend call.
 */

"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import Image from "next/image"
import { ArrowLeft, ExternalLink, Trash2 } from "lucide-react"
import { Logo, LogoMark } from "@/components/Logo"

const STORAGE_KEY = "asa_saved_looks"

// Schema coordinated with OutfitBoard.tsx's persistSavedLookId — keep in sync.
interface SavedLookEntry {
  id: string
  url: string
  savedAt: string
  occasion?: string | null
  itemCount?: number
  thumbnailUrl?: string | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Narrow an unknown JSON value down to a well-formed SavedLookEntry. */
function isSavedLookEntry(value: unknown): value is SavedLookEntry {
  if (typeof value !== "object" || value === null) return false
  const v = value as Record<string, unknown>
  return typeof v.id === "string" && typeof v.url === "string" && typeof v.savedAt === "string"
}

/**
 * Read + parse the saved-looks localStorage key defensively. Legacy data
 * (the old write-only format was a bare string[] of ids) and any malformed
 * JSON are both treated as "no saved looks" rather than crashing the page.
 */
function parseSavedLooks(raw: string | null): SavedLookEntry[] {
  if (!raw) return []
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isSavedLookEntry)
  } catch {
    return []
  }
}

/** Format occasion slug for display — "sangeet_look" → "Sangeet Look". */
function formatOccasion(slug: string): string {
  return slug.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Simple manual relative-time formatter — no date library in this repo. */
function formatRelativeTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ""

  const diffSec = Math.round((Date.now() - date.getTime()) / 1000)
  if (diffSec < 60) return "just now"

  const diffMin = Math.round(diffSec / 60)
  if (diffMin < 60) return `${diffMin} minute${diffMin === 1 ? "" : "s"} ago`

  const diffHour = Math.round(diffMin / 60)
  if (diffHour < 24) return `${diffHour} hour${diffHour === 1 ? "" : "s"} ago`

  const diffDay = Math.round(diffHour / 24)
  if (diffDay < 30) return `${diffDay} day${diffDay === 1 ? "" : "s"} ago`

  const diffMonth = Math.round(diffDay / 30)
  if (diffMonth < 12) return `${diffMonth} month${diffMonth === 1 ? "" : "s"} ago`

  const diffYear = Math.round(diffMonth / 12)
  return `${diffYear} year${diffYear === 1 ? "" : "s"} ago`
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function SavedLooksPage() {
  const [mounted, setMounted] = useState(false)
  const [looks, setLooks] = useState<SavedLookEntry[]>([])

  useEffect(() => {
    setMounted(true)
    setLooks(parseSavedLooks(localStorage.getItem(STORAGE_KEY)))
  }, [])

  function handleRemove(id: string) {
    setLooks((prev) => {
      const next = prev.filter((look) => look.id !== id)
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      } catch {
        // localStorage unavailable or full; best-effort.
      }
      return next
    })
  }

  if (!mounted) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">
        Loading&hellip;
      </div>
    )
  }

  return (
    <main className="min-h-screen bg-background">
      <header className="flex shrink-0 items-center justify-between border-b border-border bg-background px-6 py-4">
        <Logo />
        <Link
          href="/demo"
          className="inline-flex min-h-11 items-center gap-1.5 px-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to styling
        </Link>
      </header>

      <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Saved looks</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Outfits you&rsquo;ve saved while styling on this device.
          </p>
        </div>

        {looks.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 py-24 text-center select-none">
            <span aria-hidden>
              <LogoMark className="h-14 w-14 text-champagne" />
            </span>
            <p className="font-serif text-xl font-semibold tracking-tight text-foreground">
              No saved looks yet
            </p>
            <p className="max-w-xs text-sm leading-relaxed text-muted-foreground">
              Save a look while styling and it will show up here so you can find it again later.
            </p>
            <Link
              href="/demo"
              className="mt-2 inline-flex min-h-11 items-center justify-center rounded-full bg-primary px-4 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
            >
              Start styling
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {looks.map((look) => (
              <div
                key={look.id}
                className="flex flex-col overflow-hidden rounded-lg border bg-card transition-shadow hover:shadow-md"
              >
                <a href={look.url} className="block">
                  <div className="relative aspect-[4/5] bg-muted">
                    {look.thumbnailUrl ? (
                      <Image
                        src={look.thumbnailUrl}
                        alt={look.occasion ? formatOccasion(look.occasion) : "Saved look"}
                        fill
                        sizes="(max-width: 640px) 50vw, 33vw"
                        unoptimized
                        className="object-cover"
                      />
                    ) : (
                      <div className="flex h-full w-full select-none items-center justify-center text-3xl">
                        👗
                      </div>
                    )}
                  </div>
                </a>
                <div className="flex flex-1 flex-col gap-1 p-2.5">
                  <div className="flex flex-wrap items-center gap-1.5">
                    {look.occasion && (
                      <span className="inline-block rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">
                        {formatOccasion(look.occasion)}
                      </span>
                    )}
                    {typeof look.itemCount === "number" && (
                      <span className="text-[10px] text-muted-foreground">
                        {look.itemCount} item{look.itemCount === 1 ? "" : "s"}
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-muted-foreground">
                    Saved {formatRelativeTime(look.savedAt)}
                  </p>
                  <div className="mt-auto flex items-center justify-between pt-1.5">
                    <a
                      href={look.url}
                      className="inline-flex min-h-11 items-center gap-1 text-xs font-medium text-primary transition-colors hover:text-primary/80"
                    >
                      View look
                      <ExternalLink className="h-3 w-3" />
                    </a>
                    <button
                      type="button"
                      onClick={() => handleRemove(look.id)}
                      aria-label="Remove saved look"
                      className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  )
}
