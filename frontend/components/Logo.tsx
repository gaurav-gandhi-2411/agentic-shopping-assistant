interface LogoMarkProps {
  className?: string
}

interface LogoProps {
  /** Wrapper className — controls layout/gap of icon + wordmark. */
  className?: string
  /** className applied to the icon svg (size, color). */
  iconClassName?: string
  /** className applied to the wordmark text. */
  wordmarkClassName?: string
  /** Set false to render the icon only (e.g. compact embed headers). */
  showWordmark?: boolean
}

/**
 * StyleMitra brand mark — the "Smiling Hanger": a clothes hanger whose bar
 * reads as a smile, evoking a friendly style companion (StyleMitra = Style +
 * Mitra/friend). Rendered with `currentColor` so it inherits whatever text
 * color utility is applied (defaults to `text-primary`, which adapts
 * automatically between light and dark themes via the shadcn/ui CSS
 * variables).
 */
export function LogoMark({ className }: LogoMarkProps) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="StyleMitra"
    >
      {/* Shoulders */}
      <path
        d="M5.5 18.2L16 9.4l10.5 8.8"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Smile bar */}
      <path
        d="M5.5 18.2c3.2 4.6 17.8 4.6 21 0"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Hook */}
      <path
        d="M16 9.4V8.1c0-1.15.9-2.05 2.05-2.05"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** Full lockup: icon + "StyleMitra" wordmark (marigold accent on "Mitra"). */
export function Logo({ className, iconClassName, wordmarkClassName, showWordmark = true }: LogoProps) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className ?? ""}`}>
      <LogoMark className={iconClassName ?? "h-5 w-5 text-primary shrink-0"} />
      {showWordmark && (
        <span className={`font-semibold tracking-tight text-sm leading-none ${wordmarkClassName ?? ""}`}>
          Style<span className="text-[#E8A33D]">Mitra</span>
        </span>
      )}
    </span>
  )
}
