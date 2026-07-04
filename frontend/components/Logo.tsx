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
 * StyleMitra brand mark — a continuous thread forming a stylised "S" (style,
 * stitching) with a small sparkle accent (the "AI stylist" cue, echoing the
 * Sparkles icon used elsewhere for stylist notes). Rendered with
 * `currentColor` so it inherits whatever text color utility is applied
 * (defaults to `text-primary`, which adapts automatically between light and
 * dark themes via the shadcn/ui CSS variables).
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
      {/* Thread/ribbon forming a stylised "S" */}
      <path
        d="M22 9c0-3-4-4-8-3s-6 3-6 5.5S12 15 16 16s8 3.5 8 6-4 4.5-8 3.5-6-3-6-5.5"
        stroke="currentColor"
        strokeWidth="3.2"
        strokeLinecap="round"
      />
      {/* Sparkle accent — the "AI stylist" cue */}
      <path
        d="M25.5 5.5l.9 2.3 2.3.9-2.3.9-.9 2.3-.9-2.3-2.3-.9 2.3-.9z"
        fill="currentColor"
        opacity="0.6"
      />
    </svg>
  )
}

/** Full lockup: icon + "StyleMitra" wordmark. */
export function Logo({ className, iconClassName, wordmarkClassName, showWordmark = true }: LogoProps) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className ?? ""}`}>
      <LogoMark className={iconClassName ?? "h-5 w-5 text-primary shrink-0"} />
      {showWordmark && (
        <span className={`font-semibold tracking-tight text-sm leading-none ${wordmarkClassName ?? ""}`}>
          StyleMitra
        </span>
      )}
    </span>
  )
}
