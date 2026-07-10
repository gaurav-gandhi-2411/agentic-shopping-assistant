interface LogoMarkProps {
  className?: string
}

interface LogoProps {
  /** Wrapper className — controls layout/gap of icon + wordmark. */
  className?: string
  /** className applied to the icon svg (size, color via `currentColor`). */
  iconClassName?: string
  /** className applied to the wordmark text. */
  wordmarkClassName?: string
  /** Set false to render the icon only (e.g. compact embed headers). */
  showWordmark?: boolean
}

/**
 * Style Maitri brand mark — the "Marigold Knot": two interlocking petal-loops
 * forming a knot, evoking both a marigold (the flower strung through Indian
 * wedding decor) and a bond/friendship knot (Maitri = friend). Inlined from
 * `public/brand/mark.svg`. Rendered with `currentColor` so callers can still
 * override color via `iconClassName` (defaults to the champagne-gold brand
 * accent, matching the static mark.svg/favicon assets).
 */
export function LogoMark({ className }: LogoMarkProps) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="Style Maitri"
    >
      <g
        transform="translate(24,24)"
        stroke="currentColor"
        strokeWidth="3.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z" />
        <path
          d="M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z"
          transform="rotate(90)"
        />
      </g>
    </svg>
  )
}

/** Full lockup: icon + "Style Maitri" wordmark ("Style" in ink, "Maitri" in champagne gold). */
export function Logo({ className, iconClassName, wordmarkClassName, showWordmark = true }: LogoProps) {
  return (
    <span className={`inline-flex items-center gap-2 ${className ?? ""}`}>
      <LogoMark className={iconClassName ?? "h-6 w-6 text-champagne shrink-0"} />
      {showWordmark && (
        <span
          className={`font-serif font-semibold tracking-tight text-base leading-none text-foreground ${wordmarkClassName ?? ""}`}
        >
          Style <span className="text-champagne">Maitri</span>
        </span>
      )}
    </span>
  )
}
