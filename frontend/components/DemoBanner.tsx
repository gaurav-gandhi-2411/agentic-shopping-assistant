"use client"

export default function DemoBanner() {
  return (
    <div className="dark w-full bg-background text-foreground text-xs py-2 px-4 flex items-center justify-center gap-4 shrink-0">
      <span>Live demo — anonymous, rate-limited.</span>
      <span>Style Maitri</span>
      <a
        href="/pdp-demo/snitch"
        target="_blank"
        rel="noopener noreferrer"
        className="text-champagne hover:underline underline-offset-2"
      >
        For brands: embed this widget →
      </a>
    </div>
  )
}
