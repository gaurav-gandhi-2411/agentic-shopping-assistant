"use client"

export default function DemoBanner() {
  return (
    <div className="w-full bg-gray-900 text-white text-xs py-2 px-4 flex items-center justify-center gap-4 shrink-0">
      <span>Live demo — anonymous, rate-limited.</span>
      <span>Built by Gaurav Gandhi.</span>
      <a
        href="https://github.com/gaurav-gandhi-2411/agentic-shopping-assistant"
        target="_blank"
        rel="noopener noreferrer"
        className="underline underline-offset-2 hover:text-gray-300 transition-colors"
      >
        GitHub ↗
      </a>
    </div>
  )
}
