import type { ReactNode } from "react"
import DemoBanner from "@/components/DemoBanner"

export const metadata = {
  title: "Agentic Shopping Assistant — Live Demo",
}

export default function DemoLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <DemoBanner />
      <div className="flex-1 flex flex-col min-h-0">{children}</div>
    </div>
  )
}
