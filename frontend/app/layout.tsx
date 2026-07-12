import type { Metadata } from "next"
import { Fraunces, Inter } from "next/font/google"
import { Providers } from "@/components/Providers"
import "./globals.css"

// Body copy: Inter. Wired as a CSS var (not .className) so both fonts coexist
// and Tailwind's font-sans/font-serif utilities (tailwind.config.ts) select
// between them per-element.
const inter = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" })

// Display/headings + the "Maitri" wordmark: Fraunces (variable font, optical
// sizing on by default via next/font).
const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-serif",
  weight: ["500", "600"],
  display: "swap",
})

export const metadata: Metadata = {
  // Explicit canonical base: without it Next falls back to the Vercel project
  // URL, which still resolved to the pre-rename asa-stylist.vercel.app alias —
  // og:image URLs on /look/[id] pointed at the old domain (sweep 2026-07-10, P1-8).
  metadataBase: new URL("https://stylemaitri.vercel.app"),
  title: "Style Maitri",
  description: "Style Maitri — your AI stylist for fashion discovery.",
  openGraph: {
    siteName: "Style Maitri",
    title: "Style Maitri",
    description: "Your AI stylist for weddings, sangeets & every day.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Style Maitri",
    description: "Your AI stylist for weddings, sangeets & every day.",
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
      <body className="font-sans">
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
