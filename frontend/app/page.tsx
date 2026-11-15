import type { Metadata } from "next"
import { AssistantLanding } from "@/components/landing/AssistantLanding"
import { COPY } from "@/lib/copy"

// `/` is the working assistant for anonymous visitors (Wave 1b V2). Signed-in users are
// sent to /chat by middleware.ts. Statically rendered: the promise line and chips are in
// the first HTML, so they paint before any JavaScript runs.
//
// metadataBase comes from app/layout.tsx. The image is the existing site-wide card
// (app/opengraph-image.tsx), referenced explicitly so Twitter gets it as well.
export const metadata: Metadata = {
  title: COPY.meta.title,
  description: COPY.meta.description,
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: "/",
    siteName: "Style Maitri",
    title: COPY.meta.ogTitle,
    description: COPY.meta.ogDescription,
    images: [
      { url: "/opengraph-image", width: 1200, height: 630, alt: COPY.meta.ogImageAlt },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: COPY.meta.twitterTitle,
    description: COPY.meta.twitterDescription,
    images: ["/opengraph-image"],
  },
}

export default function RootPage() {
  return <AssistantLanding />
}
