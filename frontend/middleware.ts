import { createServerClient, type CookieOptions } from "@supabase/ssr"
import { NextResponse, type NextRequest } from "next/server"

export async function middleware(request: NextRequest) {
  // Create a mutable response we can attach refreshed cookies to.
  // Re-assigned inside setAll when Supabase rotates the session token.
  let supabaseResponse = NextResponse.next({ request })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet: { name: string; value: string; options: CookieOptions }[]) {
          // Forward new cookies onto the outgoing request so downstream
          // server components read the refreshed session.
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          // Recreate the response so it carries the updated request cookies.
          supabaseResponse = NextResponse.next({ request })
          // Also set them on the response so the browser stores them.
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  // IMPORTANT: call getUser() immediately after createServerClient with no
  // code in between. Any await between creation and getUser() can cause
  // session-refresh race conditions that randomly log users out.
  const {
    data: { user },
  } = await supabase.auth.getUser()

  const { pathname } = request.nextUrl

  // Unauthenticated users may only access /login, /auth/*, /demo/*, /embed/*, /pdp-demo/*, /look/*.
  // /embed and /pdp-demo are public-facing widget routes — they bootstrap their
  // own anonymous demo session via POST /demo/session (same as /demo).
  // /look/[id] is a capability URL (unguessable UUID) for sharing saved looks —
  // it must be publicly viewable without login, like a shared link.
  if (
    !user &&
    !pathname.startsWith("/login") &&
    !pathname.startsWith("/auth") &&
    !pathname.startsWith("/demo") &&
    !pathname.startsWith("/embed") &&
    !pathname.startsWith("/pdp-demo") &&
    !pathname.startsWith("/look") &&
    // Site-wide OG card (app/opengraph-image.tsx): fetched by link-preview
    // crawlers (WhatsApp/Twitter) with no session — gating it behind auth
    // serves them the login page instead of the image.
    !pathname.startsWith("/opengraph-image")
  ) {
    const url = request.nextUrl.clone()
    url.pathname = "/login"
    return NextResponse.redirect(url)
  }

  // Authenticated users landing on /login go straight to /chat.
  if (user && pathname === "/login") {
    const url = request.nextUrl.clone()
    url.pathname = "/chat"
    return NextResponse.redirect(url)
  }

  // Return supabaseResponse (not a freshly constructed NextResponse.next())
  // so the refreshed session cookies are forwarded to the browser.
  return supabaseResponse
}

export const config = {
  matcher: [
    // Skip Next.js internals, static files, root-level JS/text/icon assets, and common image extensions.
    // widget.js must be excluded so anonymous users can load the embed loader script (HTTP 200)
    // without hitting the auth redirect. The pattern also covers any other root-level .js/.txt/.ico
    // files that should be publicly accessible without authentication.
    "/((?!_next/static|_next/image|favicon\\.ico|.*\\.(?:js|txt|ico|svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
}
