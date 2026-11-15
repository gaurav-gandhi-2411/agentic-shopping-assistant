// ALL user-visible marketing and state copy for the public landing page (`/`) lives in
// this one file, so GG can review and change every public string in one place.
// Rules for anything added here: neutral and accurate; no counts, accuracy figures,
// superlatives, or claims about prices, stock or availability.
//
// Status: PROPOSED wording (Wave 1b V2). Strings marked "reused" already appear on the
// live site today; the rest are new and need GG's approval before the PR is merged.

/** Occasion chips: each label is also the exact query sent when it is tapped. 4-6 chips. */
export const OCCASION_CHIPS = [
  "Sangeet outfit under ₹3,000",
  "Haldi kurta set",
  "Wedding guest saree",
  "Office wear under ₹1,500",
  "Men's kurta for a wedding",
] as const

export const COPY = {
  meta: {
    // Page <title> and meta description for `/`.
    title: "Style Maitri — your AI stylist", // reused: og image alt text
    description: "Your AI stylist for weddings, sangeets & every day.", // reused: login + og
    ogTitle: "Style Maitri — your AI stylist",
    ogDescription: "Your AI stylist for weddings, sangeets & every day.", // reused
    twitterTitle: "Style Maitri — your AI stylist",
    twitterDescription: "Your AI stylist for weddings, sangeets & every day.", // reused
    ogImageAlt: "Style Maitri — your AI stylist", // reused: app/opengraph-image.tsx alt
  },
  landing: {
    /** The one-line promise above the chips. */
    promise: "Your AI stylist for weddings, sangeets & every day.", // reused (login page)
    chipsHeading: "Try one of these",
    noSignIn: "No sign-in needed.",
    savedLooksLink: "Saved looks", // reused (demo chat header)
    inputLabel: "Ask the stylist",
    queuedPhoto: "Photo", // label of a queued photo upload while the stylist wakes up
  },
  slowStart: {
    // Shown when the API has not answered within 4 s (cold start can take about a minute).
    title: "The stylist is waking up",
    message: "This can take about a minute. Please keep this page open.",
    skeletonLabel: "Loading the stylist",
  },
  notices: {
    retryButton: "Try again",
    /** `when` is e.g. "about 12 minutes" (see formatRetryAfter) or null when unknown. */
    rateLimited: (when: string | null): string =>
      `You've sent a lot of requests in a short time. Please try again ${
        when ? `in ${when}` : "in a little while"
      }.`,
    demoLimit: "The daily demo limit has been reached. Please try again tomorrow.",
    busy: "The stylist is busy right now. Please try again in about a minute.",
    sessionError:
      "We couldn't reach the stylist. Please check your connection and try again.",
  },
  save: {
    // Optional sign-in, offered only after a look has been saved (non-blocking link).
    signInNudge: "Signing in is optional.",
    signInLink: "Sign in",
  },
  login: {
    // /login description when the Google button is hidden (flag not set).
    descriptionEmailOnly: "Enter your email for a magic link.",
  },
} as const

export type OccasionChip = (typeof OCCASION_CHIPS)[number]
