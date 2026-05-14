import * as Sentry from "@sentry/nextjs"

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  tracesSampleRate: 0.1,

  // Replays disabled for v1.
  replaysSessionSampleRate: 0,
  replaysOnErrorSampleRate: 0,

  beforeSend(event) {
    // Mask any captured request body so user messages never reach Sentry.
    if (event.request?.data) {
      event.request.data = "[REDACTED]"
    }
    return event
  },
})
