# V2 landing page screenshots (Wave 1b)

All images are 390x844 CSS px at 2x (780x1688 px), Chromium touch emulation, taken from a
LOCAL production build with a STUBBED API (Playwright route interception and a mocked
WebSocket). They are NOT production and show canned sample data, not real catalogue results.

| File | State |
|---|---|
| `before-root-390.png` | Before: `/` redirected anonymous visitors to `/login` (origin/master 4b4fb4e), with the Google button |
| `after-root-390.png` | After: `/` first paint: promise line, 5 occasion chips (44 px tall), input box, all above the fold |
| `after-slow-start-390.png` | After: session held for 15 s; a chip was tapped (queued); skeleton and the "waking up, about a minute" message at about 4 s |
| `after-limit-390.png` | After: `rate_limited` frame ("Retry in 720s") shown as "try again in about 12 minutes" with a retry button |
| `after-answer-390.png` | After: one chip tap streamed a canned answer and one item card (the user bubble is scrolled under the header) |
| `ver-answer-outfit-390.png` | After: canned outfit board with the "Save look" button (verifier harness) |
| `ver-look-saved-390.png` | After: look saved; the optional "Signing in is optional. Sign in" link on one line (verifier harness) |
