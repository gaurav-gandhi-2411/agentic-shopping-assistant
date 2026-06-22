# Manual UI Checklist — G8 Image UX

## Before fixes
- Product image cards had no height cap — tall images pushed content off-screen
- After image upload, user bubble showed "Styling around your uploaded photo" with NO image thumbnail

## After fixes
- Product image cards: max-h-52 prevents oversized renders; verify no card is taller than ~220px
- After image upload: user bubble shows the uploaded image as a thumbnail (h-20 rounded)
  - Thumbnail is a local object URL (never uploaded beyond the API call)
  - Thumbnail persists in the message bubble for the session lifetime

## Visual checks to perform locally
- [ ] Upload a JPEG/WebP photo — thumbnail appears in the user message bubble
- [ ] Product card images are appropriately sized (not full-screen)
- [ ] Variant chip labels show "Style 1", "Colour Palette", etc. (not "Base", "Colour story")
- [ ] No "Change brand" link in the chat header
- [ ] "for my wife" query returns women's items
