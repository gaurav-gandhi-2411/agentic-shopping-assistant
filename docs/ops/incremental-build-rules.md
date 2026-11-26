# Incremental index builder: pre-registered decision rules (R3 6d)

Committed BEFORE the code that implements it. A later change to any rule below needs a commit to
this file in the same PR as the code change. Code: `src/refresh/` and `scripts/ops/incremental_build.py`.
Nothing here is wired into prod; the builder only writes a state file and a report.

## 1. Content hash and reuse

- Item key: Shopify product id (string). Hash = sha256 of the canonical JSON of
  `[title, price, available, image_url, sorted(tags)]` (title stripped; price as float).
  Description, vendor and product_type are NOT hashed (the task fixes the field list). Known
  consequence: a description-only edit does not refresh the stored text vector until another hashed
  field changes. Accepted; revisit if the dry run shows it matters.
- Same id AND same hash -> reuse the stored row, text vector and image vector. No embedding.
- Same id, different hash -> "changed": re-prepare and re-embed text. The image vector is reused
  when the image URL is unchanged, otherwise re-embedded (subject to the cap).
- Id not in the previous state -> "new": prepare, embed text, embed image (subject to the cap).
- Previous id absent from the fetched feed, or present but no longer kept (sold out, unpriced, or
  removed by the build filters) -> "dropped".
- "kept" = survives the build filters (`build_unified_index._clean_brand_catalogue` replay, same
  as `feed_delta_funnel`). Items rejected by those filters are not stored, so they are re-prepared
  (cheap, no embedding) each run and never counted as new.

## 2. Image-embedding cap

- `IMAGE_EMBED_CAP_PER_RUN = 2000` new image embeddings per run (CLI `--image-cap`).
- Work list is ordered by (store slug, article id), so the choice of what is embedded is
  deterministic. The cap is global across stores.
- Items beyond the cap are KEPT with their text vector, no image vector and `image_pending=true`.
  The next run treats a pending row as needing an image embedding even if its hash is unchanged.
  The report lists `cap_hit`, `images_embedded`, `images_deferred`, per-store deferred counts.
- An embedder returning no vector for an image (download or decode failure) is also left pending
  and counted as `images_failed`; never dropped, retried next run.
- Items with no image URL have no image vector and are not pending.

## 3. Store status and carry-over

- Only `status == "ok"` stores are rebuilt. Every other status (`rate_limited`, `capped`,
  `challenge`, `empty`, `http_error`, `robots_*`, `opted_out`, ...) carries the previous rows over
  untouched (including vectors) and is reported with its status.
- A store in the previous state but absent from the fetch results is carried over, reason
  `not_attempted`.
- Shrink guard: an `ok` store whose kept row count would fall below `SHRINK_GUARD_RATIO = 0.5` of
  its previous count (previous count >= `SHRINK_GUARD_MIN_PREVIOUS = 20`) is carried over instead,
  reason `shrink_guard`, because a silently partial feed must not delete half a store.
- A store with no previous rows and a non-ok status contributes nothing.

## 4. State file

- One file, `index-state.npz`, for a GitHub Release asset on a pre-release tag `index-state-*`.
  No GCS reads in the builder. Uses `numpy.savez_compressed`, loaded with `allow_pickle=False`.
- Header JSON: `schema_version` (int, currently 1), `built_at`, `dims`, `store_status`, `checksum`.
  Checksum = sha256 over the payload arrays (names, dtypes, shapes, bytes, in sorted-name order).
  The reader refuses an unknown schema version, wrong dims or a checksum mismatch; the builder
  treats an unreadable state as a hard error, never as "empty state".

## 5. Gender correction (build-time hygiene)

Applies to rows that are new or changed in this run (stored rows were corrected when built).

- Name gender: the FIRST word of the product name (letters only, lowercased; `Men's` -> `men`).
  men set: men, mens, man, gents, gent, male, boys, boy. women set: women, womens, woman, ladies,
  lady, female, girls, girl. Boys maps to label `men`, Girls to `women` (label domain is
  men/women/unknown; kids handling stays with `is_kids_item`). First word not in either set -> no
  check.
- Conflict: name gender G in {men, women}, current label L in {men, women}, L != G. A label of
  `unknown`/`mixed`/empty is not a conflict (a separate backfill's job).
- Resolution, in this order, first rule that applies wins:
  1. A `unisex` word in name, product_type or tags -> unchanged (ambiguous).
  2. product_type words: if it names exactly one gender: G -> correct, L -> unchanged; both ->
     unchanged.
  3. tags words, same logic over the union of all tags.
  4. No gender evidence in product_type or tags -> correct to G (the leading word decides).
- Every correction is logged `(article_id, store, name, before, after, reason)`; the report carries
  per-store correction counts and the unchanged-but-conflicting ("ambiguous") count.

### 5a. Leading-word guards (amendment, verifier finding M1)

Pre-registered before the code change. The leading-word fallback (and the correction it drives)
must not fire on names where the first word is not a reliable gender signal.

- The leading token is the first WHITESPACE-delimited word (not a letters-only regex split). After
  removing one trailing possessive (`'s` or a curly `'s`), it must be purely `a-z`. A hyphen or any
  other punctuation inside it ("Man-Made", "Woman-Owned", "Men-in-Black", "Boy-friend") means the
  name evidence is unreliable: label unchanged, outcome ambiguous (reason
  `hyphenated_leading_word`), and counted as `gender_ambiguous_hyphenated` (per store and total)
  only when the first hyphen piece is a gender word naming a gender different from the label (i.e.
  the old rule would have corrected it). Such rows are also included in `gender_ambiguous`.
- Denylist of leading two-word phrases whose first word is a gender word but not a gender signal:
  `man repeller`, `ladies man`, `boy friend`, `man made`, `girl friday`. A match stops the
  fallback (label unchanged, no conflict, reason `denylisted_phrase`). Keep this list short; add an
  entry only with a real product name as evidence.
- Product_type/tags evidence still decides first for names that pass both guards, as in section 5.
- Known residual: a name whose first word is a genuine gender word used non-semantically but not in
  the denylist is still corrected when no product_type/tags evidence exists. The correction log
  (`gender_corrections`) is the audit trail for that.
