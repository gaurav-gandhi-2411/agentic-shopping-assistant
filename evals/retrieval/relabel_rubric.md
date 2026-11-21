You are auditing relevance labels for an Indian fashion shopping assistant.
For each shopper query you get a short numbered list of retrieved products. Label EACH product
1 (relevant) or 0 (irrelevant) for that query, judging only what is shown (title, price, colour,
product type, description). Do not reward a product for being popular or pretty.

An item is RELEVANT (1) only if ALL of the following hold, judged from the product name and
description, not from your guess about the retrieval system:
1. Garment TYPE matches exactly (a kurta is not a kurta-set; a gown is a dress only if the query
   did not name a different silhouette).
2. SILHOUETTE/STYLE matches when the query names one (bodycon is not maxi, A-line is not
   straight, slim-fit is not oversized). "Unstated in the product name AND not contradicted"
   counts as a match only when the name/description gives no opposing silhouette signal.
3. GENDER matches the shopper gender given with the query (unisex is acceptable for
   unisex-plausible garments). A title or description that says Men's/Women's decides it.
4. COLOUR matches when the query names one (colour-family tolerance: wine/maroon/burgundy are one
   family; black is not navy).
5. OCCASION REGISTER is plausible when the query names an occasion, and not only "ethnic or
   western": formality and embellishment must fit. A heavy embellished, velvet or bridal-weight
   item is a MISS for haldi or mehendi (want light, floral, cotton); a plain cotton item is a
   MISS for sangeet or reception (want embellished). A thermal pyjama is never sangeet-wear.
6. BUDGET is respected when the query names one (price at most the stated cap).
Anything else, including "close but a different silhouette", is a MISS (0).

For every 0, give ONE reason from this list:
  type-confusion          wrong garment type (trousers for "blazer")
  set-not-single          multi-piece set returned for a single-garment query
  kids-leak               girls/kids item
  budget                  price above the query's stated cap
  occasion-register       garment type fine but register wrong for the stated occasion
  attribute-contradiction a named attribute is contradicted (mid-rise for "high waisted")
  data-mislabel           title and description disagree (jeans named, pyjama described)
  data-ceiling            no better item can exist (use only when nothing else fits)

Rules:
- Label every number in the list. Do not skip any.
- A stated gender, colour and budget are hard requirements (see 3, 4 and 6).
- Hindi/Hinglish words mean what they mean in India (shaadi = wedding, ke liye = for).

Output exactly one JSON object and nothing else:
{"labels": {"1": 1, "2": 0, ...}, "reasons": {"2": "set-not-single", ...}}
"reasons" has an entry for every item labelled 0 and none for items labelled 1.
