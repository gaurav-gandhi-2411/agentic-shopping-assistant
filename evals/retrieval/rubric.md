You are grading product search results for an Indian fashion shopping assistant.
For one shopper query and a numbered list of candidate products, give EACH product a graded
relevance label for that query. Judge only what is shown (title, price, category, colour,
description). Do not reward a product for being popular or pretty; judge fit to the query.

Labels:

2 = clearly satisfies the query. Right gender (or gender-neutral when the query names none),
    right category / product type, suits the stated occasion, matches the stated colour if the
    query names one, and costs at most the stated budget if the query gives one.

1 = partially satisfies. Right gender and plausibly the right category, but at least one of:
    the colour differs from the stated colour; the price is above the budget by at most 25%;
    the occasion fit is only loose (for example everyday wear for a wedding); the title and
    description are too vague to confirm the category.

0 = does not satisfy. Any of: clearly the other gender; wrong category (not the garment or
    accessory asked for, a kids item, an unstitched fabric piece); clearly unsuitable for the
    occasion (sleepwear, swimwear for a wedding); price more than 125% of the stated budget.

Rules:
- A stated gender is a hard constraint: an item clearly for the other gender is always 0.
- A stated budget is a ceiling: up to 25% over is a 1, beyond that is a 0. Cheaper is never worse.
- A stated colour mismatch caps the label at 1 (never 2).
- If the query names no gender, items for any gender can be 2 when otherwise fitting.
- Hindi/Hinglish words mean what they mean in India (shaadi = wedding, ke liye = for, chahiye =
  want/need, se kam = less than).
- Label every number in the list. Do not skip any.

Output exactly one JSON object and nothing else:
{"labels": {"1": 2, "2": 0, ...}, "reason": "at most 15 words"}
