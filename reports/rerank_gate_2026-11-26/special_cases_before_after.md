# Special-case fixture: top-5 titles before/after deterministic filters

Offline simulation (no live call). before = retrieval order top-5 (what the model's order or the retrieval-order fallback shows with no rules); after = production rerank() (rerank_filters.apply_rules) with a simulated model that accepts retrieval order 1..8. Source: evals/fixtures/rerank_special_cases.json, scripts/rerank_rules_report.py (RetrievalOrderModel). Not judged by the producer.

## s01: beach vacation dress under 2000
before:
  1. Black Bodycon Midi Dress | Black | dress | INR 1499.0
  2. Formal Tailored Blazer Dress | Grey | dress | INR 1899.0
  3. Beige Structured Shift Dress | Beige | dress | INR 1299.0
  4. Cotton Chinos Trousers | Khaki | trousers | INR 999.0
  5. Wool Blend Cardigan Sweater | Brown | sweater | INR 1499.0
after:
  1. White Linen Sundress | White | dress | INR 1199.0
  2. Striped Bikini Set | Blue | swimwear | INR 899.0
  3. Floral Beach Cover-up | Yellow | cover-up | INR 799.0
  4. Printed Tank Top | Coral | top | INR 499.0
  5. Cotton Board Shorts | Navy | shorts | INR 699.0

## s02: pool party outfit for women
before:
  1. Beige Sand Knit Top | Beige | top | INR 899.0
  2. Heavy Wool Sweater | Grey | sweater | INR 1599.0
  3. Slim Fit Jeans | Blue | jeans | INR 1299.0
  4. Red Bikini Top | Red | swimwear | INR 599.0
  5. Sarong Wrap Skirt | Green | skirt | INR 699.0
after:
  1. Red Bikini Top | Red | swimwear | INR 599.0
  2. Sarong Wrap Skirt | Green | skirt | INR 699.0
  3. White Cotton Sundress | White | dress | INR 999.0
  4. Printed Swimsuit | Pink | swimsuit | INR 1099.0
  5. Casual Tank Top | Yellow | top | INR 399.0

## s03: date night dress
before:
  1. Cotton Pyjama Set | Black | sleepwear | INR 799.0
  2. Beige Ribbed Midi Dress | Beige | dress | INR 1399.0
  3. Brown Knit Sweater Dress | Brown | dress | INR 1699.0
  4. Pastel Pink Skater Dress | Pastel Pink | dress | INR 1299.0
  5. Black Satin Slip Dress | Black | dress | INR 1999.0
after:
  1. Black Satin Slip Dress | Black | dress | INR 1999.0
  2. Black Bodycon Dress | Black | dress | INR 1799.0
  3. Black Velvet Wrap Dress | Black | dress | INR 2199.0
  4. Black Lace Dress | Black | dress | INR 2499.0
  5. Wine Red Satin Dress | Wine | dress | INR 2299.0

## s04: evening out outfit women
before:
  1. Camel Hooded Sweatshirt | Camel | sweatshirt | INR 1299.0
  2. Light Brown Cotton Hoodie | Light Brown | hoodie | INR 1199.0
  3. Plain White T-Shirt | White | t-shirt | INR 399.0
  4. Silk Pyjama Night Suit | Red | sleepwear | INR 1599.0
  5. Black Sequin Top | Black | top | INR 1499.0
after:
  1. Black Sequin Top | Black | top | INR 1499.0
  2. Burgundy Satin Dress | Burgundy | dress | INR 2199.0
  3. Black Co-ord Set | Black | co-ord | INR 2499.0
  4. Navy Blue Jumpsuit | Navy Blue | jumpsuit | INR 2299.0
  5. Black Palazzo Pants | Black | pants | INR 1299.0

## s05: winter wedding guest outfit
before:
  1. Cotton Shorts Set | Blue | shorts | INR 899.0
  2. Beach Bikini | Pink | swimwear | INR 799.0
  3. Open Toe Sandals | Beige | sandals | INR 999.0
  4. Beige Linen Dress | Beige | dress | INR 1699.0
  5. Brown Casual Hoodie | Brown | hoodie | INR 1499.0
after:
  1. Velvet Anarkali Suit | Maroon | suit | INR 3499.0
  2. Wool Blend Coat Dress | Black | dress | INR 3999.0
  3. Embroidered Shawl Saree | Green | saree | INR 3299.0
  4. Silk Lehenga Choli | Red | lehenga | INR 4499.0
  5. Full Sleeve Velvet Gown | Wine | gown | INR 3799.0

## s06: summer party dress
before:
  1. Heavy Wool Overcoat | Black | coat | INR 3999.0
  2. Chunky Knit Sweater | Grey | sweater | INR 1799.0
  3. Quilted Puffer Jacket | Black | jacket | INR 2999.0
  4. Cotton Sundress | Yellow | dress | INR 1299.0
  5. Linen Wrap Dress | White | dress | INR 1699.0
after:
  1. Cotton Sundress | Yellow | dress | INR 1299.0
  2. Linen Wrap Dress | White | dress | INR 1699.0
  3. Chiffon Maxi Dress | Coral | dress | INR 1999.0
  4. Floral Skater Dress | Pink | dress | INR 1499.0
  5. Printed Midi Dress | Blue | dress | INR 1599.0

## s07: monsoon kurta for women
before:
  1. Cotton Printed Kurta | Blue | kurta | INR 799.0
  2. Rayon Straight Kurta | Green | kurta | INR 899.0
  3. Quick Dry Kurta Set | Teal | kurta set | INR 1299.0
  4. Chikankari Kurta | White | kurta | INR 1499.0
  5. Anarkali Kurta | Maroon | kurta | INR 1699.0
after:
  1. Cotton Printed Kurta | Blue | kurta | INR 799.0
  2. Rayon Straight Kurta | Green | kurta | INR 899.0
  3. Quick Dry Kurta Set | Teal | kurta set | INR 1299.0
  4. Chikankari Kurta | White | kurta | INR 1499.0
  5. Anarkali Kurta | Maroon | kurta | INR 1699.0

## s08: office wear for women under 3000
before:
  1. Cotton Night Suit | Pink | sleepwear | INR 899.0
  2. Silk Pyjama Set | Blue | sleepwear | INR 1299.0
  3. Lace Lingerie Set | Black | lingerie | INR 999.0
  4. Floral Bikini | Red | swimwear | INR 799.0
  5. Tailored Blazer | Navy Blue | blazer | INR 2499.0
after:
  1. Tailored Blazer | Navy Blue | blazer | INR 2499.0
  2. Formal Trousers | Black | trousers | INR 1499.0
  3. Crepe Shirt | White | shirt | INR 1199.0
  4. Pencil Skirt | Grey | skirt | INR 1299.0
  5. Smart Casual Kurta | Blue | kurta | INR 1199.0

## s09: haldi outfit for men
before:
  1. Cotton Pyjama Night Suit | Yellow | sleepwear | INR 899.0
  2. Boxer Briefs Pack | Blue | underwear | INR 499.0
  3. Swim Shorts | Yellow | swimwear | INR 699.0
  4. Yellow Kurta Pyjama Set | Yellow | kurta set | INR 1999.0
  5. White Cotton Kurta Pajama | White | kurta set | INR 1799.0
after:
  1. Yellow Kurta Pyjama Set | Yellow | kurta set | INR 1999.0
  2. White Cotton Kurta Pajama | White | kurta set | INR 1799.0
  3. Printed Haldi Kurta | Yellow | kurta | INR 1499.0
  4. Linen Kurta Churidar | Cream | kurta set | INR 2199.0
  5. Floral Nehru Jacket | Yellow | jacket | INR 1799.0

## s10: cocktail party dress
before:
  1. Camel Sweater Dress | Camel | dress | INR 1699.0
  2. Beige Shirt Dress | Beige | dress | INR 1399.0
  3. Black Graphic Tee Dress | Black | dress | INR 899.0
  4. Black Sequin Dress | Black | dress | INR 2999.0
  5. Black Satin Dress | Black | dress | INR 2499.0
after:
  1. Black Sequin Dress | Black | dress | INR 2999.0
  2. Black Satin Dress | Black | dress | INR 2499.0
  3. Black Velvet Dress | Black | dress | INR 2799.0
  4. Black Bodycon Dress | Black | dress | INR 1999.0
  5. Red Satin Dress | Red | dress | INR 2599.0
