# Reranker gate (production rerank() vs eval variant C)

n LLM queries 53, n with defined nDCG 49; excluded: {"pool_le_top_k_or_empty": 7, "ndcg_undefined_all_zero_pool": ["q09", "q22", "q29", "q35"]}
nDCG@5 production 0.890 vs eval C 0.864; diff +0.026 [-0.007, +0.076] (paired bootstrap, 10000 resamples, seed 42); need lower bound >= -0.03 -> PASS
outcomes {"ok": 41, "partial": 10, "empty_selection": 2}; length hits 0; paths {"primary": 53}
failure rate (parse_error+length+http_error+budget) 0.000 -> PASS (<= 0.05)
hard-rule violations in shown top-5: {"production": {"violations": 0, "queries_with_violation": 0}, "eval_c_no_rules": {"violations": 1, "queries_with_violation": 1}, "eval_a_old_master": {"violations": 1, "queries_with_violation": 1}}
tokens/turn 745; latency p50 700 ms, p95 901 ms
Labels are LLM consensus (three model families), not human ground truth.

## Chip queries (live, unlabelled: review the shown titles)
- 'Sangeet outfit under ₹3,000': partial, tok 722+56, 422 ms -> ['Lehariya Printed Sequin Embellished Straight Kurta with Palazzo & Dupatta - Blue (Blue kurta)', 'PAISELY IVORY SHERWANI SET (Off White sherwani)', "Men's Black Rayon Sherwani Kurta Pyjama And Dupata Set (Black kurta)", 'Cream Silk Blend Kurta Sherwani Set with Jacquard Indo-Western Jacket and Trousers – Festive & Wedding Ensemble (Off White sherwani)', 'MLS KURTA JAWAHAR SET 3 PCS ( kurta)']
- 'Haldi kurta set': ok, tok 688+138, 298 ms -> ['Yellow Geometric Printed Kurta Set (Yellow kurta)', 'Yellow Printed Front Open Kurta Set (Yellow kurta)', 'White Cotton Kurta Set with Mustard Ikat-Inspired Prints and Cream Trousers – Festive & Wedding Ensemble (Yellow kurta)', 'Yellow Self Design Regular Fit Kurta Set | Azania (Yellow kurta)', 'Haldi Yellow Printed Silk Blend Kurta | The Kurta Company (Yellow kurta)']
- 'Wedding guest saree': ok, tok 530+142, 439 ms -> ['Pink Embellished Satin Saree (Pink saree)', 'Dityaa Gold (Saree) (Red saree)', 'Burgundy Silk Blend Saree (Dark Red saree)', 'Navy Georgette Saree With Floral Embellished Pallu (Dark Blue saree)', 'Lavender Solid Chiffon Saree (Purple saree)']
- 'Office wear under ₹1,500': ok, tok 672+103, 267 ms -> ['Light Blue Checked Slim Fit Formal Shirt | Metal (Light Blue shirt)', 'Peacock Blue Solid Slim Fit Formal Shirt | Metal (Blue shirt)', 'White Solid Slim Fit Formal Shirt | Metal (White shirt)', 'Mid Grey Washed Denim Shirt (Grey shirt)', 'Grey Sky Zipper Neck T-Shirt (Grey top)']
- "Men's kurta for a wedding": ok, tok 636+149, 679 ms -> ['Men Black Silk Embroidered Kurta (Black kurta)', "Maroon with Circular Bandhani Print Cotton Blend Men's Kurta (Dark Red kurta)", 'Mens Black Round Nack Solid Kurta (Black kurta)', "Plus Men's Black Cotton Linen Kurta (Black kurta)", 'Mens Off White Round Nack Printed Kurta (Off White kurta)']
- 'sherwani for groom': ok, tok 384+158, 613 ms -> ['Cream Embroidered Regular Fit Sherwani | TULA (Off White sherwani)', 'Off-White Sherwani | Azania (Off White sherwani)', 'White Sherwani | TULA (White sherwani)', 'Mint Green Sherwani | TULA (Green sherwani)', 'MLS SHERWANI 4PCS ( sherwani)']
- 'lehenga for sangeet': ok, tok 434+149, 636 ms -> ['Set of 3 - Berry Blue Floral Handblock Cotton Lehenga Set (Blue lehenga)', 'Aadrika Lehenga (Mustard lehenga)', 'Set of 3 Ajrakh Printed Red Lehenga Set (Red lehenga)', 'Aparna Blue (Lehenga) (Blue lehenga)', 'Aparna Yellow (Lehenga) (Yellow lehenga)']
- "men's blazer for the office": ok, tok 462+154, 625 ms -> ['White Solid Blazer | Wyre (White blazer)', "Men's's Navy Blue Solid Lachka Imported Suiting Blazer (Dark Blue blazer)", "Men's Black Silk Blend Blazer (Black blazer)", "Men's Black Solid Imported Suiting Blazer (Black blazer)", 'Black checked Regular Fit Blazer | JB X JB (Black blazer)']
