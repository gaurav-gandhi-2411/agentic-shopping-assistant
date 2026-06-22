"""
Eval: Category normalization precision.

Runs the GarmentNormalizer on the labeled spot-check fixture and reports:
- Precision at high-confidence
- Coverage (% items reaching high/medium confidence)
- Per-store breakdown (if store info is in fixture)
- Items that fail the precision check (with actual vs expected)

Usage:
    python -m eval.category_normalization
"""

from __future__ import annotations

import json
from pathlib import Path

from src.catalogue.normalizer import normalize_garment_type

FIXTURE_PATH = Path("evals/fixtures/category_spotcheck.json")


def main() -> None:
    """Run category normalization precision eval and print a report."""
    items = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    total = len(items)
    correct = 0
    high_conf_total = 0
    high_conf_correct = 0
    failures: list[dict[str, str | None]] = []

    for item in items:
        result = normalize_garment_type(
            prod_name=item["prod_name"],
            product_type_name=item.get("product_type_name"),
            brand=item.get("brand"),
        )
        expected_gt = item["expected_garment_type"]
        is_correct = result.garment_type == expected_gt
        if is_correct:
            correct += 1

        if result.type_confidence in ("high", "medium"):
            high_conf_total += 1
            if is_correct:
                high_conf_correct += 1
            else:
                failures.append(
                    {
                        "prod_name": item["prod_name"],
                        "brand": item.get("brand"),
                        "store_label": item.get("product_type_name"),
                        "expected": expected_gt,
                        "got": result.garment_type,
                        "confidence": result.type_confidence,
                        "note": item.get("note", ""),
                    }
                )

    precision = high_conf_correct / high_conf_total if high_conf_total else 0.0
    coverage = high_conf_total / total if total else 0.0

    print("\n=== Category Normalization Precision ===")
    print(f"Total items in fixture:        {total}")
    print(f"High/medium confidence:        {high_conf_total} ({coverage:.1%} coverage)")
    print(f"Correct at high/medium:        {high_conf_correct}")
    print(f"Precision at high/medium conf: {precision:.1%}")
    print()

    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(
                f"  [{f['confidence']}] '{f['prod_name']}' "
                f"(brand={f['brand']}, store_label={f['store_label']})"
            )
            print(f"       expected={f['expected']} | got={f['got']} | {f['note']}")
    else:
        print("All high/medium-confidence items classified correctly.")


if __name__ == "__main__":
    main()
