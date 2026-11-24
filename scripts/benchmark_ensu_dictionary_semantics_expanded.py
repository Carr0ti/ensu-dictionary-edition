
#!/usr/bin/env python3
"""Run the existing Ensu semantic benchmark with an expanded query dataset."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent

BASE_SCRIPT = ROOT / "scripts/benchmark_ensu_dictionary_semantics.py"
EXPANDED_CASES_FILE = ROOT / "scripts/ensu_semantic_benchmark_cases.json"

EXPECTED_BASE_COUNT = 25
EXPECTED_EXPANDED_COUNT = 106
EXPECTED_TOTAL_COUNT = EXPECTED_BASE_COUNT + EXPECTED_EXPANDED_COUNT


def load_module():
    if not BASE_SCRIPT.is_file():
        raise SystemExit(f"Base benchmark script not found: {BASE_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "ensu_semantic_benchmark_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise SystemExit("Could not load the existing benchmark script")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def load_expanded_cases() -> list[dict]:
    if not EXPANDED_CASES_FILE.is_file():
        raise SystemExit(
            f"Expanded query dataset not found: {EXPANDED_CASES_FILE}"
        )

    try:
        cases = json.loads(
            EXPANDED_CASES_FILE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Invalid expanded benchmark JSON: {error}"
        ) from error

    if not isinstance(cases, list):
        raise SystemExit("Expanded benchmark dataset must be a JSON list")

    if len(cases) != EXPECTED_EXPANDED_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_EXPANDED_COUNT} expanded queries; "
            f"found {len(cases)}"
        )

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise SystemExit(f"Expanded query {index} is not an object")

        query = case.get("query")
        expected = case.get("expected")
        category = case.get("category")

        if not isinstance(query, str) or not query.strip():
            raise SystemExit(f"Expanded query {index} has no valid query")

        if (
            not isinstance(expected, list)
            or not expected
            or not all(
                isinstance(word, str) and word.strip()
                for word in expected
            )
        ):
            raise SystemExit(
                f"Expanded query {index} has invalid expected answers"
            )

        if not isinstance(category, str) or not category.strip():
            raise SystemExit(
                f"Expanded query {index} has no valid category"
            )

    return cases


def main() -> None:
    benchmark = load_module()

    base_cases = benchmark.CASES

    if len(base_cases) != EXPECTED_BASE_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_BASE_COUNT} original queries; "
            f"found {len(base_cases)}"
        )

    expanded_cases = load_expanded_cases()
    combined_cases = base_cases + expanded_cases

    if len(combined_cases) != EXPECTED_TOTAL_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_TOTAL_COUNT} total queries; "
            f"found {len(combined_cases)}"
        )

    benchmark.CASES = combined_cases

    print("=" * 68)
    print("ENSU EXPANDED SEMANTIC BENCHMARK")
    print("=" * 68)
    print(f"Original queries:  {len(base_cases)}")
    print(f"New queries:       {len(expanded_cases)}")
    print(f"Total queries:     {len(combined_cases)}")
    print()
    print("Category distribution:")

    category_counts = {}
    for case in combined_cases:
        category = case["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in sorted(category_counts.items()):
        print(f"  {category:<16} {count:>3}")

    print("=" * 68)
    print("Starting the existing benchmark implementation...")
    print("=" * 68, flush=True)

    benchmark.main()


if __name__ == "__main__":
    main()
