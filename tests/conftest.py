from __future__ import annotations

from pathlib import Path

import pytest

_INDEX_SENTINEL = Path("data/processed/dense.faiss")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Auto-skip requires_index tests when the FAISS index has not been built.

    Keeps `pytest -m "not requires_ollama"` clean on a fresh checkout:
    tests that need pre-built indices skip with a descriptive message instead
    of erroring with a FileNotFoundError or FAISS RuntimeError.
    """
    if item.get_closest_marker("requires_index") and not _INDEX_SENTINEL.exists():
        pytest.skip(
            "requires pre-built retrieval index — "
            "run `python scripts/01_build_retrieval.py` first"
        )
