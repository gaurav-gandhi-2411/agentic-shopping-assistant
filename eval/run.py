"""Entry point for ``python -m eval.run --provider <provider>``."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so scripts.eval_harness can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_harness import main  # noqa: E402

if __name__ == "__main__":
    main()
