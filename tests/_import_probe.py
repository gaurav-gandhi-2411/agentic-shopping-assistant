"""Imports the module named in argv[1]; used by test_db_leak_subprocess (no output of its own)."""
from __future__ import annotations

import importlib
import sys

importlib.import_module(sys.argv[1])
