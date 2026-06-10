"""Make the repository root importable as the package root.

This repo uses a flat layout (`core`, `agents`, `tools`, … are top-level
packages). Inserting the repo root on ``sys.path`` lets ``tests/`` import them
regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
