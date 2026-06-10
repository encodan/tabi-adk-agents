"""The README points readers at ``examples/demo_grounding_redaction.py`` as the
60-second offline proof of grounding enforcement. Run it in CI so the promise
can't rot: the script asserts its own outcome (fabricated figures redacted,
grounded figures preserved, scope gate respected) and raises on any drift in
validator tolerances or the enforcement seam."""

from __future__ import annotations

import runpy
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo_grounding_redaction.py"


def test_grounding_demo_runs_and_self_verifies(capsys) -> None:
    runpy.run_path(str(DEMO), run_name="__main__")
    out = capsys.readouterr().out
    assert "[unverified]" in out
    assert "OK: fabricated figures redacted, grounded figures preserved." in out
