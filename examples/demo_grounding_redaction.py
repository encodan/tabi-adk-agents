"""60-second offline demo of grounding enforcement — no credentials, no LLM call.

This reproduces the exact mechanism the production runner applies before any
answer streams to a user ("the agent that won't make up numbers"):

1. Query the (mock) semantic layer — the only ground truth the agent has.
2. Take a specialist answer that mixes figures **present in the data** with a
   confident, invented "benchmark" — the failure mode that motivated the work.
3. Run the same ``enforce_grounding_redaction`` gate production uses: every
   numeric claim is classified against the query results; unsourced figures
   are redacted to ``[unverified]`` and a caveat is prepended, while grounded
   figures survive untouched.

Run from the repo root:

    python examples/demo_grounding_redaction.py

Deterministic by design — the redaction layer is pure validation logic, which
is what makes the guarantee demoable (and testable in CI) without a model in
the loop. The same gate is exercised against a live Gemini answer in the eval
harness (``evaluation/adk_bridge.py``).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

# Flat showcase layout — make the repo root importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.grounding_enforcement import (  # noqa: E402
    GROUNDING_ENFORCED_AGENTS,
    enforce_grounding_redaction,
)
from core.handoff import SpecialistRunResult  # noqa: E402
from core.response_validator import (  # noqa: E402
    GROUNDING_REDACTION_MARKER,
    ResponseValidator,
)
from tools.mock_semantic_layer import SemanticLayerTool  # noqa: E402

DIVIDER = "─" * 72


def main() -> None:
    # 1. The agent's ground truth: a real tool call against the mock layer
    #    (per-stage funnel — the pipeline_analyst bottleneck fixture).
    envelope = asyncio.run(
        SemanticLayerTool().query_metrics(
            metrics=["time_in_stage"],
            group_by=["stage_funnel__stage_name"],
        )
    )
    tool_outputs = [{"name": "query_recruitment_metrics", "response": envelope}]

    stages = {r["stage_funnel__stage_name"]: r["time_in_stage"] for r in envelope["data"]}
    print(f"Query results (the ONLY data the agent saw): {stages}")
    print(DIVIDER)

    # 2. A specialist answer mixing grounded figures (14.8, 5.4 — present in
    #    the data) with two confident inventions (21.5-day "benchmark",
    #    "1,250 candidates" — in no query result).
    answer = (
        "Technical Interview is your bottleneck at 14.8 days, versus just "
        "5.4 days in Phone Screen. Industry benchmarks complete technical "
        "stages in 21.5 days, and roughly 1,250 candidates are stalled there."
    )
    print(f"BEFORE (as generated):\n  {answer}")
    print(DIVIDER)

    # 3. The production gate, verbatim: same function, same validator, same
    #    scope constant (``pipeline_analyst`` is grounding-enforced).
    result = SpecialistRunResult(
        agent_name="pipeline_analyst", text=answer, tool_outputs=tool_outputs
    )
    host = SimpleNamespace(validator=ResponseValidator())
    enforce_grounding_redaction(result, host, "pipeline_analyst")

    print(f"AFTER (what the user gets):\n  {result.text}")
    print(DIVIDER)

    # The guarantee, asserted: inventions redacted, grounded figures intact.
    assert GROUNDING_REDACTION_MARKER in result.text, "expected a redaction"
    assert "21.5" not in result.text, "fabricated benchmark must not survive"
    assert "1,250" not in result.text, "fabricated count must not survive"
    assert "14.8" in result.text and "5.4" in result.text, "grounded figures must survive"

    # 4. Scope gate: the same answer from a non-enforced agent is untouched —
    #    enforcement is an explicit, per-agent decision, not a blanket rewrite.
    untouched = SpecialistRunResult(
        agent_name="general_analyst", text=answer, tool_outputs=tool_outputs
    )
    enforce_grounding_redaction(untouched, host, "general_analyst")
    assert untouched.text == answer
    print(
        f"Scope gate: GROUNDING_ENFORCED_AGENTS={sorted(GROUNDING_ENFORCED_AGENTS)} — "
        "the same answer from general_analyst passes through unmodified."
    )
    print("\nOK: fabricated figures redacted, grounded figures preserved.")


if __name__ == "__main__":
    main()
