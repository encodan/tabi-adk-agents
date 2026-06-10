"""Golden dataset schema and loader for automated evaluation.

Provides the GoldenExample dataclass and YAML loading utilities for the
evaluation framework. Golden examples define expected outcomes (correct route,
tools, metrics, answer properties) for automated correctness testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GoldenExample:
    """A single evaluation example with expected outcomes."""

    # Identity
    id: str
    category: str  # "routing" | "factuality" | "tool_usage" | "safety"
    difficulty: str  # "easy" | "medium" | "hard"

    # Input
    question: str
    context: dict[str, Any] | None = None

    # Expected routing
    expected_agent: str | None = None
    expected_sub_intent: str | None = None
    expected_agents: list[str] | None = None  # For multi-agent questions
    # Ambiguous-routing helper: the classifier may land on ANY of these agents;
    # `eval_route_correctness` passes if the actual agent is in the set. Use
    # this for the `routing/ambiguous` edge-case cell.
    acceptable_agents: list[str] | None = None

    # Expected tool usage
    expected_tools: list[str] | None = None
    expected_metrics: list[str] | None = None
    # `eval_query_plan` passes if any inner list is a subset of the agent's
    # requested metrics. `expected_numbers` must carry the same numeric value
    # under each inner list's metric key so the factuality evaluator matches
    # against whichever shape the agent picked.
    acceptable_metric_sets: list[list[str]] | None = None
    forbidden_tools: list[str] | None = None

    # Expected answer properties
    reference_answer: str | None = None
    must_contain: list[str] | None = None
    must_not_contain: list[str] | None = None

    # Data grounding
    mock_query_results: list[dict[str, Any]] | None = None
    expected_numbers: dict[str, float] | None = None

    # Entity extraction / deterministic query-plan shape. Used by
    # `eval_query_plan` to verify the executed queries applied the right
    # filters and time ranges. Keys in use: `department`, `source`,
    # `job_name`, `time_range`. Additional keys are permitted.
    expected_entities: dict[str, Any] | None = None

    # Multi-turn conversation sequence. When set, `eval_multi_turn` takes over
    # and the single-turn fields above are ignored. Each turn has shape:
    #   {"question": str,
    #    "expected_agent": str?,
    #    "expected_filters": list[dict]?,
    #    "must_contain": list[str]?,
    #    "must_not_contain": list[str]?}
    turns: list[dict[str, Any]] | None = None

    # Expert pre-judgment for the LLM-as-Judge alignment workflow.
    # Values: "pass" | "fail" | None. Used to record
    # the domain expert's verdict during critique shadowing and to spot
    # judge↔expert disagreements in nightly runs.
    judge_verdict: str | None = None

    # "Should there be a chart?" override for the ChartIntent judge.
    # When ``None`` the runner reads ``route_result.chart_likely``;
    # set this to encode intent the router might mis-classify
    # (e.g. an unambiguous scalar question on a borderline sub_intent).
    expected_chart_likely: bool | None = None


def load_golden_dataset(path: str | Path) -> list[GoldenExample]:
    """Load golden examples from a YAML file.

    Args:
        path: Path to a YAML file containing a list of example dicts.

    Returns:
        List of GoldenExample instances.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML content is not a list.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of examples in {path}, got {type(raw).__name__}")

    examples: list[GoldenExample] = []
    for item in raw:
        examples.append(GoldenExample(**item))
    return examples


def load_golden_datasets(directory: str | Path) -> list[GoldenExample]:
    """Load all golden examples from YAML files in a directory.

    Args:
        directory: Path to a directory containing .yaml files.

    Returns:
        Combined list of GoldenExample instances from all files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Golden dataset directory not found: {directory}")

    all_examples: list[GoldenExample] = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        all_examples.extend(load_golden_dataset(yaml_file))
    return all_examples
