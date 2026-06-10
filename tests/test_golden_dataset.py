"""The golden dataset ships with this repo and must stay loadable + well-formed.

The README's Agent Simulation section quotes concrete counts; these tests pin
them so the documented numbers can't silently drift from the committed
artifact. Counts are exact (not ``>=``) on purpose: editing the goldens means
consciously updating both this file and the README in the same change.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from config import SPECIALIST_AGENTS
from evaluation.golden_dataset import GoldenExample, load_golden_datasets

GOLDENS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "goldens"

EXPECTED_TOTAL = 103
EXPECTED_BY_CATEGORY = {
    # 35 single-agent routes + 10 multi-agent compound questions — both are
    # routing decisions, so multi_agent.yaml entries carry category=routing.
    "routing": 45,
    "factuality": 18,
    "tool_usage": 15,
    "safety": 10,
    "multi_turn": 15,
}


@pytest.fixture(scope="module")
def goldens() -> list[GoldenExample]:
    return load_golden_datasets(GOLDENS_DIR)


def test_goldens_match_documented_counts(goldens: list[GoldenExample]) -> None:
    assert len(goldens) == EXPECTED_TOTAL, (
        f"Golden count drifted: {len(goldens)} != {EXPECTED_TOTAL}. "
        "Update EXPECTED_* here AND the Agent Simulation section of README.md."
    )
    assert Counter(e.category for e in goldens) == EXPECTED_BY_CATEGORY


def test_golden_ids_are_unique(goldens: list[GoldenExample]) -> None:
    ids = [e.id for e in goldens]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    assert not dupes, f"Duplicate golden ids: {dupes}"


def test_expected_agents_are_real_specialists(goldens: list[GoldenExample]) -> None:
    """Every routing expectation must point at an agent that exists, so a
    specialist rename can't orphan goldens silently."""
    known = set(SPECIALIST_AGENTS)
    for example in goldens:
        for agent in filter(None, [example.expected_agent]):
            assert agent in known, f"{example.id}: unknown expected_agent {agent!r}"
        for agent in example.expected_agents or []:
            assert agent in known, f"{example.id}: unknown expected_agents entry {agent!r}"
        for agent in example.acceptable_agents or []:
            assert agent in known, f"{example.id}: unknown acceptable_agents entry {agent!r}"


def test_safety_goldens_assert_non_leakage(goldens: list[GoldenExample]) -> None:
    """Safety goldens are simulation probes (PII requests, injection attempts);
    each must encode at least one forbidden output so a regression is
    detectable, not just narrated."""
    for example in (e for e in goldens if e.category == "safety"):
        assert example.must_not_contain, (
            f"{example.id}: safety golden carries no must_not_contain assertions"
        )
