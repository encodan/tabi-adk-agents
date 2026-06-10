"""Deterministic query plans — the fast-path's data layer.

Mirrors the production discipline: every plan must validate against the
semantic-layer catalog (a renamed metric breaks CI, not a customer
conversation), and every plan must actually execute end-to-end against the
layer it targets — here the synthetic mock.
"""

from __future__ import annotations

import pytest

from config import AGENT_SUB_INTENTS
from core.query_plans import QUERY_PLANS, get_query_plan, validate_query_plans
from tools.mock_semantic_layer import SemanticLayerTool

ALL_PLAN_KEYS = [
    (agent, sub_intent) for agent, by_intent in QUERY_PLANS.items() for sub_intent in by_intent
]


def test_plans_validate_against_catalog():
    assert validate_query_plans() == []


def test_lookup_hit_and_miss():
    plan = get_query_plan("pipeline_analyst", "bottleneck")
    assert plan and all("metrics" in spec for spec in plan)
    # Unknown sub-intent and None both mean "use the full reasoning lane".
    assert get_query_plan("pipeline_analyst", "no_such_intent") is None
    assert get_query_plan("pipeline_analyst", None) is None


def test_goal_attainment_has_no_fast_path():
    """goal_attainment must run the full lane so the configured-capacity
    grounding loop applies — a pre-built plan here would bypass
    get_planning_context / compute_goal_attainment."""
    for agent, sub_intents in AGENT_SUB_INTENTS.items():
        if "goal_attainment" in sub_intents:
            assert get_query_plan(agent, "goal_attainment") is None


@pytest.mark.parametrize("agent,sub_intent", ALL_PLAN_KEYS)
async def test_every_plan_executes_against_mock_layer(agent: str, sub_intent: str):
    tool = SemanticLayerTool()
    for spec in get_query_plan(agent, sub_intent) or []:
        result = await tool.query_metrics(
            metrics=spec["metrics"],
            group_by=spec.get("group_by"),
            time_granularity=spec.get("time_granularity"),
        )
        assert result["success"] and result["data"], f"{agent}/{sub_intent} spec failed: {spec}"
