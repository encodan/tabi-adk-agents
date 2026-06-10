"""Deterministic query plans for known routing sub-intents.

This is the fast-path's data layer: when the router classifies a question into
a known ``(agent, sub_intent)`` pair with high confidence, the orchestration
layer skips LLM query planning entirely and executes the pre-built plan below
— one of the two lanes in the architecture diagram ("deterministic fast-path"
vs. the full ADK ``Runner.run_async`` reasoning loop). Each plan is the list
of semantic-layer queries needed to answer that intent class, in the exact
schema ``query_multiple_recruitment_metrics`` accepts, so the specialist
receives grounded data without spending an LLM round-trip deciding what to
query.

[public-repo genericization] The production module maps every routable
``(agent, sub_intent)`` pair to multi-query plans over 50+ MetricFlow metrics
and validates them against dbt-generated metadata; those mappings encode
observed LLM query patterns and per-metric tuning, and are excluded as
proprietary. The plans below are a *representative* subset rebuilt over the
synthetic catalog in :mod:`tools.mock_semantic_layer`, preserving the shape,
the lookup API, and the validate-at-test-time discipline (every metric and
group-by in a plan must exist in the catalog — see
``tests/test_query_plans.py``).
"""

from __future__ import annotations

from typing import Any

# Type alias for a single query specification.
# Matches the schema accepted by query_multiple_recruitment_metrics.
QuerySpec = dict[str, Any]

# Canonical stage group-by — pairs every stage-level plan with the funnel
# dimension the mock fixtures cover (production additionally carries a
# canonical-order companion column for deterministic chart sorting).
_STAGE_BY_NAME = ["stage_funnel__stage_name"]
_SOURCE_BY_NAME = ["application__source_name"]

# Query plans: agent_name -> sub_intent -> list of query specs.
QUERY_PLANS: dict[str, dict[str, list[QuerySpec]]] = {
    "pipeline_analyst": {
        "bottleneck": [
            # Per-stage dwell time + conversion — the bottleneck signal pair.
            {
                "metrics": ["time_in_stage", "pipeline_conversion_rate"],
                "group_by": _STAGE_BY_NAME,
            },
            # End-to-end timing context for the narrative.
            {"metrics": ["time_to_hire"]},
            # Volume context so "stalled" reads against pipeline size.
            {"metrics": ["applications", "hires"]},
        ],
        "recruitment_funnel": [
            {"metrics": ["pipeline_conversion_rate"], "group_by": _STAGE_BY_NAME},
            {"metrics": ["applications", "hires"]},
            {"metrics": ["time_to_hire"]},
        ],
        "pipeline_health": [
            {"metrics": ["time_in_stage"], "group_by": _STAGE_BY_NAME},
            {"metrics": ["time_to_hire", "interviews_per_hire"]},
            {"metrics": ["applications", "hires"]},
        ],
    },
    "general_analyst": {
        "overview": [
            {"metrics": ["applications", "hires"]},
            {"metrics": ["time_to_hire", "offer_acceptance_rate"]},
        ],
        "trends": [
            {"metrics": ["applications", "hires"], "time_granularity": "month"},
            {"metrics": ["time_to_hire"], "time_granularity": "month"},
        ],
    },
    "sourcing_strategist": {
        "source_comparison": [
            {"metrics": ["hires", "offer_acceptance_rate"], "group_by": _SOURCE_BY_NAME},
            {"metrics": ["applications"]},
        ],
        "source_quality": [
            {"metrics": ["offer_acceptance_rate", "hires"], "group_by": _SOURCE_BY_NAME},
            {"metrics": ["interviews_per_hire"]},
        ],
    },
    "offer_advisor": {
        "acceptance_analysis": [
            {"metrics": ["offer_acceptance_rate"]},
            {"metrics": ["offer_acceptance_rate"], "group_by": _SOURCE_BY_NAME},
            {"metrics": ["time_to_hire"]},
        ],
    },
    "capacity_planner": {
        "velocity": [
            {"metrics": ["hires", "recruiter_capacity"]},
            {"metrics": ["time_to_hire"]},
        ],
        # NOTE: ``goal_attainment`` deliberately has NO pre-built plan — it
        # must flow through the full reasoning lane so ``get_planning_context``
        # / ``compute_goal_attainment`` run and the configured-capacity
        # grounding loop (core/goal_attainment_retry.py) applies.
    },
}


def get_query_plan(agent_name: str, sub_intent: str | None) -> list[QuerySpec] | None:
    """Return the pre-built plan for ``(agent, sub_intent)``, or ``None``.

    ``None`` means "no deterministic plan" — the orchestration layer falls
    back to the full ADK reasoning loop, where the specialist plans its own
    queries via tools. Returning ``None`` (rather than an empty plan) keeps
    the two lanes mutually exclusive and observable: the trace's query-plan
    panel is populated on this lane and empty on the reasoning lane.
    """
    if sub_intent is None:
        return None
    return QUERY_PLANS.get(agent_name, {}).get(sub_intent)


def validate_query_plans() -> list[str]:
    """Cross-check every plan against the semantic-layer catalog.

    Returns a list of human-readable issues (empty == valid). Production runs
    the same check against dbt-generated MetricFlow metadata so a renamed
    metric breaks CI, not a customer conversation; here the catalog is the
    synthetic one in :mod:`tools.mock_semantic_layer`.
    """
    from config import AGENT_SUB_INTENTS, GROUPING_DIMENSIONS
    from tools.mock_semantic_layer import SYNTHETIC_METRICS

    known_metrics = set(SYNTHETIC_METRICS)
    known_dimensions = set(GROUPING_DIMENSIONS)

    issues: list[str] = []
    for agent, by_intent in QUERY_PLANS.items():
        if agent not in AGENT_SUB_INTENTS:
            issues.append(f"{agent}: not a routable agent")
            continue
        for sub_intent, specs in by_intent.items():
            if sub_intent not in AGENT_SUB_INTENTS[agent]:
                issues.append(f"{agent}/{sub_intent}: unknown sub_intent")
            for spec in specs:
                for m in spec.get("metrics", []):
                    if m not in known_metrics:
                        issues.append(f"{agent}/{sub_intent}: unknown metric {m!r}")
                for dim in spec.get("group_by", []) or []:
                    if dim not in known_dimensions:
                        issues.append(f"{agent}/{sub_intent}: unknown dimension {dim!r}")
    return issues
