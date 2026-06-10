"""Canonical names for planning-derived claims.

Single source of truth used by:
  - get_planning_context / compute_goal_attainment tool docstrings
  - capacity_planner / data_scientist prompt templates
  - goal_attainment_verifier

Centralising these strings here prevents docstring-vs-template-vs-verifier
drift as the planning context grows (sourcing budgets, channel mix,
recruiter roster).
"""

from __future__ import annotations

HIRING_TARGET = "hiring_target"
RECRUITER_CAPACITY_PER_MONTH = "recruiter_capacity_per_month"
ACTIVE_RECRUITERS = "active_recruiters"
ACTUAL_YTD_HIRES = "actual_ytd_hires"
PROJECTED_FULL_YEAR_HIRES = "projected_full_year_hires"
PROJECTED_FULL_YEAR_HIRES_LOWER = "projected_full_year_hires_lower"
PROJECTED_FULL_YEAR_HIRES_UPPER = "projected_full_year_hires_upper"
HIRING_GAP = "hiring_gap"
CAPACITY_CONSTRAINED_MAX_HIRES = "capacity_constrained_max_hires"
MONTHS_ELAPSED = "months_elapsed"
MONTHS_REMAINING = "months_remaining"

# ACTUAL_YTD_HIRES is intentionally EXCLUDED — that metric comes from a
# MetricFlow query and is verified by the existing keyed_results contract.
# The goal-attainment verifier only triggers when a response contains
# numbers it can ground against PlanningContext or replay through
# compute_goal_attainment.
PLANNING_METRICS: frozenset[str] = frozenset(
    {
        HIRING_TARGET,
        RECRUITER_CAPACITY_PER_MONTH,
        ACTIVE_RECRUITERS,
        PROJECTED_FULL_YEAR_HIRES,
        PROJECTED_FULL_YEAR_HIRES_LOWER,
        PROJECTED_FULL_YEAR_HIRES_UPPER,
        HIRING_GAP,
        CAPACITY_CONSTRAINED_MAX_HIRES,
        MONTHS_ELAPSED,
        MONTHS_REMAINING,
    }
)

SQID_PREFIX_PLANNING_CONTEXT = "planning_context:"
SQID_PREFIX_COMPUTE_GOAL_ATTAINMENT = "compute_goal_attainment:"


def planning_target_sqid(role: str, year: int) -> str:
    return f"{SQID_PREFIX_PLANNING_CONTEXT}targets:{role}:{year}"


# Specialists wired with the planning tools (get_planning_context +
# compute_goal_attainment). Only these agents can produce planning claims;
# the goal-attainment verifier no-ops cleanly for everyone else, but we
# can skip the dispatch entirely for agents we know are unaffected.
PLANNING_AWARE_AGENTS: frozenset[str] = frozenset({"capacity_planner", "data_scientist"})
