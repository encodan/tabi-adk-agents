"""Pydantic models for tenant planning context.

Type contract between the API layer (loader), session plumbing, the
``get_planning_context`` / ``compute_goal_attainment`` tools, the verifier,
and the query-plan augmentation pipeline. Each ``PlanningContext`` is
scoped to a single year; the loader returns ``dict[int, PlanningContext]``
covering every configured year, pre-bound to ``ToolContext.planning_contexts``
at session create.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HiringTarget(BaseModel):
    """A single role-scoped hiring target for a given year."""

    role_label: str
    count: int = Field(gt=0)
    job_name_filter: list[str] = Field(default_factory=list)


class PlanningContext(BaseModel):
    """Tenant-scoped planning state surfaced to capacity-aware specialists.

    Validation:
      - ``recruiter_capacity_per_month`` must be strictly positive when set —
        a zero/negative rate would silently neutralise capacity-constrained
        projections.
      - ``active_recruiters`` must be at least 1 when set — zero would make
        capacity_constrained_max_hires equal actual_ytd_hires (zero remaining
        capacity), silently producing a misleading ceiling.
    """

    year: int = Field(ge=1970, le=2100)
    targets: list[HiringTarget] = Field(default_factory=list)
    recruiter_capacity_per_month: float | None = Field(default=None, gt=0)
    active_recruiters: int | None = Field(default=None, ge=1)
    source: str = "tenant_config"
