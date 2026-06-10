"""ADK tools and turn-scoped state for tenant planning context.

Houses three concerns:

1. ``TurnPlanningState`` + module-level ``ContextVar`` helpers — turn-scoped
   state (issued context tokens, recorded compute invocations, Reflexion
   retry budget). Matches the established pattern in
   :mod:`tools.adk_tools` (``_turn_query_results``): a
   token-based ContextVar bound at turn start by
   :meth:`AgentSession._bind_turn_scope`, ensuring a leaked task cannot
   smuggle stale turn state into a later turn.

2. ``get_planning_context`` — read-only tool: returns the tenant's targets
   and recruiter capacity for a given year, plus an opaque ``context_token``
   that downstream compute calls in the same turn must present.

3. ``compute_goal_attainment`` — pure deterministic math (kernel) wrapped
   with a token gate and a per-call ``source_query_id``. The verifier
   imports the kernel directly to replay recorded calls without re-entering
   the token registry.
"""

from __future__ import annotations

import contextvars
import math
import secrets
from dataclasses import dataclass, field
from typing import Any

import structlog

from core.planning_claims import SQID_PREFIX_COMPUTE_GOAL_ATTAINMENT
from tools.tool_context import (
    GoalAttainmentInvocation,
    get_tool_context,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Turn-scoped state
# ---------------------------------------------------------------------------


@dataclass
class TurnPlanningState:
    """All turn-scoped planning state, held on a ContextVar.

    One instance is created at turn start by ``set_turn_planning_state``,
    set on the ContextVar, and discarded when the token is reset at turn
    end. Tools mutate the live instance via ``get_turn_planning_state``.

    ``last_verifier_outcomes`` is written by the goal-attainment verifier
    on each run so the retry evaluator can render the critique without
    re-parsing structured logs.
    """

    context_tokens: set[str] = field(default_factory=set)
    recorded_goal_attainment_calls: list[GoalAttainmentInvocation] = field(default_factory=list)
    retry_budget: int = 1
    last_verifier_outcomes: list[tuple[str, str]] = field(default_factory=list)


_turn_planning_state: contextvars.ContextVar[TurnPlanningState | None] = contextvars.ContextVar(
    "_turn_planning_state", default=None
)


def set_turn_planning_state() -> contextvars.Token[TurnPlanningState | None]:
    """Bind a fresh ``TurnPlanningState`` at turn start.

    Returns a token to pass to :func:`reset_turn_planning_state` at turn end.
    """
    return _turn_planning_state.set(TurnPlanningState())


def reset_turn_planning_state(token: contextvars.Token[TurnPlanningState | None]) -> None:
    """Restore the binding set by :func:`set_turn_planning_state`."""
    _turn_planning_state.reset(token)


def get_turn_planning_state() -> TurnPlanningState:
    """Read the active turn's planning state.

    Raises ``RuntimeError`` when no turn is bound. The planning tools are
    only ever invoked from inside ``_bind_turn_scope``, so this is a
    programming-bug guard. The verifier tolerates an absent state via its
    own catch.
    """
    state = _turn_planning_state.get()
    if state is None:
        raise RuntimeError(
            "TurnPlanningState not bound — planning tools must run inside _bind_turn_scope"
        )
    return state


def peek_turn_planning_state() -> TurnPlanningState | None:
    """Read the active turn's planning state without raising.

    Used by the verifier (:mod:`response_validator`) and the retry
    evaluator (:mod:`goal_attainment_retry`); both tolerate an unbound
    turn state by no-op'ing cleanly rather than crashing the post-response
    dispatch.
    """
    return _turn_planning_state.get()


# ---------------------------------------------------------------------------
# get_planning_context — read-only tool
# ---------------------------------------------------------------------------


def get_planning_context(year: int | None = None) -> dict[str, Any]:
    """Return the tenant's configured hiring targets and recruiter capacity.

    Call this tool when the user asks about hiring goals, target attainment
    ("are we on track to hit X"), recruiter capacity, or back-planning.
    The returned values are authoritative for this tenant — cite them
    rather than the generic industry benchmark ranges in your prompt.

    Args:
        year: Optional year to load planning context for. Defaults to the
            session's current year (re-resolved on each call so a session
            spanning Dec 31 stays correct). Pass a different year for
            multi-year capacity planning (e.g. "what about 2027 if we
            keep this pace?").

    The returned ``context_token`` MUST be passed to ``compute_goal_attainment``
    in the same turn — that tool will reject calls without a valid token.
    This causally pins compute to a prior grounding read.

    Returns:
        On hit::

            {
              "context_token": str,            # opaque token; pass to compute
              "year": int,
              "targets": [
                {"role_label": str, "count": int, "job_name_filter": [str, ...]},
                ...
              ],
              "recruiter_capacity_per_month": float | null,
              "active_recruiters": int | null,
              "source": "tenant_config"
            }

        On miss::

            {
              "available": false,
              "reason": "year_not_configured",
              "configured_years": [int, ...]
            }

        In the miss branch, tell the user the target was not configured
        and ask whether they'd like to specify one. Do NOT call
        ``compute_goal_attainment`` in this branch (no token was issued).
    """
    ctx = get_tool_context()
    if ctx is None:
        # Defensive — tools are only invoked from inside a configured
        # session, but we don't want to crash the model on a malformed
        # invocation. Return the "unavailable" branch so the model falls
        # back to the "please specify a target" path.
        return {
            "available": False,
            "reason": "tool_context_unbound",
            "configured_years": [],
        }

    resolved_year = year if year is not None else ctx.current_year_provider()
    pc = ctx.planning_contexts.get(resolved_year)
    if pc is None:
        return {
            "available": False,
            "reason": "year_not_configured",
            "configured_years": sorted(ctx.planning_contexts.keys()),
        }

    # 16 hex chars (= 8 random bytes) — same width as the per-call
    # source_query_id minted by compute_goal_attainment, so the two IDs
    # are visually consistent in logs and easy to grep.
    token = secrets.token_hex(8)
    state = peek_turn_planning_state()
    if state is not None:
        state.context_tokens.add(token)
    payload: dict[str, Any] = pc.model_dump(mode="json")
    payload["context_token"] = token
    return payload


# ---------------------------------------------------------------------------
# compute_goal_attainment — kernel + wrapper
# ---------------------------------------------------------------------------


def compute_goal_attainment_kernel(
    target: int,
    actual_ytd_hires: int,
    months_elapsed: int,
    capacity_per_recruiter_per_month: float | None = None,
    active_recruiters: int | None = None,
    projection_method: str = "linear",
) -> dict[str, Any]:
    """Pure deterministic math. No ToolContext access, no side effects.

    Public so the verifier can import it cross-module to replay recorded
    calls. Tests can call this directly; the wrapper adds the token gate,
    per-call ``source_query_id``, and the recording step.
    """
    months_elapsed = max(1, months_elapsed)  # avoid /0; calling with 0 is meaningless
    months_remaining = max(0, 12 - months_elapsed)
    run_rate = actual_ytd_hires / months_elapsed
    projected_full_year = actual_ytd_hires + int(round(run_rate * months_remaining))
    gap = target - projected_full_year
    on_track = gap <= 0

    capacity_max: int | None = None
    configured_monthly_capacity: float | None = None
    if capacity_per_recruiter_per_month is not None and active_recruiters is not None:
        configured_monthly_capacity = capacity_per_recruiter_per_month * active_recruiters
        capacity_max = actual_ytd_hires + int(configured_monthly_capacity * months_remaining)

    if projection_method != "linear":
        raise ValueError(f"unsupported projection_method: {projection_method}")

    # 80% confidence band on the projection under a Poisson-arrival
    # assumption: var(future arrivals) ≈ run_rate * months_remaining;
    # ±1.28σ for 80% CI. First-order model — seasonality, ramp time, and
    # in-flight pipeline are out of scope for v1. The band narrows as
    # months_elapsed and actual_ytd grow.
    sigma = math.sqrt(max(0.0, run_rate * months_remaining))
    half_width = int(round(1.28 * sigma))
    upper = projected_full_year + half_width
    # Capacity is a structural ceiling: even in the best Poisson scenario,
    # you can't hire faster than recruiters allow.
    if capacity_max is not None:
        upper = min(upper, capacity_max)
    band = {
        "lower": max(0, projected_full_year - half_width),
        "upper": upper,
        "method": "linear_poisson",
        "ci_level": 0.80,
    }

    # Capacity-adequacy narration aids. Even when the model
    # passed the configured capacity here, its prose tends to substitute a
    # generic "2-4 hires/recruiter/month" industry benchmark and conclude the
    # team is badly understaffed. Returning a ready-to-quote verdict — mirroring
    # the ``months_remaining`` "quote verbatim" contract below — anchors the
    # narrative on the configured numbers. Pure functions of the inputs, so the
    # verifier's kernel-replay (response_validator.py) still matches exactly.
    remaining_need = max(0, target - actual_ytd_hires)
    required_run_rate_per_month = (
        round(remaining_need / months_remaining, 2) if months_remaining > 0 else 0.0
    )
    # Only phrase the "...to hit the target" verdict while the year is still in
    # play: at months_remaining == 0 the required run-rate is 0, which would
    # otherwise read "SUFFICIENT to hit the target" even on a missed year. The
    # caller falls back to ``gap``/``on_track`` for the final-outcome case.
    capacity_adequacy_verdict: str | None = None
    if configured_monthly_capacity is not None and months_remaining > 0:
        verdict_word = (
            "SUFFICIENT"
            if configured_monthly_capacity >= required_run_rate_per_month
            else "INSUFFICIENT"
        )
        capacity_adequacy_verdict = (
            f"Configured monthly capacity is {configured_monthly_capacity:.0f} hires/month "
            f"({active_recruiters} recruiters × {capacity_per_recruiter_per_month:.0f}/recruiter/month) "
            f"versus a required run-rate of {required_run_rate_per_month:.1f} hires/month "
            f"→ capacity is {verdict_word} to hit the target."
        )

    return {
        "projected_full_year_hires": projected_full_year,
        "projection_confidence_band": band,
        "gap": gap,
        "on_track": on_track,
        "run_rate_per_month": round(run_rate, 2),
        "required_run_rate_per_month": required_run_rate_per_month,
        "capacity_constrained_max_hires": capacity_max,
        "configured_monthly_capacity": configured_monthly_capacity,
        "capacity_adequacy_verdict": capacity_adequacy_verdict,
        "months_elapsed": months_elapsed,
        "months_remaining": months_remaining,
    }


def compute_goal_attainment(
    target: int,
    actual_ytd_hires: int,
    months_elapsed: int,
    context_token: str,
    capacity_per_recruiter_per_month: float | None = None,
    active_recruiters: int | None = None,
    projection_method: str = "linear",
) -> dict[str, Any]:
    """Compute goal-attainment metrics deterministically.

    Projects full-year hires from the current run-rate, computes the gap to
    target, returns an 80% confidence band on the projection (clamped
    above by capacity when capacity inputs are present), and (if capacity
    inputs are provided) computes the capacity-constrained maximum hires
    achievable in the remaining months.

    Always call this tool for hiring goal-attainment questions — do not
    perform the arithmetic in narrative. You MUST pass ``context_token``
    from a prior ``get_planning_context`` call in this turn; the tool will
    raise a ValueError otherwise.

    Args:
        target: Target hires for the full year (from get_planning_context).
        actual_ytd_hires: Hires year-to-date (from a metric query — the
            query MUST filter on ``job_name_filter`` from
            ``get_planning_context`` for the matching role; mismatches will
            be flagged by the verifier).
        months_elapsed: How many calendar months of the current year are
            complete.
        context_token: The token returned by a prior get_planning_context
            call in the same turn. Causally pins compute to a grounding read.
        capacity_per_recruiter_per_month: Optional, from get_planning_context.
        active_recruiters: Optional headcount, from get_planning_context.
            If both this and ``capacity_per_recruiter_per_month`` are
            provided, enables the capacity-constrained-max branch and
            clamps the band's upper bound at the capacity ceiling.
        projection_method: ``"linear"`` (default). Future methods plug in
            here without changing the call signature.

    Returns the dict produced by ``compute_goal_attainment_kernel``, with
    one added key: ``source_query_id`` — a unique opaque string for THIS
    call. Use that exact string as ``source_query_id`` in any ``Claim``
    sourced to this call so the verifier can match the claim back to this
    invocation.

    Result dict includes ``months_elapsed`` (clamped echo of the input) and
    ``months_remaining`` (``12 - months_elapsed``). When the narrative
    mentions either figure, quote these fields verbatim — do not derive a
    calendar horizon from ``time_to_hire`` or other day-scale metrics.

    When capacity inputs are supplied, the dict also includes
    ``configured_monthly_capacity`` (``capacity_per_recruiter_per_month`` ×
    ``active_recruiters``), ``required_run_rate_per_month``, and a ready-made
    ``capacity_adequacy_verdict`` string. For any statement about recruiter
    capacity, headcount adequacy, or "how many recruiters do we need", quote
    ``capacity_adequacy_verdict`` verbatim — NEVER substitute a generic
    industry hires-per-recruiter benchmark for the configured capacity, and
    never compute a required recruiter count from such a benchmark.
    """
    turn_state = get_turn_planning_state()
    if context_token not in turn_state.context_tokens:
        raise ValueError(
            "compute_goal_attainment requires a context_token from a prior "
            "get_planning_context call in this turn. Call get_planning_context first."
        )

    args = {
        "target": target,
        "actual_ytd_hires": actual_ytd_hires,
        "months_elapsed": months_elapsed,
        "capacity_per_recruiter_per_month": capacity_per_recruiter_per_month,
        "active_recruiters": active_recruiters,
        "projection_method": projection_method,
    }
    result = compute_goal_attainment_kernel(**args)
    source_query_id = f"{SQID_PREFIX_COMPUTE_GOAL_ATTAINMENT}{secrets.token_hex(8)}"
    result["source_query_id"] = source_query_id
    turn_state.recorded_goal_attainment_calls.append(
        GoalAttainmentInvocation(
            source_query_id=source_query_id,
            args=args,
            result=result,
        )
    )
    return result
