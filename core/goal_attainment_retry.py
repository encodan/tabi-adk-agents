"""Reflexion-style goal-attainment retry decision: Emit | Retry | Salvage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import structlog

from core.planning_claims import (
    CAPACITY_CONSTRAINED_MAX_HIRES,
    HIRING_GAP,
    MONTHS_ELAPSED,
    MONTHS_REMAINING,
    PROJECTED_FULL_YEAR_HIRES,
    PROJECTED_FULL_YEAR_HIRES_LOWER,
    PROJECTED_FULL_YEAR_HIRES_UPPER,
)
from core.response_validator import goal_attainment_verifier
from core.specialist_schema import Claim, SpecialistResponse
from tools.planning_tools import (
    TurnPlanningState,
    peek_turn_planning_state,
)
from tools.tool_context import GoalAttainmentInvocation

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Emit:
    response: SpecialistResponse


@dataclass(frozen=True)
class Retry:
    critique_prompt: str


@dataclass(frozen=True)
class Salvage:
    pass


Decision = Emit | Retry | Salvage


# Severity ordering: replay/input mismatches first (hardest to recover from
# narration alone), then value mismatches, then filter mismatches.
_OUTCOME_SEVERITY: Final[dict[str, int]] = {
    "replay_mismatch": 0,
    "claim_replay_mismatch": 0,
    "compute_input_mismatch": 1,
    "compute_input_multi_reference": 1,
    "compute_target_input_mismatch": 1,
    "compute_capacity_input_mismatch": 1,
    "compute_headcount_input_mismatch": 1,
    "claim_unknown_source": 1,
    "claims_missing": 2,
    "unconfigured": 3,
    "target_role_unconfigured": 3,
    "target_mismatch": 4,
    "capacity_mismatch": 4,
    "headcount_mismatch": 4,
    "filter_mismatch": 5,
}


def _outcome_sort_key(detail: tuple[str, str]) -> tuple[int, str]:
    outcome, _ = detail
    return (_OUTCOME_SEVERITY.get(outcome, 99), outcome)


def _render_critique(
    outcomes_with_details: list[tuple[str, str]],
    prior_context_token: str | None,
) -> str:
    """Render the retry prompt.

    Renders every flagged outcome (not just the first) in severity order;
    surfaces the most recent ``context_token`` issued this turn so the
    model can re-use it without re-calling ``get_planning_context``.
    """
    ordered = sorted(outcomes_with_details, key=_outcome_sort_key)
    bullets = "\n".join(
        f"  - {outcome}: {detail}" for outcome, detail in ordered if outcome != "match"
    )

    token_line = f'       context_token = "{prior_context_token}"' if prior_context_token else ""
    token_clause = (
        (
            "1. Re-read `get_planning_context()` if needed (sync; cheap), OR re-use\n"
            "   the context_token you obtained earlier this turn:\n"
            f"{token_line}\n"
        )
        if prior_context_token
        else ("1. Re-read `get_planning_context()` to obtain a fresh context_token.\n")
    )

    return (
        "Your previous response failed grounding verification. The following\n"
        "mismatches must be corrected:\n\n"
        f"{bullets}\n\n"
        "To revise:\n"
        f"{token_clause}"
        "2. Re-run any metric queries with corrected filters.\n"
        "3. Re-call `compute_goal_attainment(..., context_token=...)` with the\n"
        "   corrected inputs. The previous compute call's `source_query_id` is\n"
        "   abandoned the moment your revised response stops referencing it —\n"
        "   the verifier only checks invocations referenced by claims, so the\n"
        "   stale call will not cause a spurious compute_input_mismatch.\n"
        "4. Emit a fresh response with claims that reflect the corrected values\n"
        "   and reference the NEW compute call's `source_query_id`.\n\n"
        "Do not invent values. If `get_planning_context` returns\n"
        "`available: false`, ask the user for the target rather than guessing."
    )


def _any_context_token(state: TurnPlanningState) -> str | None:
    """Return any context token issued this turn (the model only needs one
    valid token to re-use), or ``None`` if none was issued.
    """
    return next(iter(state.context_tokens), None)


def _synthesize_claims_from_invocation(
    invocation: GoalAttainmentInvocation,
) -> list[Claim]:
    """Build the planning Claims a well-behaved model would have emitted for
    this ``compute_goal_attainment`` call.

    Skipped on input metrics (``hiring_target``, ``actual_ytd_hires``,
    capacity inputs) — those have their own provenance (``get_planning_context``
    or MetricFlow queries) that's not addressable from the invocation alone.
    Only output metrics directly recoverable from ``invocation.result`` are
    synthesized.
    """
    result = invocation.result
    sqid = invocation.source_query_id
    band = result.get("projection_confidence_band") or {}
    # ``unit="count"`` for every metric, including ``months_elapsed`` and
    # ``months_remaining``: semantically those are durations measured in
    # months, but the ``Claim.unit`` schema is a closed Literal
    # (``count|percentage|days|currency|ratio``) — adding ``months`` would
    # require coordinating a schema change against the constrained-decode
    # response_schema and frontend types, which is out of scope here.
    # Safe in practice because the goal-attainment verifier matches by
    # ``source_query_id``, not by unit; the duration sub-unit guard in
    # ``_match_claim`` keys off ``ClaimedValue.value_type`` (regex-extracted
    # from prose), not ``Claim.unit``.
    candidates: list[tuple[str, float | None]] = [
        (PROJECTED_FULL_YEAR_HIRES, result.get("projected_full_year_hires")),
        (PROJECTED_FULL_YEAR_HIRES_LOWER, band.get("lower")),
        (PROJECTED_FULL_YEAR_HIRES_UPPER, band.get("upper")),
        (HIRING_GAP, result.get("gap")),
        (CAPACITY_CONSTRAINED_MAX_HIRES, result.get("capacity_constrained_max_hires")),
        (MONTHS_ELAPSED, result.get("months_elapsed")),
        (MONTHS_REMAINING, result.get("months_remaining")),
    ]
    claims: list[Claim] = []
    for metric, value in candidates:
        if value is None:
            continue
        claims.append(
            Claim(
                metric=metric,
                value=float(value),
                unit="count",
                filters={},
                source_query_id=sqid,
                text_fragment=str(value),
            )
        )
    return claims


def _classify_synthesis_grounding(response: SpecialistResponse, state: TurnPlanningState) -> str:
    """Prose-grounding telemetry: when claims are synthesized for a turn that fed
    configured capacity into ``compute_goal_attainment``, classify whether the
    *prose* actually engaged that capacity.

    A grounded narrative cites the configured monthly capacity (recruiters ×
    per-recruiter rate); a drifted one substitutes a generic benchmark or asks
    for values it already retrieved — yet both look identical once the claims
    are synthesized in. Returns ``"synthesized_masked"`` when no
    configured-capacity figure appears in the prose (a likely prose-grounding
    miss the synthesis hides), ``"synthesized_benign"`` when one does, or
    ``"synthesized_no_capacity"`` when the turn fed no capacity inputs (signal
    not applicable). Read-only — no behaviour depends on the result.
    """
    monthly_totals = [
        total
        for inv in state.recorded_goal_attainment_calls
        if (total := inv.result.get("configured_monthly_capacity")) is not None
    ]
    if not monthly_totals:
        return "synthesized_no_capacity"
    prose = response.answer_markdown or ""
    # Match the configured MONTHLY TOTAL (e.g. 80), not the per-recruiter rate:
    # small per-recruiter rates (2-5) collide with the industry benchmark text
    # ("2-4"), which would false-label benchmark-substituting prose as grounded.
    # Word-boundary so the total isn't found inside a longer number (800/2026).
    grounded = any(re.search(rf"\b{int(total)}\b", prose) for total in monthly_totals)
    return "synthesized_benign" if grounded else "synthesized_masked"


def _render_prose_grounding_critique(
    state: TurnPlanningState,
    prior_context_token: str | None,
) -> str:
    """Corrective-retry prompt for a goal-attainment turn whose
    *narrative* disengaged from the configured capacity — it asked for the
    target/recruiter capacity it had already retrieved, or substituted a generic
    industry benchmark, even though the claims synthesised cleanly.

    Embeds the kernel's ready-made ``capacity_adequacy_verdict`` (and, as a
    fallback, the configured monthly-capacity figure) so the revised prose can
    quote the grounded numbers verbatim rather than re-deriving or re-requesting
    them. Only reached when ``_classify_synthesis_grounding`` returned
    ``"synthesized_masked"``, which guarantees at least one recorded invocation
    carried a non-null ``configured_monthly_capacity`` — so the grounding block
    is never empty.
    """
    verdicts = [
        v
        for inv in state.recorded_goal_attainment_calls
        if (v := inv.result.get("capacity_adequacy_verdict"))
    ]
    grounding_lines = [f"  - {v}" for v in verdicts]
    if not grounding_lines:
        # months_remaining == 0 suppresses the verdict string; fall back to the
        # raw configured monthly total so the prose still has a figure to cite.
        monthly_totals = [
            total
            for inv in state.recorded_goal_attainment_calls
            if (total := inv.result.get("configured_monthly_capacity")) is not None
        ]
        joined = ", ".join(f"{int(t)} hires/month" for t in monthly_totals)
        grounding_lines = [
            f"  - Configured monthly recruiter capacity: {joined} "
            "(already retrieved this turn — do not ask for it)."
        ]
    grounding_block = "\n".join(grounding_lines)

    token_line = f'       context_token = "{prior_context_token}"' if prior_context_token else ""
    token_clause = (
        (f"Re-use the context_token you already obtained this turn:\n{token_line}\n")
        if prior_context_token
        else "Re-read `get_planning_context()` to recover the configured values.\n"
    )

    return (
        "Your previous response treated the hiring target and recruiter capacity\n"
        "as missing — but BOTH were already retrieved via `get_planning_context()`\n"
        "this turn and fed into `compute_goal_attainment`. Do not ask the user for\n"
        "values you already have, and do not substitute a generic industry\n"
        "hires-per-recruiter benchmark for the configured capacity.\n\n"
        "Revise the narrative so it states the configured-capacity verdict\n"
        "explicitly. Quote this verbatim:\n\n"
        f"{grounding_block}\n\n"
        f"{token_clause}"
        "Keep the existing compute_goal_attainment results (projection, confidence\n"
        "band, gap, capacity-constrained max) and emit the structured claims that\n"
        "reference that call's `source_query_id`. Do not invent values."
    )


def _synthesizable_claims_missing(
    state: TurnPlanningState,
    real_outcomes: list[tuple[str, str]],
) -> bool:
    """``claims_missing`` is the *only* flagged outcome and we have recorded
    ``compute_goal_attainment`` invocations to synthesise the missing Claims
    from — the precondition for both the synthesis short-circuit and the
    corrective prose-grounding retry."""
    if not real_outcomes or any(o != "claims_missing" for o, _ in real_outcomes):
        return False
    return bool(state.recorded_goal_attainment_calls)


def _synthesize_missing_claims(
    response: SpecialistResponse,
    state: TurnPlanningState,
    grounding: str,
) -> None:
    """Fill in the missing Claims from the recorded ``compute_goal_attainment``
    invocations rather than burning a retry pass.

    Usually the model produced acceptable narrative text and just skipped the
    structured-claim emission step; synthesizing from the kernel result
    preserves grounding (the values came from the deterministic kernel) and
    saves ~10–15 s of LLM time on the retry pass.

    Callers gate this on :func:`_synthesizable_claims_missing`. ``grounding`` is
    the pre-computed :func:`_classify_synthesis_grounding` label, logged so the
    masked case stays visible — the behavioural fix for the masked case (a
    corrective prose-grounding retry) is handled upstream in
    :func:`evaluate` before synthesis is reached.
    """
    synthesized: list[Claim] = []
    for invocation in state.recorded_goal_attainment_calls:
        synthesized.extend(_synthesize_claims_from_invocation(invocation))
    if not synthesized:
        return

    response.claims = [*response.claims, *synthesized]
    response.agent_error = False
    # Re-run the verifier so downstream code sees a clean outcome trace.
    goal_attainment_verifier(response)
    logger.info(
        "goal_attainment.claims_synthesized",
        synthesized_count=len(synthesized),
        invocations=len(state.recorded_goal_attainment_calls),
    )
    # Surface the prose-grounding status the synthesis would otherwise mask
    # (an earlier incident showed synthesis hiding a benchmark-substituting
    # narrative). When ``grounding == "synthesized_masked"`` reaches here the
    # corrective retry was already spent or unavailable.
    logger.info(
        "goal_attainment.prose_grounding_suspect",
        agent=response.agent_name,
        action=grounding,
        synthesized_count=len(synthesized),
    )


def evaluate(response: SpecialistResponse | None) -> Decision:
    """Verify ``response`` and decide whether to emit, retry, or salvage.

    The verifier mutates ``response.agent_error`` in place when a mismatch
    is detected (see :func:`goal_attainment_verifier`).
    """
    if response is None:
        return Salvage()

    # Snapshot agent_error so a response that already carried
    # agent_error=True from upstream salvage isn't re-flagged here.
    pre_error = response.agent_error

    goal_attainment_verifier(response)

    state = peek_turn_planning_state()
    if state is None:
        return Emit(response)

    flagged = response.agent_error and not pre_error
    if not flagged:
        return Emit(response)

    real_outcomes = [(o, d) for o, d in state.last_verifier_outcomes if o != "match"]
    if not real_outcomes:
        return Emit(response)

    # ``claims_missing`` with recorded compute calls is normally recoverable by
    # synthesizing the structured claims from the kernel result instead of
    # burning a retry. But when the *prose* disengaged from the configured
    # capacity (asked for values it already had / substituted a benchmark),
    # synthesis would heal the claims while leaving a user-visible contradiction
    # in the narrative. Fire one corrective retry in that masked
    # case; fall back to synthesis when the retry budget is spent or the prose
    # is already grounded.
    if _synthesizable_claims_missing(state, real_outcomes):
        grounding = _classify_synthesis_grounding(response, state)
        if grounding == "synthesized_masked" and state.retry_budget > 0:
            state.retry_budget -= 1
            critique = _render_prose_grounding_critique(state, _any_context_token(state))
            logger.info(
                "goal_attainment.retry_emitted",
                outcomes=["prose_grounding_masked"],
                remaining_budget=state.retry_budget,
            )
            return Retry(critique)
        _synthesize_missing_claims(response, state, grounding)
        return Emit(response)

    if state.retry_budget <= 0:
        logger.info(
            "goal_attainment.retry_outcome",
            retry_outcome="retry_exhausted",
            outcomes=[o for o, _ in real_outcomes],
        )
        return Salvage()

    state.retry_budget -= 1
    critique = _render_critique(real_outcomes, _any_context_token(state))
    logger.info(
        "goal_attainment.retry_emitted",
        outcomes=[o for o, _ in real_outcomes],
        remaining_budget=state.retry_budget,
    )
    return Retry(critique)


__all__ = [
    "Decision",
    "Emit",
    "Retry",
    "Salvage",
    "evaluate",
]
