"""Shared handoff primitives used by the orchestrator and the fast-path
dispatch loop so detection, guards, and synthesis behave identically.

Owns handoff-detection / tool-output shaping primitives and the
deterministic confidence aggregator. The synthesis call itself lives in
:mod:`tabi_analytics.core.synthesizer`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from config import _calibration_passed

if TYPE_CHECKING:
    from core.specialist_schema import SpecialistResponse

    # [public-repo stub] models.answer_envelope excluded — alias to Any so the
    # type annotations below resolve. These are type-only (PEP 563 strings at
    # runtime), so this never affects execution in the showcase.
    AnswerEnvelope = Any
    PassOneOutput = Any

logger = structlog.get_logger(__name__)


# Maximum number of handoffs allowed per turn (prevents infinite loops).
# Shared by the orchestrator and the fast-path dispatch loop.
MAX_HANDOFF_DEPTH = 3

# Cap on tool outputs threaded forward across a handoff chain. At MAX_HANDOFF_DEPTH
# the accumulated list could otherwise grow to (depth+1) × per-specialist calls;
# when exceeded we keep the most recent entries so the freshest context wins.
MAX_ACCUMULATED_TOOL_OUTPUTS = 20

# Canonical specialist agent set.
VALID_SPECIALISTS: frozenset[str] = frozenset(
    {
        "pipeline_analyst",
        "general_analyst",
        "sourcing_strategist",
        "offer_advisor",
        "interviewing_coach",
        "capacity_planner",
        "data_scientist",
    }
)


# Matches ValidationConfig.soft_threshold (0.05) so the synthesis floor and
# the validator's "annotate" boundary cross at the same point.
_CONFLICT_TOLERANCE = 0.05


@dataclass(frozen=True)
class CalibrationGate:
    """Read-only view of ``calibration.passed`` from ``baseline.yaml``.
    Injected into :func:`aggregate_confidence` so tests pass a stub rather
    than monkey-patching module state."""

    calibrated: bool

    @classmethod
    def from_baseline(cls) -> CalibrationGate:
        return cls(calibrated=_calibration_passed())


def _default_gate() -> CalibrationGate:
    # Resolved at call time so tests patching ``_calibration_passed`` see
    # the new value (the underlying reader is lru_cached).
    return CalibrationGate.from_baseline()


def _has_numerical_conflict(
    responses: list[SpecialistResponse],
) -> bool:
    """Return True if two specialists report values for the same
    ``(metric, filters)`` differing by more than ``_CONFLICT_TOLERANCE``.
    Zero-valued claims fall back to absolute delta (relative deviation is
    undefined at zero).

    The ``filters`` dict on each claim is the safety net against false
    positives: legitimately-different scopes (e.g. one specialist queried
    last week, another queried this month) only collide when both omit the
    distinguishing filter (e.g. ``time_range``). Specialist prompts must
    record every filter that scoped a claim — the v3.1 base specialist
    spells this out under Output Format. A missing filter manifests as a
    spurious confidence floor here, which is the safer failure mode."""
    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}
    for response in responses:
        if response is None:
            continue
        for claim in response.claims:
            key = (claim.metric, tuple(sorted(claim.filters.items())))
            grouped.setdefault(key, []).append(claim.value)

    for values in grouped.values():
        if len(values) < 2:
            continue
        lo, hi = min(values), max(values)
        if lo == 0:
            if abs(hi) > _CONFLICT_TOLERANCE:
                return True
            continue
        if abs(hi - lo) / abs(lo) > _CONFLICT_TOLERANCE:
            return True
    return False


def aggregate_confidence(
    responses: list[SpecialistResponse],
    weights: dict[str, float],
    *,
    gate: CalibrationGate | None = None,
) -> float:
    """Relevance-weighted mean of specialist confidences, bounded to [0, 1].

    ``weights`` maps ``agent_name`` → relevance in this turn:
      - Primary agent weight = ``route.confidence``.
      - Each secondary agent weight =
        ``route.confidence * ModelTieringConfig.secondary_agent_weight``.

    Bounds:
      - Upper bound = max(specialist confidences). Synthesis cannot be more
        confident than its most-confident contributor.
      - Lower bound = min(confidences) iff ``_has_numerical_conflict``,
        else 0. A genuine numerical disagreement floors the aggregate at the
        weakest link.

    Calibration fallback: when calibration has not passed, by design we
    short-circuit to ``min(confidences)``. Single source of truth — the gate
    is read once via :func:`_default_gate` and may be overridden by tests
    via the ``gate`` keyword (no env-var override, by design).
    """
    if not responses:
        return 0.0

    clipped = [max(0.0, min(1.0, r.confidence)) for r in responses]
    if len(clipped) == 1:
        return clipped[0]

    resolved_gate = gate if gate is not None else _default_gate()
    if not resolved_gate.calibrated:
        return min(clipped)

    total_w = 0.0
    weighted_sum = 0.0
    for confidence, response in zip(clipped, responses, strict=True):
        w = max(0.0, weights.get(response.agent_name or "", 0.0))
        total_w += w
        weighted_sum += confidence * w

    if total_w <= 0.0:
        # No relevance signal — fall back to the unweighted mean rather than
        # silently returning 0 (which would be misleading to the UI).
        weighted = sum(clipped) / len(clipped)
    else:
        weighted = weighted_sum / total_w

    upper = max(clipped)
    lower = min(clipped) if _has_numerical_conflict(responses) else 0.0

    bounded = max(lower, min(upper, weighted))
    return max(0.0, min(1.0, bounded))


@dataclass(frozen=True)
class HandoffRequest:
    """A specialist's request to hand off to another specialist."""

    source_agent: str
    target_agent: str
    reason: str
    context_summary: str


@dataclass
class SpecialistRunResult:
    """Captured output from running one specialist on the fast path.

    When structured output is active, ``response`` carries the
    parsed ``SpecialistResponse``; downstream code (validator, synthesiser)
    reads typed fields off it. When structured output is disabled, ``response``
    stays ``None`` and only ``text`` is populated.
    """

    agent_name: str
    text: str
    handoff: HandoffRequest | None = None
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    response: SpecialistResponse | None = None
    # True iff the ADK runner raised ``LlmCallsLimitExceededError`` during
    # this run. The cap is swallowed inside the specialist runner's
    # collection path (the user-visible salvage path takes over), but the goal-attainment
    # retry wrapper reads this flag to emit ``retry_cap_exhausted`` in
    # ``goal_attainment.turn_outcome`` so we can size the retry cap.
    cap_exhausted: bool = False
    # Two-pass intermediates; ``None`` on the legacy single-pass path.
    envelope: AnswerEnvelope | None = None
    pass_one: PassOneOutput | None = None


def detect_handoff(part: Any, source_agent: str) -> HandoffRequest | None:
    """Return a ``HandoffRequest`` if the ADK event part is a handoff call.

    Returns ``None`` when:
    - the part has no ``function_response`` (text, function_call, etc.),
    - the function response is for a different tool,
    - the response payload is missing or not a dict, or
    - the payload's ``status`` is not ``"handoff_requested"`` (e.g. the
      tool returned ``status="error"`` for an invalid target).

    That gating means an invalid target never flows into the dispatch loop.
    """
    func_response = getattr(part, "function_response", None)
    if func_response is None:
        return None
    if getattr(func_response, "name", None) != "request_specialist_handoff":
        return None

    data = getattr(func_response, "response", None)
    if not isinstance(data, dict) or data.get("status") != "handoff_requested":
        return None

    return HandoffRequest(
        source_agent=source_agent,
        target_agent=str(data.get("target", "")),
        reason=str(data.get("reason", "")),
        context_summary=str(data.get("context", "")),
    )


def collect_tool_response(part: Any) -> dict[str, Any] | None:
    """Return ``{"name", "response"}`` for a non-handoff tool response part.

    Returns ``None`` for parts that aren't tool responses, or for the
    handoff tool itself (which is routed through ``detect_handoff``).
    """
    func_response = getattr(part, "function_response", None)
    if func_response is None:
        return None
    name = getattr(func_response, "name", None)
    if name is None or name == "request_specialist_handoff":
        return None
    return {"name": name, "response": getattr(func_response, "response", None)}


def truncate_tool_response(response: Any, max_rows: int = 50) -> Any:
    """Cap list-valued ``data`` fields at ``max_rows`` to bound prompt size.

    Mirrors the original orchestrator ``_truncate_tool_response`` behavior:
    only truncates when ``response["data"]`` is a list longer than the cap,
    and flags the truncation via ``metadata.data_truncated``.
    """
    if not isinstance(response, dict):
        return response

    data = response.get("data")
    if not isinstance(data, list) or len(data) <= max_rows:
        return response

    truncated = dict(response)
    truncated["data"] = data[:max_rows]
    metadata = dict(response.get("metadata", {}))
    metadata["data_truncated"] = True
    metadata["data_original_count"] = len(data)
    truncated["metadata"] = metadata
    return truncated


def format_tool_outputs(outputs: list[dict[str, Any]]) -> str:
    """Render a list of ``{"name", "response"}`` dicts as prompt-ready JSON."""
    if not outputs:
        return ""

    lines: list[str] = []
    for idx, tool_output in enumerate(outputs, start=1):
        name = tool_output.get("name", "unknown_tool")
        response = tool_output.get("response")
        sanitized = truncate_tool_response(response)
        rendered = json.dumps(
            sanitized,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
            indent=2,
        )
        lines.append(f"[Tool {idx}] {name} response:\n{rendered}\n")
    return "\n".join(lines)


def build_handoff_question(
    original_question: str,
    source_agent: str,
    handoff: HandoffRequest,
    tool_outputs: list[dict[str, Any]],
) -> str:
    """Build the question the target specialist will see.

    The target specialist is the synthesizer in the default (``target``)
    mode: it must open with a one-sentence framing of the source's
    diagnosis and then build on it with its own expertise, producing one
    continuous answer. Tool outputs from the source are threaded in so the
    target can reason about the same data without re-querying.
    """
    sections = [
        f"Original question from the user: {original_question}",
        "",
        (
            f"The `{source_agent}` specialist analyzed this first and "
            "produced the following handoff context:"
        ),
        f"- Reason for handoff: {handoff.reason}",
        f"- Key findings: {handoff.context_summary}",
    ]

    if tool_outputs:
        sections.extend(
            [
                "",
                "Shared tool outputs from the previous specialist (reuse these — do NOT re-run queries):",
                format_tool_outputs(tool_outputs),
            ]
        )

    sections.extend(
        [
            "",
            "## Your task",
            (
                "Absorb the previous specialist's diagnosis and build on it with your own expertise. "
                "Produce ONE continuous analysis that reads as a single voice:"
            ),
            (
                "1. Open with a single-sentence framing of the previous diagnosis "
                "(do NOT say the phrase 'handoff' or name another specialist)."
            ),
            "2. Then deliver your own analysis and recommendations, extending — not repeating — their findings.",
            (
                "3. Do NOT re-run queries you can answer from the shared tool outputs above. "
                "Only query if new data is strictly required."
            ),
            (
                "4. The user is not aware multiple specialists exist. "
                "Write as a single analyst answering the original question."
            ),
        ]
    )

    return "\n".join(sections)


__all__ = [
    "CalibrationGate",
    "HandoffRequest",
    "MAX_ACCUMULATED_TOOL_OUTPUTS",
    "MAX_HANDOFF_DEPTH",
    "SpecialistRunResult",
    "VALID_SPECIALISTS",
    "aggregate_confidence",
    "build_handoff_question",
    "collect_tool_response",
    "detect_handoff",
    "format_tool_outputs",
    "truncate_tool_response",
]
