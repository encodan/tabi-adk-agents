"""Structured response schema for specialist agents.

Defines the single shape every specialist — and the handoff-chain synthesiser —
returns via Gemini structured output. Downstream code (validator, chart
extraction, handoff detection, confidence UX) reads typed fields instead of
parsing free-form markdown.

See the structured-output design for the full detail.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

import structlog
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from config import SAFETY_SETTINGS
from core.handoff import VALID_SPECIALISTS

logger = structlog.get_logger(__name__)

# Source-of-truth tuple for the specialist enum. Built once from VALID_SPECIALISTS
# (a frozenset[str]) so the Pydantic validator and the Gemini schema enum can
# never drift. Used by HandoffIntent.target_agent (specialists only),
# SpecialistResponse.agent_name (specialists + "synthesis"), and the Gemini
# schema below.
SPECIALIST_NAMES: Final[tuple[str, ...]] = tuple(sorted(VALID_SPECIALISTS))
_VALID_AGENT_NAMES: Final[frozenset[str]] = VALID_SPECIALISTS | {"synthesis"}

# Fallback confidence emitted when structured parse fails and we wrap raw text.
# Confidence-badge calibration treats values <= FALLBACK_CONFIDENCE as the
# "info / parse-failed" badge regardless of model output.
FALLBACK_CONFIDENCE: Final[float] = 0.3


# Filter-value shape. ``dict[str, str]`` keeps the schema simple for Gemini and
# matches how the prompt presents filters to the model. Encoding rules:
#   - time ranges:   "Q1 2026", "2026-01-01..2026-03-31", "last_30d"
#   - multi-value:   "Engineering|Data" (pipe-separated)
#   - single value:  "Engineering"
ClaimFilters = dict[str, str]


class Claim(BaseModel):
    """A single numerical assertion in ``answer_markdown``.

    Every number the user reads should appear here as a claim with a
    ``source_query_id`` pointing into the ``keyed_results`` dict that the
    session threads through from query execution.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    value: float
    unit: Literal["count", "percentage", "days", "currency", "ratio"]
    filters: ClaimFilters
    source_query_id: str
    text_fragment: str


class ChartSpec(BaseModel):
    """A chart the answer references. Frontend renders these directly; no
    ``[chart:...]`` markers are embedded in ``answer_markdown``."""

    model_config = ConfigDict(frozen=True)

    chart_id: str
    chart_type: Literal["line", "bar", "pie", "funnel", "area", "scatter"]
    title: str
    x_axis: str
    y_axis: str
    group_by: str | None = None
    source_query_ids: list[str]


class HandoffIntent(BaseModel):
    """Typed handoff signal. ``target_agent`` is validated against the
    canonical specialist set — a specialist may never hand off to a
    non-specialist name."""

    model_config = ConfigDict(frozen=True)

    target_agent: str
    reason: str
    context_summary: str

    @field_validator("target_agent")
    @classmethod
    def _check_specialist(cls, v: str) -> str:
        if v not in VALID_SPECIALISTS:
            raise ValueError(
                f"target_agent={v!r} not in VALID_SPECIALISTS ({sorted(VALID_SPECIALISTS)})"
            )
        return v


class SpecialistResponse(BaseModel):
    """Structured response returned by every specialist and the synthesiser.

    ``agent_name`` is caller-set after ``model_validate_json`` returns —
    specialists never self-identify. Synthesis sets it to ``"synthesis"``.

    Nested models (``Claim``, ``ChartSpec``, ``HandoffIntent``) are frozen
    for immutability; the parent is mutable so the caller can patch
    ``agent_name``.

    ``handoff is not None`` is the single source of truth for handoff
    intent — there is no separate ``needs_handoff`` flag, which avoids a
    two-field invariant.
    """

    agent_name: str | None = None
    answer_markdown: str
    claims: list[Claim] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    confidence: float
    data_sufficient: bool
    handoff: HandoffIntent | None = None
    reasoning_summary: str | None = None
    agent_error: bool = False
    """Set when the specialist exhausted ``max_llm_calls`` without producing
    text (the salvage path) — the eval reads this to bucket the example
    as agent-error rather than scoring the user-facing salvage text."""

    @field_validator("agent_name")
    @classmethod
    def _check_agent_name(cls, v: str | None) -> str | None:
        # Caller-set, but validated here so a typo (e.g. "pipline_analyst")
        # fails loudly rather than corrupting downstream provenance.
        if v is not None and v not in _VALID_AGENT_NAMES:
            raise ValueError(f"agent_name={v!r} not in {sorted(_VALID_AGENT_NAMES)}")
        return v

    @field_validator("confidence")
    @classmethod
    def _check_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
        return v


# --- Gemini response schema (machine-enforced) -------------------------------
#
# Hand-mirrored from the Pydantic models. ``test_specialist_schema_parity.py``
# guards against drift between the two encodings.
#
# ``agent_name`` is deliberately absent — caller sets it after parse so the
# specialist never has to remember its own name.
#
# Property ``description`` strings are mandatory: Gemini structured-output
# quality degrades materially without them, so every field carries a one-liner.

SPECIALIST_RESPONSE_SCHEMA: Final[genai_types.Schema] = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    required=[
        "answer_markdown",
        "claims",
        "charts",
        "confidence",
        "data_sufficient",
    ],
    properties={
        "answer_markdown": genai_types.Schema(
            type=genai_types.Type.STRING,
            description="The answer text the user reads. Markdown permitted.",
        ),
        "claims": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            description=(
                "Every numerical assertion in answer_markdown must appear here "
                "as a structured claim with a source_query_id pointing into "
                "the Pre-Queried Data block. Empty if the answer contains no "
                "numbers."
            ),
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=[
                    "metric",
                    "value",
                    "unit",
                    "filters",
                    "source_query_id",
                    "text_fragment",
                ],
                properties={
                    "metric": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="Metric identifier, e.g. 'hire_rate', 'time_to_hire'.",
                    ),
                    "value": genai_types.Schema(
                        type=genai_types.Type.NUMBER,
                        description="Numerical value as quoted in the answer.",
                    ),
                    "unit": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        enum=["count", "percentage", "days", "currency", "ratio"],
                        description=("Unit of `value`. Percentages are 0-100, not 0-1."),
                    ),
                    "filters": genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        description=(
                            "Filter dimensions applied to the metric, as "
                            "string→string. Pipe-separate multiple values: "
                            "'Engineering|Data'. Encode time ranges as "
                            "'Q1 2026' or 'YYYY-MM-DD..YYYY-MM-DD'."
                        ),
                    ),
                    "source_query_id": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "The query_id from the Pre-Queried Data block "
                            "whose result this claim was read from. Must "
                            "match exactly."
                        ),
                    ),
                    "text_fragment": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "The exact substring of answer_markdown the "
                            "claim refers to, e.g. '14.2 days'."
                        ),
                    ),
                },
            ),
        ),
        "charts": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            description=(
                "Charts the answer references. Frontend renders these "
                "directly; do NOT embed [chart:...] markers in answer_markdown."
            ),
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=[
                    "chart_id",
                    "chart_type",
                    "title",
                    "x_axis",
                    "y_axis",
                    "source_query_ids",
                ],
                properties={
                    "chart_id": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description=(
                            "Stable id (e.g. 'chart_abc123') used by the frontend for referencing."
                        ),
                    ),
                    "chart_type": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        enum=["line", "bar", "pie", "funnel", "area", "scatter"],
                        description="Visualisation type.",
                    ),
                    "title": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="Human-readable chart title.",
                    ),
                    "x_axis": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="Field name plotted on the x-axis.",
                    ),
                    "y_axis": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="Field name plotted on the y-axis.",
                    ),
                    "group_by": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        nullable=True,
                        description="Optional grouping/series field.",
                    ),
                    "source_query_ids": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        description=(
                            "query_ids from the Pre-Queried Data block that "
                            "supplied this chart's data."
                        ),
                        items=genai_types.Schema(type=genai_types.Type.STRING),
                    ),
                },
            ),
        ),
        "confidence": genai_types.Schema(
            type=genai_types.Type.NUMBER,
            description=(
                "0.0 to 1.0 — how confident the model is in the answer. "
                "Calibrated against eval factuality; emit "
                "honestly, not optimistically."
            ),
        ),
        "data_sufficient": genai_types.Schema(
            type=genai_types.Type.BOOLEAN,
            description=("False if the Pre-Queried Data block was missing data needed to answer."),
        ),
        "handoff": genai_types.Schema(
            type=genai_types.Type.OBJECT,
            nullable=True,
            description=(
                "Set when this specialist cannot answer and another should. "
                "Presence (not a separate boolean) IS the handoff signal — "
                "leave null to answer directly."
            ),
            required=["target_agent", "reason", "context_summary"],
            properties={
                "target_agent": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    enum=list(SPECIALIST_NAMES),
                    description=(
                        "Specialist to route to. Must be a different specialist than this one."
                    ),
                ),
                "reason": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description=("One-line justification, surfaced in logs and (optionally) UX."),
                ),
                "context_summary": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description=(
                        "Findings to carry forward so the target specialist "
                        "doesn't restart from scratch."
                    ),
                ),
            },
        ),
        "reasoning_summary": genai_types.Schema(
            type=genai_types.Type.STRING,
            nullable=True,
            description=(
                "Optional — why the agent chose this analytical approach. "
                "Not shown to users by default."
            ),
        ),
    },
)


# --- Synthesis draft ---------------------------------------------------------
#
# The handoff synthesiser emits a strict subset of ``SpecialistResponse``:
# ``answer_markdown``, ``claims``, ``charts``, ``data_sufficient``. The caller
# (``handoff.py``) deterministically derives ``confidence`` (via
# ``aggregate_confidence``) and sets ``handoff = None`` (synthesis is
# terminal). Defining a separate draft schema saves the tokens that would be
# spent emitting an ignored field and removes a foot-gun where a future
# reader thinks the model's confidence is authoritative.


class SynthesisDraft(BaseModel):
    """Strict subset of :class:`SpecialistResponse` returned by the handoff
    synthesiser. ``confidence`` and ``handoff`` are caller-supplied — the
    model only authors content."""

    model_config = ConfigDict(frozen=True)

    answer_markdown: str
    claims: list[Claim] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    data_sufficient: bool


SYNTHESIS_DRAFT_SCHEMA: Final[genai_types.Schema] = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    required=["answer_markdown", "claims", "charts", "data_sufficient"],
    properties={
        "answer_markdown": SPECIALIST_RESPONSE_SCHEMA.properties["answer_markdown"],
        "claims": SPECIALIST_RESPONSE_SCHEMA.properties["claims"],
        "charts": SPECIALIST_RESPONSE_SCHEMA.properties["charts"],
        "data_sufficient": SPECIALIST_RESPONSE_SCHEMA.properties["data_sufficient"],
    },
)


# --- Safety pin --------------------------------------------------------------


def build_generate_content_config(
    extra: genai_types.GenerateContentConfig | None = None,
) -> genai_types.GenerateContentConfig:
    """Return a ``GenerateContentConfig`` with ``SAFETY_SETTINGS`` pinned.

    All other caller-supplied fields on ``extra`` are retained verbatim.
    If the caller passes their own ``safety_settings`` they are **dropped**
    (with a ``safety_settings_override_dropped`` warning so operators see
    the override was ignored — e.g. an eval_config.json hand-edit adding
    ``safety_settings`` would otherwise vanish silently). The pin is
    non-negotiable; overriding it per-site would defeat the anti-drift
    guarantee that ``test_safety_settings_pin`` enforces.

    Use this at every Agent constructor and every direct
    ``client.aio.models.generate_content`` call so a model upgrade can't
    silently shift safety defaults underneath us.

    Defensive list-copy: ``model_copy(update=...)`` is shallow, so without
    the per-call ``list(SAFETY_SETTINGS)`` the returned config would alias
    the module-level constant — any future in-place mutation of
    ``cfg.safety_settings`` (e.g. by ADK instrumentation) would corrupt
    every other call site sharing the same list reference.
    """
    pinned = list(SAFETY_SETTINGS)
    if extra is None:
        return genai_types.GenerateContentConfig(safety_settings=pinned)
    if extra.safety_settings:
        logger.warning(
            "safety_settings_override_dropped",
            caller_count=len(extra.safety_settings),
            reason="pin is non-negotiable; see specialist_schema.build_generate_content_config",
        )
    return extra.model_copy(update={"safety_settings": pinned})


# --- Fallback construction ---------------------------------------------------


def _apply_structured_output(llm_request: object) -> None:
    """Mutate ``llm_request.config`` so Gemini emits structured JSON.

    ADK's ``LlmAgent.__init__`` rejects ``generate_content_config.response_schema``
    at construction time (it expects ``output_schema``, which disables tools —
    not viable for specialists). We set the schema on the ``LlmRequest`` config
    just before the call, bypassing ADK's constructor validator without
    touching the Runner's tool-dispatch loop.

    Idempotent: safe if invoked multiple times on the same request.
    """
    config = getattr(llm_request, "config", None)
    if config is None:
        return
    config.response_mime_type = "application/json"
    config.response_schema = SPECIALIST_RESPONSE_SCHEMA


def build_structured_output_callback(
    agent_name: str,
) -> object:
    """Return a ``before_model_callback`` that forces structured JSON output
    for ``agent_name`` when ``feature_flags.structured_output_enabled`` is on
    *at request time*.

    The flag is re-read on every invocation (not at agent-construction time)
    so the callback and the session's parser — which also reads the flag at
    request time in ``_run_specialist_collect`` and ``synthesize_specialist_responses``
    — can never disagree about whether output is structured. Without this, a
    ``clear_config_cache()`` between agent build and request handling would
    leave the model emitting JSON while the session expected free-form text
    (or vice versa), surfacing as raw JSON in the user's chat or as parse-
    failure spam in the logs.

    Always returns a callable. When the flag is off the callback is a no-op
    and the pre-Phase-3 free-form behaviour is preserved byte-identically.
    """

    def _callback(
        callback_context: object,  # CallbackContext — unused, matches ADK sig
        llm_request: object,
    ) -> None:
        # Lazy import inside the callback to avoid a circular import: ``config``
        # imports ``handoff.VALID_SPECIALISTS`` which also feeds this module.
        from config import get_config

        if not get_config().feature_flags.structured_output_enabled:
            return None
        _apply_structured_output(llm_request)
        return None

    _callback.__name__ = f"structured_output_callback[{agent_name}]"
    return _callback


def build_fallback_response(
    raw_text: str,
    *,
    agent_name: str | None = None,
    agent_error: bool = False,
) -> SpecialistResponse:
    """Synthesise a minimal ``SpecialistResponse`` from raw text.

    Used when both the primary parse and the "reformat" retry fail — never
    raise, the user must still receive an answer. Fallback confidence
    collapses to the parse-failed badge in the confidence UX.
    """
    return SpecialistResponse(
        agent_name=agent_name,
        answer_markdown=raw_text,
        claims=[],
        charts=[],
        confidence=FALLBACK_CONFIDENCE,
        data_sufficient=False,
        handoff=None,
        reasoning_summary=None,
        agent_error=agent_error,
    )


# A fenced code block whose body is a single JSON object. Used to detect a
# leaked structured response the model echoed into prose.
_FENCED_JSON_OBJECT_RE: Final = re.compile(r"```[a-zA-Z]*\s*(\{.*?\})\s*```", re.DOTALL)

# Top-level keys unique to a ``SpecialistResponse``-shaped object. A
# user-facing JSON example would not carry these.
_LEAKED_STRUCT_KEYS: Final = ("claims", "answer_markdown")


def strip_leaked_structured_block(text: str) -> str:
    """Remove fenced code blocks that are a leaked structured response.

    The two-pass synthesis path assigns ``answer_markdown`` straight from the
    constrained-decoded envelope text. If the model emits a ```json {...}```
    block carrying ``claims``/``answer_markdown`` (the structured schema it
    saw in its prompt), it would otherwise reach the user verbatim — the leak
    seen on the deterministic/fast path. Only blocks whose body parses to a
    dict containing a structured-schema key are removed, so legitimate
    user-facing JSON examples are preserved.
    """
    if not text or "```" not in text:
        return text

    def _replace(match: re.Match[str]) -> str:
        try:
            parsed = json.loads(match.group(1))
        except (ValueError, TypeError):
            return match.group(0)
        if isinstance(parsed, dict) and any(k in parsed for k in _LEAKED_STRUCT_KEYS):
            return ""
        return match.group(0)

    return _FENCED_JSON_OBJECT_RE.sub(_replace, text).strip()


def parse_specialist_response(
    raw_text: str,
    *,
    agent_name: str,
) -> SpecialistResponse:
    """Parse ``raw_text`` as a ``SpecialistResponse`` with fallback handling.

    Primary path: ``SpecialistResponse.model_validate_json`` — nested
    ``Claim``/``ChartSpec``/``HandoffIntent`` reconstruct recursively.
    Fallback: ``build_fallback_response`` wraps the raw text with
    ``FALLBACK_CONFIDENCE`` and ``data_sufficient=False``.

    Never raises — the user must still receive an answer by design. We do
    NOT currently issue a "reformat" retry; the fallback
    is one step, and the structured-output flag gives operators a kill switch
    if retry latency/cost proves worthwhile later.

    Always sets ``agent_name`` on the returned response — caller-set,
    validated against the canonical specialist set by ``SpecialistResponse``.
    """
    text = (raw_text or "").strip()
    if not text:
        logger.warning(
            "specialist_response.parse_failed",
            agent=agent_name,
            outcome="empty_text",
            raw_text_len=0,
        )
        return build_fallback_response("", agent_name=agent_name)
    try:
        response = SpecialistResponse.model_validate_json(text)
    except ValidationError as exc:
        logger.warning(
            "specialist_response.parse_failed",
            agent=agent_name,
            outcome="validation_error",
            error=repr(exc),
            raw_text_len=len(text),
        )
        return build_fallback_response(text, agent_name=agent_name)
    # Counterpart success event so dashboards can compute parse-failure rate
    # as failed / (failed + succeeded) — the budget is <2% on the golden
    # set.
    logger.info(
        "specialist_response.parse_succeeded",
        agent=agent_name,
        raw_text_len=len(text),
        claim_count=len(response.claims),
    )
    response.agent_name = agent_name
    response.answer_markdown = strip_leaked_structured_block(response.answer_markdown)
    return response


# --- keyed_results lifetime helper -------------------------------------------


def freeze_keyed_results(
    keyed_results: Mapping[str, object],
) -> Mapping[str, object]:
    """Return a read-only snapshot of ``keyed_results`` whose lifetime is
    decoupled from the per-turn ContextVar.

    Consumers that need the dict beyond a single turn (e.g. a critique
    pass running after turn cleanup) should snapshot here rather than
    retain a live reference — the ContextVar is reset on each new turn,
    and a retained reference to the old dict would stop tracking fresh
    writes. The ``MappingProxyType`` wrapper also prevents accidental
    mutation by downstream code.
    """
    return MappingProxyType(dict(keyed_results))


__all__ = [
    "Claim",
    "ChartSpec",
    "ClaimFilters",
    "FALLBACK_CONFIDENCE",
    "HandoffIntent",
    "SPECIALIST_NAMES",
    "SPECIALIST_RESPONSE_SCHEMA",
    "SYNTHESIS_DRAFT_SCHEMA",
    "SpecialistResponse",
    "SynthesisDraft",
    "build_fallback_response",
    "build_structured_output_callback",
    "freeze_keyed_results",
    "parse_specialist_response",
]
