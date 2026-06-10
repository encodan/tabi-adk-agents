"""Pipeline-shaped trace model — the data shape TABI's custom trace viewer renders.

This is what makes TABI's observability domain-specific rather than generic.
Generic agent-observability tools (Langfuse / Phoenix / Braintrust) flatten every
interesting decision — route vs. query-plan vs. salvage vs. narrative — into
"just another LLM call." TABI instead renders each turn in the *pipeline's own
shape*::

    input -> route -> query_plan -> tools -> specialist -> salvage -> narrative
          -> streamed_output -> judges  (+ a human annotation rail)

These Pydantic models are the contract between three things:

1. the manual OTel spans emitted in :mod:`core.spans` (the *writer* side — the
   stages ADK does not instrument: turn root, router classify, query execute,
   metrics plan, salvage, narrative);
2. the projector in :mod:`core.trace_projection` (which flattens a span tree into
   these panels); and
3. the trace-viewer UI (which renders one component per non-null panel).

In the full TABI platform the projection runs server-side over a three-store
fan-out (BigQuery log sink × Firestore × Postgres ``turn_summaries``) and the UI
lives in the Next.js frontend; both are excluded from this public repo. What's
preserved here is the *shape* — the part that makes the observability
domain-specific rather than generic — wired to the span emission and the
mock-backed projector so it runs and is unit-tested.

Mirrors the private platform's trace model; the
annotation-write / judge-rerun / κ-alignment API surface is omitted as
platform-specific.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# A judge rollup over a turn. ``pass`` iff every judge passed, ``fail`` iff every
# judge failed, ``mixed`` iff split, ``none`` iff no judge has scored it yet.
JudgeSummary = Literal["pass", "fail", "mixed", "none"]
# The human label on the annotation rail (drives the annotate -> evalset loop).
AnnotationLabel = Literal["pass", "fail", "skip"]


# ---------------------------------------------------------------------------
# Stage panels — one per pipeline stage. ``None`` when the stage didn't fire.
# Each is a thin mirror of the structured span/log shape; the viewer renders
# one component per non-null panel.
# ---------------------------------------------------------------------------


class InputPanel(BaseModel):
    """The recruiter's question that opened the turn."""

    text: str
    received_at: datetime | None = None


class RoutePanel(BaseModel):
    """Router/classifier outcome — the decision generic tracers can't show.

    ``confidence`` (0.0–1.0) is the signal that selects the deterministic
    fast-path vs. the full ADK reasoning loop — surfacing it is the whole point
    of a pipeline-shaped view.
    """

    agent_name: str | None
    sub_intent: str | None
    confidence: float | None
    rationale: str | None = None


class QueryPlanPanel(BaseModel):
    """The MetricFlow query plan the specialist ran (metric names + group-bys).

    Empty on the deterministic fast-path in the private platform (a known
    substrate gap documented in the spec); always populated here from the
    ``tabi.query.execute`` span the mock pipeline emits.
    """

    plan_id: str | None = None
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time_window: str | None = None


class ToolCall(BaseModel):
    """One ADK ``FunctionTool`` invocation, with args and a response summary."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    response_summary: dict[str, Any] | None = None
    duration_ms: float | None = None
    errored: bool = False


class SpecialistPanel(BaseModel):
    """Which specialist(s) answered, the routing path, and the grounded text."""

    specialists_invoked: list[str] = Field(default_factory=list)
    was_synthesized: bool = False
    response_text: str | None = None
    # Path taken: fast_route / deterministic / adaptive / multi_agent.
    routing: str | None = None
    handoff_count: int | None = None


class SalvagePanel(BaseModel):
    """Only present on a fallback turn — the panel that *appears* on failure.

    A salvage panel surfacing only when the backstop fired is exactly the kind
    of decision a flattened span tree buries.
    """

    reason: str
    fallback_text: str | None = None


class NarrativePanel(BaseModel):
    """Storytelling output, when the turn produced a narrative deep-dive."""

    story_id: str | None = None
    story_title: str | None = None
    slide_count: int | None = None


class JudgeResult(BaseModel):
    """One LLM-as-judge / deterministic metric score against the turn.

    Rendered beside the human annotation rail — the side-by-side that powers the
    judge-vs-human Cohen's-κ alignment in the platform.
    """

    metric_name: str
    score: float | None
    threshold: float | None
    passed: bool | None
    rationale: str | None = None
    errored: bool = False


class AnnotationDetail(BaseModel):
    """A human label on the turn — the rail that closes the optimization loop.

    Annotating a turn ``fail`` is what an operator exports into an
    ADK-canonical ``*.evalset.json`` the CI gate then consumes unmodified.
    """

    label: AnnotationLabel
    category: str | None = None
    notes: str | None = None


class SpanNode(BaseModel):
    """One node in the per-turn OpenTelemetry span tree, rooted at ``tabi.turn``.

    The viewer's primary surface: the pipeline-shaped tree, each node carrying its
    structural ``attributes`` (queryable schema — ids, metric names, dims,
    counts, latency) and, in dev/eval only, a ``payload`` (tenant-bearing
    tool/LLM content) for a raw-JSON expander. The span-derived panels
    (``query_plan`` / ``tools`` / ``salvage``) are a flattened projection of this
    same tree.
    """

    span_id: str
    parent_span_id: str | None = None
    name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] | None = None
    children: list[SpanNode] = Field(default_factory=list)


class TurnTrace(BaseModel):
    """The full pipeline-shaped view of one turn.

    Every stage panel is ``None`` when that stage didn't fire in the turn — the
    UI renders only what happened, in the pipeline's order. ``span_tree`` is the
    raw OTel tree the panels are projected from (empty when capture is off).
    """

    # Identity
    turn_id: str
    conversation_id: str | None = None
    tenant_id: str | None = None
    started_at: datetime | None = None
    duration_ms: float | None = None

    # Stage panels, in pipeline order.
    input: InputPanel | None = None
    route: RoutePanel | None = None
    query_plan: QueryPlanPanel | None = None
    tools: list[ToolCall] = Field(default_factory=list)
    specialist: SpecialistPanel | None = None
    salvage: SalvagePanel | None = None
    narrative: NarrativePanel | None = None
    judges: list[JudgeResult] = Field(default_factory=list)
    annotation: AnnotationDetail | None = None
    span_tree: list[SpanNode] = Field(default_factory=list)

    @property
    def agent_error(self) -> bool:
        """True iff the turn ended on the salvage backstop."""
        return self.salvage is not None

    @property
    def judge_summary(self) -> JudgeSummary:
        """Rollup of all judge verdicts for the list view."""
        verdicts = [j.passed for j in self.judges if j.passed is not None]
        if not verdicts:
            return "none"
        if all(verdicts):
            return "pass"
        if not any(verdicts):
            return "fail"
        return "mixed"


# Resolve SpanNode's self-referential ``children`` forward ref (deferred by
# ``from __future__ import annotations``).
SpanNode.model_rebuild()
