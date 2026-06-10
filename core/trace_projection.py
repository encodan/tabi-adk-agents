"""Project a turn's OpenTelemetry span tree into the pipeline-shaped trace model.

This is the *reader* side of the observability contract: it takes the spans
emitted by :mod:`core.spans` (plus ADK's own ``execute_tool`` spans) and flattens
them into the :class:`~core.trace_panels.TurnTrace` the viewer renders — route,
query-plan, tools, specialist, salvage, narrative, judges.

In the full TABI platform this projection runs server-side in the trace
projection service (excluded from this showcase) over a three-store
fan-out (a BigQuery log
sink × Firestore × Postgres ``turn_summaries``), degrade-isolated so a missing
store yields empty panels rather than a 500. Here it's a pure, in-memory function
over span dicts — no I/O — so the *shape* of the projection is preserved and
unit-testable without the proprietary data path.

A span dict is the lightweight, store-agnostic mirror of an OTel span::

    {
        "span_id": "s1",
        "parent_span_id": None,            # None for the tabi.turn root
        "name": "tabi.turn",               # see core.spans SPAN_* constants
        "started_at": datetime | None,
        "ended_at": datetime | None,
        "status": "OK" | "ERROR" | None,
        "attributes": {"tabi.turn_id": "...", ...},   # structural (always)
        "payload": {...} | None,            # tenant-bearing (dev/eval only)
    }
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from core import spans
from core.trace_panels import (
    InputPanel,
    NarrativePanel,
    QueryPlanPanel,
    RoutePanel,
    SalvagePanel,
    SpanNode,
    SpecialistPanel,
    ToolCall,
    TurnTrace,
)

# ADK's tool-execution span name + the attribute keys the query/tool callers set
# on their spans. Mirrors what the emitting side writes (see core.spans); named
# once here so the read and write sides don't drift silently.
SPAN_EXECUTE_TOOL = "execute_tool"

ATTR_QUERY_PLAN_ID = "tabi.query.plan_id"
ATTR_QUERY_METRICS = "tabi.query.metrics"
ATTR_QUERY_GROUP_BY = "tabi.query.group_by"
ATTR_QUERY_TIME_WINDOW = "tabi.query.time_window"

ATTR_TOOL_NAME = "tabi.tool.name"
ATTR_TOOL_ARGS = "tabi.tool.arguments"
ATTR_TOOL_RESPONSE = "tabi.tool.response_summary"

ATTR_ROUTING_PATH = "tabi.routing_path"
ATTR_SALVAGE_CAUSE = "tabi.salvage.cause"
ATTR_STORY_ID = "tabi.story_id"
ATTR_STORY_TITLE = "tabi.story_title"
ATTR_SLIDE_COUNT = "tabi.slide_count"


def _duration_ms(started: datetime | None, ended: datetime | None) -> float | None:
    if started is None or ended is None:
        return None
    return (ended - started).total_seconds() * 1000.0


def _as_list(value: Any) -> list[str]:
    """Coerce an attribute that may be a list or comma-joined string to a list.

    OTel attributes are primitives or homogeneous primitive sequences, so a
    metric list can arrive either way depending on the exporter.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def _build_nodes(span_dicts: list[Mapping[str, Any]]) -> tuple[list[SpanNode], dict[str, SpanNode]]:
    """Build the SpanNode tree, returning (roots, by_id)."""
    by_id: dict[str, SpanNode] = {}
    for s in span_dicts:
        node = SpanNode(
            span_id=s["span_id"],
            parent_span_id=s.get("parent_span_id"),
            name=s["name"],
            started_at=s.get("started_at"),
            ended_at=s.get("ended_at"),
            duration_ms=_duration_ms(s.get("started_at"), s.get("ended_at")),
            status=s.get("status"),
            attributes=dict(s.get("attributes") or {}),
            payload=s.get("payload"),
        )
        by_id[node.span_id] = node

    roots: list[SpanNode] = []
    for node in by_id.values():
        parent = by_id.get(node.parent_span_id) if node.parent_span_id else None
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)
    return roots, by_id


def _first(nodes: Iterable[SpanNode], name: str) -> SpanNode | None:
    return next((n for n in nodes if n.name == name), None)


def project_turn(
    span_dicts: list[Mapping[str, Any]],
    *,
    input_text: str | None = None,
    response_text: str | None = None,
    judges: list[Any] | None = None,
    annotation: Any | None = None,
) -> TurnTrace:
    """Flatten a turn's spans into the pipeline-shaped :class:`TurnTrace`.

    ``input_text`` / ``response_text`` / ``judges`` / ``annotation`` come from
    sources outside the span tree in the real system (the SSE log, the eval
    store, the annotation table); they're passed in here so the projector stays
    a pure function of its inputs.

    Degrade contract: a stage with no span simply yields a ``None`` panel — the
    turn still projects. This mirrors the platform's "a missing store yields an
    empty panel, never a 500" guarantee.
    """
    nodes = list(_build_nodes(span_dicts)[1].values())

    turn = _first(nodes, spans.SPAN_TURN)
    attrs = turn.attributes if turn else {}

    trace = TurnTrace(
        turn_id=str(attrs.get(spans.ATTR_TURN_ID, "")),
        conversation_id=attrs.get(spans.ATTR_CONVERSATION_ID),
        tenant_id=attrs.get(spans.ATTR_TENANT_ID),
        started_at=turn.started_at if turn else None,
        duration_ms=turn.duration_ms if turn else None,
        span_tree=_build_nodes(span_dicts)[0],
    )

    if input_text is not None:
        trace.input = InputPanel(text=input_text, received_at=turn.started_at if turn else None)

    # Route — sourced from the tabi.turn root attrs (stamped at classification).
    if turn and attrs.get(spans.ATTR_PRIMARY_AGENT) is not None:
        trace.route = RoutePanel(
            agent_name=attrs.get(spans.ATTR_PRIMARY_AGENT),
            sub_intent=attrs.get(spans.ATTR_SUB_INTENT),
            confidence=attrs.get(spans.ATTR_ROUTE_CONFIDENCE),
        )

    # Query plan — from the load-bearing tabi.query.execute span (or metrics_plan).
    query_span = _first(nodes, spans.SPAN_QUERY_EXECUTE) or _first(nodes, spans.SPAN_METRICS_PLAN)
    if query_span:
        qa = query_span.attributes
        trace.query_plan = QueryPlanPanel(
            plan_id=qa.get(ATTR_QUERY_PLAN_ID),
            metrics=_as_list(qa.get(ATTR_QUERY_METRICS)),
            dimensions=_as_list(qa.get(ATTR_QUERY_GROUP_BY)),
            time_window=qa.get(ATTR_QUERY_TIME_WINDOW),
        )

    # Tools — every ADK execute_tool span.
    for node in nodes:
        if node.name != SPAN_EXECUTE_TOOL:
            continue
        ta = node.attributes
        trace.tools.append(
            ToolCall(
                name=str(ta.get(ATTR_TOOL_NAME, "tool")),
                arguments=ta.get(ATTR_TOOL_ARGS) or {},
                response_summary=ta.get(ATTR_TOOL_RESPONSE),
                duration_ms=node.duration_ms,
                errored=(node.status == "ERROR"),
            )
        )

    # Specialist — agent(s) + routing path + compound flag.
    if turn and attrs.get(spans.ATTR_PRIMARY_AGENT) is not None:
        is_compound = bool(attrs.get(spans.ATTR_IS_COMPOUND))
        invoked = [attrs[spans.ATTR_PRIMARY_AGENT]]
        trace.specialist = SpecialistPanel(
            specialists_invoked=invoked,
            was_synthesized=is_compound,
            response_text=response_text,
            routing=attrs.get(ATTR_ROUTING_PATH),
        )

    # Salvage — only present when the backstop fired.
    salvage_span = _first(nodes, spans.SPAN_SALVAGE)
    if salvage_span:
        trace.salvage = SalvagePanel(
            reason=str(salvage_span.attributes.get(ATTR_SALVAGE_CAUSE, "salvage")),
            fallback_text=(salvage_span.payload or {}).get("fallback_text"),
        )

    # Narrative — only present on storytelling turns.
    narrative_span = _first(nodes, spans.SPAN_NARRATIVE)
    if narrative_span:
        na = narrative_span.attributes
        trace.narrative = NarrativePanel(
            story_id=na.get(ATTR_STORY_ID),
            story_title=na.get(ATTR_STORY_TITLE),
            slide_count=na.get(ATTR_SLIDE_COUNT),
        )

    if judges:
        trace.judges = list(judges)
    if annotation is not None:
        trace.annotation = annotation

    return trace
