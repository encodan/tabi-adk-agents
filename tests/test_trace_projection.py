"""Tests for the pipeline-shaped trace projection.

Demonstrates the observability innovation in a runnable form: a turn's spans
(the shape TABI emits in :mod:`core.spans`) flatten into the pipeline-shaped
:class:`~core.trace_panels.TurnTrace` the custom viewer renders — route,
query-plan, tools, specialist, salvage, narrative, judges — rather than a
generic flat span list.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from core import spans
from core.trace_panels import JudgeResult
from core.trace_projection import (
    ATTR_QUERY_GROUP_BY,
    ATTR_QUERY_METRICS,
    ATTR_QUERY_TIME_WINDOW,
    ATTR_ROUTING_PATH,
    ATTR_SALVAGE_CAUSE,
    ATTR_TOOL_ARGS,
    ATTR_TOOL_NAME,
    SPAN_EXECUTE_TOOL,
    project_turn,
)

T0 = datetime(2026, 6, 9, 12, 0, 0)


def _turn_root_span(**extra_attrs):
    attrs = {
        spans.ATTR_TURN_ID: "01TURN",
        spans.ATTR_CONVERSATION_ID: "01CONV",
        spans.ATTR_TENANT_ID: "tenant-x",
        spans.ATTR_PRIMARY_AGENT: "pipeline_analyst",
        spans.ATTR_SUB_INTENT: "pipeline-bottleneck",
        spans.ATTR_ROUTE_CONFIDENCE: 0.97,
        ATTR_ROUTING_PATH: "deterministic",
    }
    attrs.update(extra_attrs)
    return {
        "span_id": "root",
        "parent_span_id": None,
        "name": spans.SPAN_TURN,
        "started_at": T0,
        "ended_at": T0 + timedelta(seconds=9),
        "status": "OK",
        "attributes": attrs,
    }


def _query_span():
    return {
        "span_id": "q1",
        "parent_span_id": "root",
        "name": spans.SPAN_QUERY_EXECUTE,
        "started_at": T0 + timedelta(seconds=1),
        "ended_at": T0 + timedelta(seconds=2),
        "status": "OK",
        "attributes": {
            ATTR_QUERY_METRICS: ["time_in_stage", "conversion_rate"],
            ATTR_QUERY_GROUP_BY: "stage",
            ATTR_QUERY_TIME_WINDOW: "last_90_days",
        },
    }


def _tool_span():
    return {
        "span_id": "t1",
        "parent_span_id": "q1",
        "name": SPAN_EXECUTE_TOOL,
        "started_at": T0 + timedelta(seconds=1, milliseconds=100),
        "ended_at": T0 + timedelta(seconds=1, milliseconds=700),
        "status": "OK",
        "attributes": {
            ATTR_TOOL_NAME: "query_multiple_recruitment_metrics",
            ATTR_TOOL_ARGS: {"metrics": ["time_in_stage"]},
        },
    }


def test_projects_pipeline_shape_from_spans():
    trace = project_turn(
        [_turn_root_span(), _query_span(), _tool_span()],
        input_text="Where's our worst pipeline bottleneck?",
        response_text="Your worst bottleneck is the Phone Screen stage at 7.35 days.",
        judges=[
            JudgeResult(
                metric_name="hallucinations_v1",
                score=0.9,
                threshold=0.6,
                passed=True,
            )
        ],
    )

    # Identity threads through from the tabi.turn root span.
    assert trace.turn_id == "01TURN"
    assert trace.conversation_id == "01CONV"
    assert trace.tenant_id == "tenant-x"
    assert trace.duration_ms == 9000.0

    # Route panel — the confidence signal generic tracers hide.
    assert trace.route is not None
    assert trace.route.agent_name == "pipeline_analyst"
    assert trace.route.confidence == 0.97

    # Query plan flattened off the load-bearing tabi.query.execute span.
    assert trace.query_plan is not None
    assert trace.query_plan.metrics == ["time_in_stage", "conversion_rate"]
    assert trace.query_plan.dimensions == ["stage"]
    assert trace.query_plan.time_window == "last_90_days"

    # Tools.
    assert len(trace.tools) == 1
    assert trace.tools[0].name == "query_multiple_recruitment_metrics"
    assert trace.tools[0].duration_ms == 600.0
    assert trace.tools[0].errored is False

    # Specialist + input/response wiring.
    assert trace.specialist is not None
    assert trace.specialist.specialists_invoked == ["pipeline_analyst"]
    assert trace.specialist.routing == "deterministic"
    assert trace.input is not None and "bottleneck" in trace.input.text

    # No salvage / narrative on a healthy turn; judges roll up to pass.
    assert trace.salvage is None
    assert trace.narrative is None
    assert trace.agent_error is False
    assert trace.judge_summary == "pass"

    # The raw span tree is preserved and correctly parented.
    assert len(trace.span_tree) == 1
    root = trace.span_tree[0]
    assert root.name == spans.SPAN_TURN
    assert {c.name for c in root.children} == {spans.SPAN_QUERY_EXECUTE}
    assert root.children[0].children[0].name == SPAN_EXECUTE_TOOL


def test_salvage_panel_appears_only_on_fallback():
    salvage_span = {
        "span_id": "sv",
        "parent_span_id": "root",
        "name": spans.SPAN_SALVAGE,
        "started_at": T0 + timedelta(seconds=3),
        "ended_at": T0 + timedelta(seconds=3, milliseconds=200),
        "status": "OK",
        "attributes": {ATTR_SALVAGE_CAUSE: "llm_calls_limit_exceeded"},
        "payload": {"fallback_text": "I couldn't complete that — here's what I have."},
    }
    trace = project_turn([_turn_root_span(), salvage_span])

    assert trace.salvage is not None
    assert trace.salvage.reason == "llm_calls_limit_exceeded"
    assert trace.salvage.fallback_text is not None
    assert trace.agent_error is True


def test_degrades_to_empty_panels_when_spans_missing():
    # A fully-populated turn for the contrast assertion at the end.
    trace = project_turn([_turn_root_span(), _query_span()])
    # A turn that never classified — only the bare root span, no route attrs.
    bare = {
        "span_id": "root",
        "parent_span_id": None,
        "name": spans.SPAN_TURN,
        "started_at": T0,
        "ended_at": T0 + timedelta(seconds=1),
        "attributes": {spans.ATTR_TURN_ID: "01BARE"},
    }
    bare_trace = project_turn([bare])
    assert bare_trace.turn_id == "01BARE"
    assert bare_trace.route is None
    assert bare_trace.query_plan is None
    assert bare_trace.tools == []
    assert bare_trace.specialist is None
    assert bare_trace.judge_summary == "none"
    # Sanity: the populated turn above still has a route.
    assert trace.route is not None
