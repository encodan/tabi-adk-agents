"""TABI manual OTel spans for the pipeline stages ADK does not instrument.

ADK 2.1.0 auto-emits ``invocation`` / ``invoke_agent`` / ``call_llm`` /
``execute_tool`` spans, but several TABI stages run *outside* ADK and so carry
no span today (verified empirically):

* the **turn root** — a multi-agent turn otherwise scatters across *N* separate
  ADK traces (one ``invocation`` per agent) with no common key; ``tabi.turn`` is
  the only thing that unifies them under one ``turn_id``;
* **router classify**, the **deterministic/adaptive query execution**, **metrics
  planning**, **salvage**, and **narrative** — all bypass ADK.

This module adds thin, no-op-safe spans for those gaps. Like
:mod:`tabi_analytics.core.trace_correlation`, ``analytics`` only *gets* a tracer;
the global ``TracerProvider`` is owned by the server-side tracing setup
(excluded from this showcase).
When no provider is registered (local without tracing, unit tests without an
exporter), ``start_as_current_span`` is a no-op, so every helper here is always
safe to call and adds ~nothing on the hot path.

**PII (dev/eval-only content capture).** *Structural* attributes —
ids, metric **names**, ``group_by`` **dimensions**, counts, latency — are schema,
not tenant data, and are always set. *Payload* attributes that can embed tenant
values — query result rows, generated SQL, filter values, narrative text — are
attached only when :func:`content_capture_enabled` is true (dev/eval). Flipping
production to capture content is gated on a dedicated redaction path, never on
this module.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

__all__ = [
    "TRACER",
    "content_capture_enabled",
    "turn_span",
    "set_turn_route",
    "set_turn_outcome",
    "router_classify_span",
    "query_execute_span",
    "metrics_plan_span",
    "salvage_span",
    "narrative_span",
    "set_attrs",
    "set_payload",
    "PAYLOAD_ATTR_KEYS",
    "SPAN_TURN",
    "SPAN_ROUTER_CLASSIFY",
    "SPAN_QUERY_EXECUTE",
    "SPAN_METRICS_PLAN",
    "SPAN_SALVAGE",
    "SPAN_NARRATIVE",
    "ATTR_TURN_ID",
    "ATTR_CONVERSATION_ID",
    "ATTR_TENANT_ID",
    "ATTR_SUB_INTENT",
    "ATTR_ROUTE_CONFIDENCE",
    "ATTR_PRIMARY_AGENT",
    "ATTR_IS_COMPOUND",
    "ATTR_AGENT_ERROR",
]

# One tracer for the whole analytics package. ADK's own span names
# are left untouched; these names fill the instrumentation gaps,
# namespaced under ``tabi.``.
TRACER = trace.get_tracer("tabi.analytics")

SPAN_TURN = "tabi.turn"
SPAN_ROUTER_CLASSIFY = "tabi.router.classify"
SPAN_QUERY_EXECUTE = "tabi.query.execute"
SPAN_METRICS_PLAN = "tabi.metrics_plan"
SPAN_SALVAGE = "tabi.salvage"
SPAN_NARRATIVE = "tabi.narrative"

# ``tabi.turn`` root-span attribute keys — the single source of truth shared by
# the span *writers* below (``turn_span`` / ``set_turn_route`` /
# ``set_turn_outcome``) and the span *reader* that rebuilds a ``turn_summaries``
# row from these spans locally (``projection/spans_to_turn_summaries``). Naming
# the keys once stops the read and write sides drifting apart silently.
ATTR_TURN_ID = "tabi.turn_id"
ATTR_CONVERSATION_ID = "tabi.conversation_id"
ATTR_TENANT_ID = "tabi.tenant_id"
ATTR_SUB_INTENT = "tabi.sub_intent"
ATTR_ROUTE_CONFIDENCE = "tabi.route_confidence"
ATTR_PRIMARY_AGENT = "tabi.primary_agent"
ATTR_IS_COMPOUND = "tabi.is_compound"
ATTR_AGENT_ERROR = "tabi.agent_error"

# Single source of truth for the ``tabi.*`` attribute keys whose values are
# tenant-bearing payloads (written via :func:`set_payload`, gated to dev/eval).
# The span store routes these to its ``payload_blob`` column rather than the
# queryable ``attributes`` column — importing this set keeps the exporter from
# silently drifting out of sync when a new payload key is added here.
PAYLOAD_ATTR_KEYS: frozenset[str] = frozenset(
    {
        "tabi.query.specs",
        "tabi.query.results",
    }
)

_PROD_ENVS = {"production", "prod"}


def content_capture_enabled() -> bool:
    """True in dev/eval, False in production (dev/eval-only capture).

    Mirrors :mod:`tabi_core.log_redact`'s production detection so the two stay
    in lock-step. Raw tenant-bearing payloads (rows, SQL, filter values,
    narrative text) are attached to spans only when this returns true; the
    structural attributes are always set regardless.

    Read per call (not cached) so an env override takes effect without a restart
    and so tests can flip ``TABI_ENV`` — matching the codebase convention (e.g.
    ``execution_engine._query_plan_threshold``). The cost is one ``getenv`` and
    only on the recording-gated path, so it is negligible.
    """
    return os.getenv("TABI_ENV", "").strip().lower() not in _PROD_ENVS


def set_attrs(span: Span, attrs: Mapping[str, Any]) -> None:
    """Set non-``None`` attributes on a recording span.

    OTel only accepts primitives and homogeneous sequences of primitives;
    ``None`` is dropped (OTel would reject it) and callers pass already-typed
    values. Dicts/objects must be JSON-encoded by the caller (see
    :func:`set_payload`).
    """
    if not span.is_recording():
        return
    for key, value in attrs.items():
        if value is None:
            continue
        span.set_attribute(key, value)


def set_payload(span: Span, key: str, value: Any) -> None:
    """Attach a JSON-encoded *payload* attribute — but only in dev/eval.

    Use for anything that can embed tenant data (result rows, generated SQL,
    filter values, narrative text). In production this is a no-op until the B4
    redaction path lands. ``value`` is JSON-serialised defensively; a value that
    cannot be serialised is skipped rather than raising on the hot path.
    """
    if not span.is_recording() or not content_capture_enabled():
        return
    try:
        span.set_attribute(key, json.dumps(value, default=str))
    except (TypeError, ValueError):
        return


@contextmanager
def turn_span(
    *,
    turn_id: str,
    conversation_id: str | None = None,
    tenant_id: str | None = None,
) -> Iterator[Span]:
    """Open the ``tabi.turn`` root span (the turn-root option).

    Opened by the execution engine right after ``turn_id`` is minted, so it
    is the parent of every per-agent ADK ``invocation`` span — the single key
    that ties a multi-agent turn's N traces back to one ``turn_id``. Carries
    only ids (no PII). Route attributes are added later via :func:`_set_attrs`
    on the yielded span once classification completes.
    """
    with TRACER.start_as_current_span(SPAN_TURN) as span:
        set_attrs(
            span,
            {
                ATTR_TURN_ID: turn_id,
                ATTR_CONVERSATION_ID: conversation_id,
                ATTR_TENANT_ID: tenant_id,
            },
        )
        yield span


def set_turn_route(
    span: Span,
    *,
    sub_intent: str | None = None,
    route_confidence: float | None = None,
    primary_agent: str | None = None,
    is_compound: bool | None = None,
) -> None:
    """Add route outcome to the ``tabi.turn`` span once classification is known.

    Separate from :func:`turn_span` because ``turn_id`` exists before routing
    but the route does not — these attrs are stamped mid-turn (all structural,
    no PII).
    """
    set_attrs(
        span,
        {
            ATTR_SUB_INTENT: sub_intent,
            ATTR_ROUTE_CONFIDENCE: route_confidence,
            ATTR_PRIMARY_AGENT: primary_agent,
            ATTR_IS_COMPOUND: is_compound,
        },
    )


def set_turn_outcome(span: Span, *, agent_error: bool) -> None:
    """Stamp the turn's terminal salvage/error outcome on the ``tabi.turn`` span.

    Set at turn finalisation — after the salvage/validation channels resolve —
    parallel to :func:`set_turn_route` (which is stamped mid-turn at routing).
    ``agent_error`` mirrors :meth:`AgentSession.get_last_agent_error`, the only
    one of the nine ``turn_summaries`` fields not otherwise present on a span.
    Capturing it here lets a span-only consumer (the local
    ``spans → turn_summaries`` projection) reconstruct the list-view row without
    the BQ/Firestore pipeline. Always written (even ``False``, which OTel keeps —
    only ``None`` is dropped); the projection COALESCEs a *missing* attribute
    (spans written before this field existed) back to ``False``.
    """
    set_attrs(span, {ATTR_AGENT_ERROR: agent_error})


@contextmanager
def router_classify_span() -> Iterator[Span]:
    """``tabi.router.classify`` — the pre-runner classifier LLM call."""
    with TRACER.start_as_current_span(SPAN_ROUTER_CLASSIFY) as span:
        yield span


@contextmanager
def query_execute_span(
    *, agent: str | None = None, sub_intent: str | None = None
) -> Iterator[Span]:
    """``tabi.query.execute`` — the deterministic/adaptive query stage.

    This is the load-bearing span: on the deterministic/adaptive path the
    queries run via ``execute_queries_directly`` *outside* ADK, so this is the
    **sole** source for the viewer's Query Plan / Tools panels on the common
    path. Structural attrs (metric names, group_by, counts, latency) are set by
    the caller via :func:`_set_attrs`; raw rows / SQL via :func:`set_payload`.
    """
    with TRACER.start_as_current_span(SPAN_QUERY_EXECUTE) as span:
        set_attrs(span, {"tabi.agent": agent, "tabi.sub_intent": sub_intent})
        yield span


@contextmanager
def metrics_plan_span(*, agent: str | None = None, sub_intent: str | None = None) -> Iterator[Span]:
    """``tabi.metrics_plan`` — the adaptive metrics-planner LLM call.

    Another direct-genai call with no ADK span (a gap the spike found beyond
    the originally enumerated set).
    """
    with TRACER.start_as_current_span(SPAN_METRICS_PLAN) as span:
        set_attrs(span, {"tabi.agent": agent, "tabi.sub_intent": sub_intent})
        yield span


@contextmanager
def salvage_span(*, cause: str | None = None) -> Iterator[Span]:
    """``tabi.salvage`` — the salvage-signal stage."""
    with TRACER.start_as_current_span(SPAN_SALVAGE) as span:
        set_attrs(span, {"tabi.salvage.cause": cause})
        yield span


@contextmanager
def narrative_span(*, story_type: str | None = None) -> Iterator[Span]:
    """``tabi.narrative`` — storytelling generation.

    Instrumented at the storytelling service's story-generation entry point
    (NOT reachable via ``session.ask()`` — confirmed by the trace spike).
    """
    with TRACER.start_as_current_span(SPAN_NARRATIVE) as span:
        set_attrs(span, {"tabi.story_type": story_type})
        yield span
