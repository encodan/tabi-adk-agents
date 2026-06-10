"""Per-session ADK plugin factory.

``build_session_plugins()`` is the single, agent-agnostic definition of which
plugins every ``App`` in a session carries — not parameterised by specialist
name so the storytelling seam can register through the same accessor.

Plugin order is fixed so later layers slot in without reshuffling: index 0 is
the ``GuardrailPlugin`` (always attached), then the flag-gated BQ
analytics plugin, then the ``ReflectAndRetryToolPlugin``, then the
``SalvagePlugin`` at index -1.

Guardrail ordering invariant (load-bearing, by design):
``GuardrailPlugin`` MUST run before any other plugin on
``before_tool_callback``. ``PluginManager._run_callbacks`` short-circuits on
the first non-``None`` return; the guardrail's sentinel return for a
tenant-scope violation is non-``None``, so reordering would let a later
plugin substitute a tool result before the guardrail has inspected the args.
Equivalent argument for ``on_user_message_callback``: index-0 ensures the
injection heuristic sees the original message before any downstream
transformer could mutate it.

Retry/salvage ordering invariant (load-bearing, by design):
``ReflectAndRetryToolPlugin`` runs BEFORE ``SalvagePlugin`` on the
``on_tool_error_callback`` chain. ADK's ``PluginManager._run_callbacks``
short-circuits on the first non-``None`` return: the retry plugin returns
reflection guidance (a non-``None`` dict) so the chain stops there, the
tool retries, and the model's recovery is not poisoned by a premature
salvage-signal write from ``SalvagePlugin`` (which mutates the
per-branch ``TurnErrorSink`` in place — a permanent write with no
clearing mechanism). Reversing the order would cause every retried-then-
recovered tool error to spuriously flag ``agent_error=True`` at the
``AgentSession.get_last_agent_error()`` union seam.

Privacy — metadata-only is enforced in code, never inherited from a default.
An audit of the *pinned ADK 1.33 wheel* (not its documented pseudo-code) found that
``content_formatter`` alone is **not** sufficient: the plugin writes tenant
data through three sinks, only one of which the formatter covers.

    ┌─────────────────────────────────┬──────────────────────────────────────┐
    │ Sink                            │ Lever (this module sets all of them) │
    ├─────────────────────────────────┼──────────────────────────────────────┤
    │ ``content`` (prompts/responses/ │ ``content_formatter`` → redactor      │
    │ tool results/agent response)    │                                      │
    │ ``content_parts`` (multi-modal) │ ``log_multi_modal_content=False``    │
    │ ``attributes.session_metadata`` │ ``log_session_metadata=False`` — the │
    │ → full ``dict(session.state)``  │ formatter does NOT cover this; ADK   │
    │ (TABI state = query results,    │ truncates but never redacts it       │
    │ entity context)                 │ (wheel: ``_enrich_attributes``)      │
    │ ``attributes.state_delta`` on   │ ``event_denylist=["STATE_DELTA"]`` — │
    │ ``STATE_DELTA`` events          │ not gated by ``log_session_metadata``│
    │                                 │ at all; only suppressible by denylist│
    │                                 │ (wheel: ``on_event_callback``). Carries│
    │                                 │ no cache/latency signal we need.     │
    └─────────────────────────────────┴──────────────────────────────────────┘

Residual (accepted, documented): the ``error_message`` column is
``str(error)`` framework exception text with no config hook to redact it. Our
own tool errors are already structured/bounded; framework error strings do not
echo prompt/response bodies. If that ever changes upstream it is caught by the
fail-closed field test plus a tenant-scoped review before the flag is enabled
(the privacy invariant).

A privacy-lever field test fails closed if **any** of the four privacy
levers below is renamed on a future ADK bump, forcing re-verification rather
than a silent regression to content capture.

Known limit of that guard (conscious acceptance, not an oversight): it is
*structural* — it asserts the four field **names** still exist, not that their
**semantics** are unchanged. An ADK bump that keeps a name but alters its
behaviour (e.g. ``log_session_metadata=False`` no longer suppressing the
``session.state`` sink) would pass the field test yet regress privacy. The
semantic backstop is deliberately out-of-band: the runtime flag defaults
**off** and a tenant-scoped privacy re-audit against the newly pinned wheel is
required before it is enabled on any ADK bump (the pin cross-check plus the
privacy invariant). This file's job is to make that audit *unskippable*,
not to replace it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import structlog

from config import get_config
from core.gcp_project import resolve_gcp_project

if TYPE_CHECKING:
    from google.adk.plugins import BasePlugin

logger = structlog.get_logger(__name__)

# Replaces the entire prompt/response payload. The plugin still records
# metadata (event_type, model, token/usage incl. cache-hit counts, latency,
# trace/span ids, timestamps) — those come from the event, NOT from the
# content the formatter sees, so redacting content does not strip the
# cache-hit signal the analytics capture needs.
_METADATA_ONLY_MARKER = "[redacted: metadata-only capture]"

# Event type whose payload is a raw session-state diff (tenant data) with no
# cache/latency signal. Dropped wholesale via ``event_denylist`` — see the
# module docstring's sink table.
_STATE_DELTA_EVENT = "STATE_DELTA"

# The four ``BigQueryLoggerConfig`` levers this module relies on to keep
# capture metadata-only. The fail-closed test asserts every one still exists
# on the installed wheel; a rename here without re-audit would silently
# re-open a content sink.
PRIVACY_CRITICAL_CONFIG_FIELDS: tuple[str, ...] = (
    "content_formatter",
    "log_multi_modal_content",
    "log_session_metadata",
    "event_denylist",
)


def _metadata_only_content_formatter(_raw_content: Any, _event_type: str) -> str:
    """Strip prompt/response content for B2B tenant-ATS privacy. Returning a
    constant ``str`` is parser-safe (the plugin's parser smart-truncates a
    bare string) and keeps the row's metadata columns intact."""
    return _METADATA_ONLY_MARKER


def _resolve_bq_analytics_target() -> tuple[str, str]:
    """Resolve ``(project_id, dataset_id)`` for the BQ analytics sink, or
    raise ``RuntimeError`` if the flag is on but the infra contract (Terraform
    dataset + surfaced env) is unmet.

    Single source of truth for the precondition so the API lifespan can
    fail-fast at **startup** (see :func:`validate_bq_agent_analytics_config`)
    rather than letting every first turn 500 — while ``build_session_plugins``
    keeps the same guard as defence-in-depth for non-API entrypoints.
    """
    project_id = resolve_gcp_project()
    dataset_id = os.environ.get("BQ_ANALYTICS_DATASET_ID")
    if not project_id or not dataset_id:
        raise RuntimeError(
            "bq_agent_analytics_enabled=true but "
            f"project_id={project_id!r} / BQ_ANALYTICS_DATASET_ID="
            f"{dataset_id!r} unresolved — provision the Terraform dataset "
            "and surface BQ_ANALYTICS_DATASET_ID to the service first."
        )
    return project_id, dataset_id


def validate_bq_agent_analytics_config() -> None:
    """Startup fail-fast for the BQ Agent Analytics contract.

    No-op when the flag is off. When on, asserts (a) the infra precondition
    (project + dataset resolvable) and (b) every privacy-critical
    ``BigQueryLoggerConfig`` field this module sets still exists on the
    installed ADK wheel. Called from the API lifespan so a misconfigured
    deploy fails the rollout instead of 500-ing every chat turn.

    Raises ``RuntimeError`` (precondition) or ``AttributeError`` via the
    dataclass field check (ADK rename) — both block startup by design.
    """
    cfg = get_config()
    if not cfg.feature_flags.bq_agent_analytics_enabled:
        return

    _resolve_bq_analytics_target()

    import dataclasses

    from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig

    fields = {f.name for f in dataclasses.fields(BigQueryLoggerConfig)}
    missing = [f for f in PRIVACY_CRITICAL_CONFIG_FIELDS if f not in fields]
    if missing:
        raise RuntimeError(
            "ADK BigQueryLoggerConfig is missing privacy-critical field(s) "
            f"{missing} on the installed wheel — metadata-only capture can no "
            "longer be guaranteed. Re-audit the plugin's content sinks before "
            "enabling bq_agent_analytics (the privacy invariant)."
        )


def build_session_plugins() -> list[BasePlugin]:
    """Return the ordered plugin list for a session's ``App`` objects.

    Built **once per session** (the BQ plugin holds a BigQuery client + async
    queue — see ``AgentSession`` caching); never per turn / per attempt /
    per run path.

    Order (load-bearing, see module docstring):
    1. ``GuardrailPlugin`` — fail-closed tenant-scope assertion + injection heuristics
    2. ``BigQueryAgentAnalyticsPlugin`` (only when ``bq_agent_analytics_enabled``)
    3. ``ReflectAndRetryToolPlugin`` — recovers tool errors before salvage observes them
    4. ``SalvagePlugin`` — last, observes whatever the retry plugin didn't recover
    """
    plugins: list[BasePlugin] = []

    # Guardrail plugin at index 0 by design. Always attached; the
    # tenant_scope_guardrail_enabled feature flag inside the plugin gates the
    # tenant-scope half (off only for the bucketing-independent isolation
    # comparison). Importing here (not at module top) keeps test-collection of
    # this module independent of the guardrail import graph.
    from core.guardrail_plugin import GuardrailPlugin

    plugins.append(GuardrailPlugin())
    logger.info(
        "guardrail_plugin_attached",
        position="index_0",
        callbacks=("before_tool_callback", "on_user_message_callback"),
    )

    cfg = get_config()
    if cfg.feature_flags.bq_agent_analytics_enabled:
        # Local import: only hard-require the BQ plugin module when the flag
        # is actually on, and surface a clear, version-attributable error if
        # the path/constructor moved on an ADK bump (the pin cross-check).
        from google.adk.plugins.bigquery_agent_analytics_plugin import (
            BigQueryAgentAnalyticsPlugin,
            BigQueryLoggerConfig,
        )

        # Fail loud (not silently log nothing) if the infra contract is unmet.
        # In the API process this has already passed at startup
        # (validate_bq_agent_analytics_config); kept here for non-API callers.
        project_id, dataset_id = _resolve_bq_analytics_target()

        plugins.append(
            BigQueryAgentAnalyticsPlugin(
                project_id=project_id,
                dataset_id=dataset_id,
                config=BigQueryLoggerConfig(
                    # Privacy defaults expressed in code by design, never
                    # inherited. All four levers from the module docstring's
                    # sink table — content_formatter alone is insufficient.
                    content_formatter=_metadata_only_content_formatter,
                    log_multi_modal_content=False,
                    # Closes the session-state sink: the formatter never sees
                    # attributes.session_metadata.state (full dict(session.state)
                    # = TABI query results / entity context).
                    log_session_metadata=False,
                    # Closes the state-delta sink: STATE_DELTA events carry a
                    # raw session-state diff and no cache/latency signal, so
                    # dropping them entirely is correct, not lossy.
                    event_denylist=[_STATE_DELTA_EVENT],
                ),
            )
        )
        logger.info(
            "bq_agent_analytics_plugin_attached",
            project_id=project_id,
            dataset_id=dataset_id,
            capture="metadata_only",
            session_metadata_logged=False,
            denied_events=[_STATE_DELTA_EVENT],
        )

    # Append ADK's built-in ReflectAndRetryToolPlugin (ordering
    # rationale in the module docstring). ``throw_exception_if_retry_exceeded
    # =False`` is the load-bearing choice — raising would be wrapped in
    # ``RuntimeError`` by ``PluginManager._run_callbacks`` and lose the
    # original exception type; the non-throw path returns "stop using this
    # tool" guidance so the model can adapt, and the empty-text terminal case
    # is caught downstream by the orchestrator's per-branch path via
    # its branch-error marker.
    from google.adk.plugins.reflect_retry_tool_plugin import (
        ReflectAndRetryToolPlugin,
        TrackingScope,
    )

    plugins.append(
        ReflectAndRetryToolPlugin(
            max_retries=3,
            throw_exception_if_retry_exceeded=False,
            tracking_scope=TrackingScope.INVOCATION,
        )
    )
    logger.info(
        "reflect_retry_plugin_attached",
        max_retries=3,
        throw_exception_if_retry_exceeded=False,
        tracking_scope="invocation",
    )

    # By design, append SalvagePlugin at the end of the list. Observes
    # cap-hit (model error) + tool-error callbacks; writes Channel A
    # (TurnErrorSink) + Channel B (SalvagePayload) signals so the
    # application layer (specialist runner / orchestrator) builds the
    # correct ``SpecialistResponse(agent_error=True)``. Always attached
    # — there is no feature flag because the plugin is purely additive
    # observation (returns None from both callbacks, ADK
    # propagates the underlying exception unchanged). The salvage
    # behaviour the application layer carries is unchanged; the plugin's
    # contribution is to mark the sink + populate the payload so the
    # other error layers (the turn watchdog, the tenant-scope guardrail)
    # converge on the same seam.
    from core.salvage_plugin import SalvagePlugin

    plugins.append(SalvagePlugin())
    logger.info(
        "salvage_plugin_attached",
        position="end_of_list",
        callbacks=("on_model_error_callback", "on_tool_error_callback"),
    )

    return plugins
