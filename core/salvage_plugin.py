"""Salvage plugin (cap-hit + tool-error entry points).

Implements the ``BasePlugin`` subclass that relocates
cap-hit and tool-error salvage detection from the specialist runner's
inline ``try/except`` into the ADK plugin
boundary. By design:

> "The plugin must expose the ``SpecialistResponse`` builder +
>  channel-correct write as a single callable seam invoked by: (a)
>  ``on_model_error_callback`` (cap-hit / model error), (b)
>  ``on_tool_error_callback`` (tool error, incl. the tenant-scope
>  raise), and (c) the runner-owned turn watchdog..."

This plugin wires entry points (a) and (b); the runner-owned turn
watchdog wires (c) (its ``CancelledError`` invokes the same shared
builder seam directly, never via the error callbacks).

## Channel discipline

Two channels carry ``agent_error`` to the union read seam
``AgentSession.get_last_agent_error()``:

- **Channel A** — boolean flag mutated on the per-branch
  :class:`TurnErrorSink` (when present, i.e. orchestrated fan-out). The
  orchestrator's existing post-gather set-AND-clear reduction reads
  the sink (the plugin does NOT change the reduction itself — it is
  preserved verbatim).

- **Channel B** — structured ``SpecialistResponse.agent_error=True`` on
  the per-run response object. Application-layer salvage construction
  (the specialist runner's collection path and the orchestrator's
  per-branch path) builds this from the
  :class:`SalvagePayload` written by the plugin.

The plugin writes **both** in every error callback. The sink mutation
is no-op when not in fan-out (sink is ``None``); the payload write is
unconditional. Application-layer readers consume whichever they own.

## Rescue semantics

Both error callbacks return ``None`` (do NOT rescue with a fabricated
``LlmResponse``). Rationale:

- ADK propagates the exception when the callback returns ``None``.
- The specialist runner's collection path has an existing
  ``except LlmCallsLimitExceededError`` at the runner-call site (now
  defensive — the plugin observes the cap *before* the exception
  propagates, but propagation still happens). With the plugin it picks up
  the payload to know *why* the run came back empty/short, even if the
  raw exception's ``cap_exhausted`` path also fires.
- Tool errors today either fail the tool (returned to the model as a
  tool-call error) or propagate upward. The plugin observing them and
  marking the sink/payload is additive — no behavioural change to the
  existing tool-error handling.

A future revision could rescue via ``LlmResponse``, but that's a
larger seam change (the specialist runner's salvage-text construction
would need to short-circuit). The intent here is "relocation, not
redesign" — the plugin observes; the application layer
constructs. Both ends now know salvage fired, via the shared payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from google.adk.plugins.base_plugin import BasePlugin

from core.salvage_payload import (
    SalvageCause,
    SalvagePayload,
    set_pending_salvage_payload,
)
from core.spans import salvage_span
from core.spans import set_attrs as set_span_attrs
from core.turn_error_sink import get_current_turn_error_sink

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext


logger = structlog.get_logger(__name__)


# Public cause discriminants — exported so the watchdog call sites in
# the specialist runner (turn watchdog) and the orchestrator
# (per-branch watchdog) use the same
# constant as the ADK callbacks rather than re-typing the literal.
# Typed as ``SalvageCause`` so mypy narrows at every call site; adding a
# fourth cause is a single conscious edit here.
CAUSE_MODEL_ERROR: SalvageCause = "model_error"
CAUSE_TOOL_ERROR: SalvageCause = "tool_error"
CAUSE_WATCHDOG: SalvageCause = "watchdog"


# Salvage prose constants — single source of truth for the user-visible
# fallback text the watchdog paths emit. Two scopes:
#
# - ``SALVAGE_TEXT_TURN_TIMEOUT`` — turn-level fallback. Whole turn was
#   cancelled (runner-owned turn watchdog). Asks the user to
#   retry/rephrase since no part of the turn succeeded.
# - ``SALVAGE_TEXT_BRANCH_TIMEOUT`` — per-branch fallback. Only one
#   branch was cancelled (orchestrator's per-branch watchdog);
#   siblings may have produced usable text the
#   user will still see, so the prose is narrower ("part of the
#   analysis") and does NOT prompt a full retry.
SALVAGE_TEXT_TURN_TIMEOUT = (
    "I had trouble finalizing this response in the time available. "
    "Please try asking again, or rephrase your question."
)
SALVAGE_TEXT_BRANCH_TIMEOUT = (
    "I had trouble finalizing this part of the analysis in the time available."
)


def record_salvage_signal(
    *,
    cause: SalvageCause,
    error: BaseException,
    extra_log_context: dict[str, Any] | None = None,
) -> None:
    """Shared salvage-signal writer — the **module-level seam** invoked by
    every salvage entry point.

    By design: "Burying the builder inside one callback body
    (forcing the other two entry points to re-derive it) re-creates
    exactly the drift this seam exists to prevent." Lives at module
    level (not on :class:`SalvagePlugin`) because the turn watchdog
    invokes it from the specialist runner — outside any plugin
    callback — and a free function is reachable from both surfaces with
    no plugin-instance plumbing.

    Channel A: mutate the current branch's :class:`TurnErrorSink` in
    place when one is bound (orchestrated fan-out). No-op when no sink
    is bound — single-pass / deterministic-plan paths use Channel B
    exclusively.

    Channel B: set the per-run :class:`SalvagePayload` for the
    application layer to read after ``run_async`` returns.

    Both writes are unconditional within this seam. Application layers
    read whichever channel they own; sink + payload may both be set on
    the same run (Channel A for fan-out reduction, Channel B for
    SpecialistResponse construction within the branch). That dual-write
    is intentional — neither channel is a "primary" — and matches the
    two-channel union seam at
    :meth:`AgentSession.get_last_agent_error`.
    """
    # ``tabi.salvage`` span: record the salvage signal on the active
    # trace so the viewer's Salvage panel has a span source. Emitted once here
    # in the shared seam so every entry point (watchdog / plugin callbacks) is
    # covered. Point-in-time; nests under ``tabi.turn``. No-op without a provider.
    with salvage_span(cause=cause) as _sspan:
        set_span_attrs(_sspan, {"tabi.salvage.error_type": type(error).__name__})

    # Channel A: mutate in place if a sink is bound. By design we forbid
    # `ContextVar.set()` here — that would be invisible to the parent
    # task. In-place mutation IS visible (same object).
    sink = get_current_turn_error_sink()
    if sink is not None:
        sink.agent_error = True

    # Channel B: signal the application layer that salvage is needed and
    # carry minimal metadata about the cause.
    payload = SalvagePayload(
        cause=cause,
        error_type=type(error).__name__,
        error_message=str(error)[:500],
    )
    set_pending_salvage_payload(payload)

    # Structured log line — pairs with the existing
    # ``specialist.salvage_diagnostic`` warning emitted by
    # specialist_runner's inline detection. Different layer, same turn —
    # both surface in `tabi-logs` for forensic review.
    #
    # Sink: structlog → stdout → Cloud Logging. NOT the BigQuery Agent
    # Analytics plugin (that one captures ADK ``Event`` objects via its
    # ``content_formatter`` — see ``session_plugins.py``
    # ``_metadata_only_content_formatter``; ContextVar-borne payloads do
    # not flow through it). The ``error_message`` is truncated at 500
    # chars in the ``SalvagePayload`` constructor above; only
    # ``error_type`` (the class name, never the message) is logged here,
    # so even on a noisy error path Cloud Logging row sizes stay
    # bounded.
    log_event = "salvage_plugin.signal_recorded"
    log_payload: dict[str, Any] = {
        "cause": cause,
        "error_type": type(error).__name__,
        "channel_a_written": sink is not None,
        "channel_b_written": True,
    }
    if extra_log_context:
        log_payload.update(extra_log_context)
    logger.warning(log_event, **log_payload)


class SalvagePlugin(BasePlugin):
    """Observes ADK model/tool errors and writes Channel A (sink) +
    Channel B (payload) signals so the application layer can build
    ``SpecialistResponse(agent_error=True)`` by design.

    Both ADK error callbacks delegate to the
    module-level :func:`record_salvage_signal` shared seam. The
    runner-owned turn watchdog and orchestrator-owned per-branch
    watchdog invoke the same module-level seam directly — they live
    outside any plugin callback, so a module-level function reaches
    both surfaces with no plugin-instance plumbing. By design:
    "Burying the builder inside one callback body (forcing the other
    two entry points to re-derive it) re-creates exactly the drift
    this seam exists to prevent."
    """

    name = "tabi_salvage_plugin"

    def __init__(self) -> None:
        super().__init__(name=self.name)

    # ------------------------------------------------------------------
    # ADK callbacks — thin wrappers delegating to the module-level seam
    # ------------------------------------------------------------------

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        """Cap-hit / model error path. Writes both channels via the
        shared seam, then returns ``None`` to let ADK propagate the
        exception (the specialist runner's inline ``except`` block still
        runs in the propagation path — this plugin is observation, not
        rescue).
        """
        record_salvage_signal(
            cause=CAUSE_MODEL_ERROR,
            error=error,
            extra_log_context={
                "agent_name": getattr(callback_context, "agent_name", None),
                "model": getattr(llm_request, "model", None),
            },
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Tool error path. Same shared-seam write. ADK propagates the
        tool error to the model (which may retry the tool or surface a
        text error) when this returns ``None``; the plugin's role is
        observation + signal write so the application layer knows
        salvage was triggered by an error, not just an empty model
        response.

        **Verify-on-wheel outcome (a shipped guardrail fix).**
        It originally seemed the guardrail's ``TenantScopeViolation``
        raise from ``before_tool_callback`` would land here. Verified
        against the pinned ADK 1.33 wheel — it does NOT:

        - ADK invokes ``run_before_tool_callback`` without a
          ``try/except``; only the actual tool call is wrapped, and only
          that wrapper dispatches to ``_run_on_tool_error_callbacks``.
        - ADK's plugin manager wraps any callback
          raise as ``RuntimeError(...) from original_exc``; the
          original class is lost at the call site (``__cause__`` is
          the only carrier).

        The guardrail therefore uses the explicitly-sanctioned fallback:
        the :class:`~tabi_analytics.core.guardrail_plugin.GuardrailPlugin`
        writes the channels via :func:`record_salvage_signal` directly
        from ``before_tool_callback`` and returns a structured
        tool-result sentinel. The application-layer salvage builder
        (the specialist runner's collection path and the orchestrator's
        per-branch path) force-fires salvage when
        ``SalvagePayload.cause == "tenant_scope"`` — that is the
        convergent seam, **not** this ``on_tool_error_callback``
        (sentinel returns from ``before_tool_callback`` become the
        ``function_response`` and do not route through
        ``_run_on_tool_error_callbacks``).

        The ADK 1.33 propagation behaviour is pinned by
        a before-tool-callback propagation test;
        an upstream bump that changes it surfaces at CI time and lets
        the guardrail flip back to the raise path with zero hierarchy change
        (:class:`TenantScopeViolation` is already a subclass of
        ``LlmCallsLimitExceededError``).
        """
        record_salvage_signal(
            cause=CAUSE_TOOL_ERROR,
            error=error,
            extra_log_context={
                "tool_name": getattr(tool, "name", None),
            },
        )
        return None
