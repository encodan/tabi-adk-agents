"""Salvage payload — carried from the SalvagePlugin
to the application-layer salvage builder.

Companion data structure to the :class:`TurnErrorSink`
(in ``turn_error_sink.py``). They serve disjoint channels by design:

- ``TurnErrorSink`` (Channel A): boolean flag, populated by the salvage
  plugin under orchestrated fan-out. Read by the orchestrator's
  post-gather reduction. Per-branch; one sink per gathered branch.

- ``SalvagePayload`` (this module — Channel B signal carrier): structured
  metadata describing *why* salvage fired. Populated by the salvage
  plugin in both error callbacks. Read by the specialist runner's
  collection path as an explicit trigger
  for salvage (in addition to the existing "empty final_text + no
  handoff" condition) so application-layer code knows the plugin
  intercepted the error, not just that the run came back empty.

The plugin populates **both** in every error callback — the sink lets
orchestrated fan-out's existing post-gather reduction fire correctly,
the payload lets single-pass / deterministic-plan paths build the
Channel-B ``SpecialistResponse(agent_error=True)`` even when the
inline empty-text detection wouldn't have fired (or would have fired
without knowing salvage was *triggered by* an error, vs the model just
returning empty).

Construction of the user-visible salvage text (chart-refs fallback,
generic "I had trouble finalizing..." message, etc.) stays in
the specialist runner and the orchestrator, by the
"relocation, not redesign" principle — those paths have access to per-attempt
``tool_outputs`` which the ADK-layer plugin does not.

A third writer, the runner-owned turn watchdog, populates this same
payload when its ``CancelledError`` fires, so all three salvage entry
points (model error, tool error, watchdog) converge on a single payload
reader in the specialist runner.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

# Four salvage entry points: model error, tool error, the runner-owned
# turn watchdog, and the tenant-scope guardrail (added when the
# sentinel-fallback path was confirmed against the pinned wheel by
# design). Spelled as a Literal so a typo
# at construction surfaces in mypy, not at runtime when the reader
# matches on it. Readers that branch on ``cause`` (in particular the
# salvage-trigger condition in the specialist runner's collection path
# and the orchestrator's per-branch path) get exhaustiveness narrowing
# at every site as a side-effect.
SalvageCause = Literal["model_error", "tool_error", "watchdog", "tenant_scope"]


@dataclass(frozen=True)
class SalvagePayload:
    """Structured signal carried from the salvage plugin's error callbacks
    to the application-layer salvage builder.

    Frozen because the payload is a *signal*, not a per-branch mutable
    state holder (the sink is the mutable one — see ``TurnErrorSink``).
    Once the plugin writes it the contents do not change; later readers
    treat it as a snapshot of "what fired and why".

    The payload intentionally omits the salvage text itself: text
    construction is the application layer's job (it has ``tool_outputs``
    for chart-ref fallback, the prompt cache miss reason, etc.). The
    payload's contract is "salvage is needed and here is why" — the
    reader decides what to display.

    Attributes:
        cause: Which entry point fired. The plugin's error callbacks
            write ``model_error`` and ``tool_error``; the runner-owned
            turn watchdog writes ``watchdog``.
        error_type: ``type(error).__name__`` from the ADK callback.
            Captured so traces / BQ can distinguish a cap-hit
            (``LlmCallsLimitExceededError``) from a generic model error
            even after the plugin rescues.
        error_message: ``str(error)[:500]`` from the ADK callback —
            truncated to keep BQ row sizes bounded.
    """

    cause: SalvageCause
    error_type: str
    error_message: str

    def requires_force_salvage(self) -> bool:
        """True when the salvage trigger must fire regardless of model text.

        By the "no answer synthesized over a hole" invariant: when the guardrail
        blocks a tool via the sentinel-fallback path, the model must not
        narrate around the block and serve that narration. Returns ``True``
        only for ``cause == "tenant_scope"`` — the other causes (cap-hit,
        tool error, watchdog) preserve any partial model text per the
        existing salvage trigger semantics. Called from both salvage sites
        (the specialist runner's collection path and the orchestrator's
        per-branch path) so the predicate has one
        owner; pinned by a force-salvage-on-tenant-scope test.
        """
        return self.cause == "tenant_scope"


# Per-run salvage payload. The ContextVar carries the payload reference
# set by the plugin's error callbacks. Read by
# the specialist runner's collection path after ``run_async``
# returns; if non-None, application-layer salvage builds the Channel-B
# ``SpecialistResponse(agent_error=True)`` and reads the payload's
# metadata for diagnostic logging.
#
# Parent→child propagation at task creation IS what we want here (same
# as ``_current_turn_error_sink``): the plugin runs in a child task
# under ADK's run loop; the payload it sets needs to be visible to the
# parent task reading after the loop. Because the plugin sets a new
# ContextVar value (not in-place mutation of an existing object), this
# requires a different mechanism than the sink — explicit binding +
# write helpers, not mutate-in-place.
#
# Reset semantics: the reader (the specialist runner) explicitly resets the
# payload after consuming it, so a second salvage in the same task
# context does not see stale state. The :func:`bind_salvage_payload_scope`
# context manager enforces this for the per-attempt loop.
_pending_salvage_payload: ContextVar[SalvagePayload | None] = ContextVar(
    "pending_salvage_payload",
    default=None,
)


def set_pending_salvage_payload(payload: SalvagePayload) -> None:
    """Set the per-run salvage payload (called by the salvage plugin's
    error callbacks).

    Idempotent on the *first* salvage of a run — a second concurrent
    error in the same task context would overwrite the first. ADK
    surfaces errors sequentially within a single ``run_async``, so this
    is fine in practice; if a future ADK rev surfaces concurrent errors
    the reader will see whichever payload fired last (typically the
    final error before the rescue path engages).

    **Cross-task invariant (load-bearing — see the documented
    cross-task write-invisibility anti-pattern).**
    This function MUST be called in the same task context as the eventual
    :func:`get_pending_salvage_payload` reader. Channel B is the *one
    deliberate exception* in this design to the rule that Channel A
    (``TurnErrorSink``) follows: Channel A uses in-place mutation
    because the sink is set up by a parent task and mutated by a child
    plugin callback; ``ContextVar.set()`` from the child wouldn't reach
    the parent. Channel B's payload is also set inside a plugin
    callback — but on ADK 1.33 the plugin is invoked via direct
    ``await self._plugin.on_model_error_callback(...)`` (no
    ``asyncio.create_task`` wrap), so the ``set`` happens in the same
    task as ``runner.run_async``'s caller, making it visible to the
    application-layer reader after ``run_async`` returns.

    **Verify on any ADK pin widen** (per the pin policy and the
    verify-on-wheel pattern). If a future ADK
    rev wraps plugin invocation in ``create_task``, Channel B's reader
    will see ``None`` because the parent's context wasn't mutated. The
    failure mode would be silent: salvage diagnostics lose the plugin's
    structured signal but the existing empty-text trigger still fires.
    An end-to-end test exercising the full
    ``Runner.run_async`` → plugin → reader path is the planned backstop;
    until then, the invariant is review-enforced.
    """
    _pending_salvage_payload.set(payload)


def get_pending_salvage_payload() -> SalvagePayload | None:
    """Read the per-run salvage payload bound by the plugin, if any.

    Returns ``None`` when no error callback has fired in the current
    task context — the normal case for clean runs.
    """
    return _pending_salvage_payload.get()


def clear_pending_salvage_payload() -> None:
    """Reset the per-run salvage payload to ``None``.

    Called by the application layer after consuming the payload so a
    second attempt / second run in the same task context doesn't see
    stale signal. Paired with :func:`bind_salvage_payload_scope` for the
    per-attempt loop in the specialist runner.
    """
    _pending_salvage_payload.set(None)


@contextmanager
def bind_salvage_payload_scope() -> Iterator[None]:
    """Scope the ContextVar token to a single per-attempt iteration of
    the specialist runner's retry loop.

    The plugin's writes are scoped to the per-attempt ``Runner`` /
    ``run_async`` invocation; the application layer reads the payload
    after that invocation returns. This context manager binds a fresh
    ``None`` token at entry and resets on exit, so retries do not see
    payloads from prior attempts (and so the orchestrator's per-branch
    coroutine does not see payloads from siblings).

    Use exactly once per ``run_async`` invocation that wants to surface
    salvage payloads.
    """
    token = _pending_salvage_payload.set(None)
    try:
        yield None
    finally:
        _pending_salvage_payload.reset(token)
