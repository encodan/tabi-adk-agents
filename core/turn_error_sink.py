"""Per-branch ``TurnErrorSink`` mechanism.

A small mutable holder per gathered branch, addressed via a single
``ContextVar`` that carries only the *reference* (not the value). The
reference is set once at branch-scope entry by the orchestrator's
per-agent coroutine —
parent→child propagation at task creation is exactly what we want;
child→parent ``ContextVar.set()`` rebinding is forbidden because it is
invisible across the ``asyncio.gather`` task boundary
(the cross-task write-invisibility invariant).

The salvage plugin reads the sink and mutates
``agent_error`` *in place*. In-place mutation IS visible across the ADK
task boundary because both sides reference the same object — only
``ContextVar.set()`` rebinding is not. The branch coroutine that owns
the sink reads it after the run completes and propagates the value onto
its ``SpecialistResult``.

**Channel-A-only mechanism.** This sink exists *solely* because the
orchestrated-streaming path has no per-run response object to carry the
flag — it must reduce a boolean (each channel reduces a single value).
The single-pass and
deterministic-query-plan paths use Channel B (per-run
``SpecialistResponse.agent_error`` directly) and require **no**
``TurnErrorSink`` and **no** ContextVar. Structured paths must not
route through the sink (the anti-pattern of a second writer of a value
that is already correctly per-run).

This module owns the sink + binding mechanism; the salvage plugin
wires the reads, and the orchestrator wires the post-run propagation
onto ``SpecialistResult``. The mechanism-level acceptance:
each gathered branch holds a distinct sink; a single shared sink
constructed before ``gather`` fails the isolation property (negative
control).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class TurnErrorSink:
    """Mutable per-branch holder for the salvage plugin's ``agent_error`` signal.

    The salvage plugin reads
    :func:`get_current_turn_error_sink`, asserts a non-``None`` value, and
    mutates ``sink.agent_error = True`` in place. The orchestrator's
    per-agent coroutine that owns the sink reads
    it after the run completes and propagates the value onto its
    ``SpecialistResult``.

    DO NOT rebind via ``ContextVar.set()`` inside child tasks — only mutate
    the field in place. See the cross-task write-invisibility invariant.
    """

    agent_error: bool = False


# Single ContextVar carrying the per-branch sink reference. Set once at
# branch-scope entry by the orchestrator's per-agent coroutine; read by
# the salvage plugin's error callbacks. Never set inside a
# plugin callback (the child→parent-invisibility anti-pattern).
_current_turn_error_sink: ContextVar[TurnErrorSink | None] = ContextVar(
    "current_turn_error_sink",
    default=None,
)


def get_current_turn_error_sink() -> TurnErrorSink | None:
    """Read the per-branch error sink bound at the current ContextVar scope.

    Returns ``None`` when called from a code path that does not run inside
    a per-branch coroutine (single-pass, deterministic-plan, or test code
    that does not bind a sink — those use Channel B directly).
    """
    return _current_turn_error_sink.get()


@contextmanager
def bind_turn_error_sink(sink: TurnErrorSink) -> Iterator[TurnErrorSink]:
    """Bind ``sink`` to the per-branch ContextVar for the duration of the
    ``with`` block, resetting the token on exit.

    Call site is the orchestrator's per-agent coroutine. Each invocation of
    that coroutine constructs a fresh sink and enters this manager — the
    structural analogue of each branch already owning its own
    ``SpecialistResult``.

    Yielded value is the same ``sink`` passed in — convenience for
    callers that want to bind and reference in one expression.
    """
    token = _current_turn_error_sink.set(sink)
    try:
        yield sink
    finally:
        _current_turn_error_sink.reset(token)
