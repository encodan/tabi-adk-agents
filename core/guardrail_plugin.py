"""Guardrail plugin (tenant-scope + prompt-injection).

Implements a global ``BasePlugin``. Two halves:

1. **Tenant-scope assertion (fail-closed)** in ``before_tool_callback``.
   The session's ``customer_id`` is bound at ``configure_tools`` into a
   ContextVar-backed :class:`ToolContext`. On any tool
   invocation the guardrail asserts (a) a session ``customer_id`` is
   bound (no tool execution outside a configured session) and (b) if the
   model ever passes a ``customer_id`` field in ``tool_args`` it equals
   the session's. Either failure mode triggers the
   :func:`record_salvage_signal` write + sentinel return described below.
   The check is prophylactic: today's tools read ``customer_id`` from the
   ContextVar (never from model-supplied args), so a mismatch would only
   arise if a future tool schema exposed the field. Defense-in-depth
   behind the API auth middleware; cross-tenant reads are the worst
   failure mode for B2B ATS data so the cost of a redundant check is
   negligible.

2. **Prompt-injection heuristics (advisory)** in
   ``on_user_message_callback``. A small frozen tuple of compiled
   regexes covers the recognised "ignore previous instructions" family +
   embedded role-marker tokens (``user:``, ``system:``, ``assistant:``
   inserted by the user). Flagged turns log
   ``guardrail.injection_pattern_matched`` and return ``None`` — they
   do **not** raise or substitute the user message. The bias is
   deliberately low-false-positive: heads-of-talent ask unusual
   analytics questions and over-blocking legitimate queries is a worse
   product outcome than a missed heuristic. The injection layer is
   defense-in-depth, not the only layer; ``safety_v1`` scores these as
   normal turns.

## The sentinel-fallback path (load-bearing, verified-on-wheel)

The design originally preferred raising
``TenantScopeViolation(LlmCallsLimitExceededError)`` from
``before_tool_callback`` so the salvage hook's existing ``except
LlmCallsLimitExceededError`` clause caught it byte-identically by Python
subclass semantics. **Verification against the pinned ADK 1.33 wheel**
proved this path is not available: ADK does not wrap the
``before_tool_callback`` invocation in the tool-error try/except (only the
actual tool call is wrapped), and its plugin manager re-wraps any
in-callback raise as a generic ``RuntimeError`` (the original class
survives only on ``__cause__``).

So a raise from ``before_tool_callback`` would bypass both the salvage
hook's subclass-catch *and* the salvage plugin's
``on_tool_error_callback`` — silently degrading to a generic
``RuntimeError``-suppressed branch. The behaviour is pinned by a
propagation test so an upstream bump that changes it surfaces at CI time
(the pin-and-verify-on-wheel discipline).

The design sanctioned a fallback: set the channels directly inside
``before_tool_callback`` and return a structured tool-result that signals
"blocked." This module implements that fallback. The convergence point is
not ``on_tool_error_callback`` (that callback is unreachable from a
sentinel return) but rather the application-layer salvage block in the
specialist runner / orchestrator, which force-fires salvage when
``SalvagePayload.cause == "tenant_scope"`` regardless of model text. This
force-salvage extension is what preserves the "no answer synthesized
over a hole" guarantee that the raise path would have given for free.

## Channel writes are byte-identical to a cap-hit (acceptance binding)

On a violation the guardrail invokes :func:`record_salvage_signal` with
``cause="tenant_scope"``. That is the same shared seam used by the
salvage plugin's ``on_model_error_callback``,
``on_tool_error_callback``, the runner-owned turn watchdog, and the
orchestrator's per-branch watchdog. Channel A (the per-branch
:class:`TurnErrorSink`, when bound under orchestrated fan-out) and
Channel B (the :class:`SalvagePayload` ContextVar) are written
identically; the application-layer salvage builder then constructs the
``SpecialistResponse(agent_error=True)`` from those channels. The
resulting object is byte-identical to a cap-hit salvage by construction
— same builder, same fields, same example-level ``error``
bucketing in the eval. The distinguishing telemetry lives in (a) the
``guardrail.tenant_scope_violation`` structured log and (b) the
``SalvagePayload.error_type == "TenantScopeViolation"`` field, neither
of which collapses the authorization-breach vs. cap-exhaustion
distinction.

The :class:`TenantScopeViolation` exception **is still** a subclass of
``LlmCallsLimitExceededError`` even though we never raise it through
ADK. The subclass relationship preserves the acceptance
assertion (``isinstance(exc, TenantScopeViolation) and isinstance(exc,
LlmCallsLimitExceededError)``) on the instance we hand to
``record_salvage_signal`` and to the structured log; it also keeps the
class hierarchy correct if a future ADK rev makes the raise path
viable (then this module can flip back to raising with zero downstream
change).

## Plugin ordering (load-bearing, see ``session_plugins.py``)

``GuardrailPlugin`` is **index 0** in ``build_session_plugins()``. Two
reasons:

1. ``PluginManager._run_callbacks`` short-circuits on the first
   non-``None`` return. The guardrail's sentinel return for a
   tenant-scope violation is non-``None``, so it must run before the
   ``ReflectAndRetryToolPlugin`` / ``SalvagePlugin`` chain on
   ``before_tool_callback``; otherwise a sibling plugin could substitute
   a tool result before the guardrail has had a chance to inspect args.
2. The order contract in the module docstring of
   ``session_plugins.py`` reserved index 0 for this plugin from the
   start — fixing the order so later layers slot in without reshuffling.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from config import get_config
from core.salvage_payload import SalvageCause
from core.salvage_plugin import record_salvage_signal
from tools.tool_context import get_tool_context

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = structlog.get_logger(__name__)


# Mirrors the ``CAUSE_*`` constants in ``salvage_plugin`` so guardrail call
# sites get mypy narrowing on the Literal value rather than re-typing the
# string. Adding a fifth cause is a single edit at ``SalvageCause``.
CAUSE_TENANT_SCOPE: SalvageCause = "tenant_scope"


# Sentinel key embedded in the tool-result dict returned to the model on a
# tenant-scope block. Lets tests and downstream log readers identify the
# block without string-matching the model-paraphrased error message.
TENANT_SCOPE_BLOCK_KEY = "__tabi_tenant_scope_blocked__"


class TenantScopeViolation(LlmCallsLimitExceededError):  # noqa: N818
    """Authorization breach: a tool call attempted to read outside the session's
    ``customer_id``. Subclass of ``LlmCallsLimitExceededError``
    — never raised through ADK (see module docstring's sentinel-fallback
    section); instantiated only so ``record_salvage_signal`` records the
    distinguishing class name and the ``isinstance`` assertion holds.

    The N818 "Error suffix" lint is suppressed because the class name is the
    acceptance contract; renaming would break the assertion.
    """


# Prompt-injection heuristics. Compiled once at module load — extending
# the set is a single conscious edit (same shape discipline as
# ``RETIRED_MODELS`` in ``config``). Patterns are deliberately conservative;
# the design mandates a low-false-positive bias over coverage because
# over-blocking legitimate analytics questions is a worse product outcome
# than a missed heuristic. Defense-in-depth, not the only layer.
#
# Each pattern is a tuple ``(name, compiled_regex)`` so the matched
# heuristic's name is recorded in the structured log without exposing the
# pattern source.
#
# [public-showcase note] The patterns below are deliberately SIMPLIFIED,
# representative versions that demonstrate the heuristic layer's shape
# (named patterns → structured-log telemetry, advisory not blocking,
# plugin-index-0 ordering). They are NOT TABI's production detection rules:
# publishing the exact regexes would hand an attacker the precise monitoring
# surface to phrase around. The live ruleset — boundary conditions, the
# token vocabulary, and the canary/role-marker variants — is maintained
# privately and out of band. Because this layer is advisory (it logs, it
# never blocks or rewrites the turn — the real containment is the pinned
# safety config + the tenant-scope guardrail), the simplification changes
# the demo's telemetry coverage, not its security properties.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        # Representative: the "ignore/disregard ... previous instructions"
        # family. The production rule covers more verbs and targets.
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior)\s+"
            r"(?:instructions?|prompts?|rules?)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        # Representative: "you are now a different assistant"-style attempts
        # to reassign the agent's role.
        re.compile(
            r"\byou\s+are\s+now\s+(?:a\s+|an\s+)?(?:different|new|unrestricted)\s+"
            r"(?:ai|assistant|model|agent)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_marker_injection",
        # Representative: a fake chat role marker on its own line
        # ("\nsystem:") — template impersonation.
        re.compile(
            r"(?:\n|^)\s*(?:system|assistant)\s*:",
            re.IGNORECASE,
        ),
    ),
)


class GuardrailPlugin(BasePlugin):
    """Global guardrail plugin.

    See the module docstring for the design rationale (sentinel-fallback
    path on ADK 1.33, plugin ordering, byte-identical salvage). This
    class is intentionally thin — it implements two ADK callbacks and
    delegates all channel writes to the shared salvage seam
    :func:`record_salvage_signal`.
    """

    name = "tabi_guardrail_plugin"

    def __init__(self) -> None:
        super().__init__(name=self.name)

    # ------------------------------------------------------------------
    # Tenant-scope assertion — `before_tool_callback`
    # ------------------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Tenant-scope fail-closed gate. See module docstring for why this
        is the sentinel-fallback path and not the originally-intended raise path.

        Returns ``None`` to let the tool execute normally on the clean
        path; returns a sentinel tool-result dict (non-``None``, so ADK
        substitutes it for the tool call) on a block.

        The feature flag exists only so the bucketing-independent
        isolation comparison can flip it off ("guardrail-off ⇒ cross-tenant
        rows returned and answered over"). Always-on by design.
        """
        if not get_config().feature_flags.tenant_scope_guardrail_enabled:
            return None

        ctx = get_tool_context()
        session_customer_id: str | None = None
        if ctx is not None:
            session_customer_id = ctx.tool_config.get("customer_id")

        tool_arg_customer_id = tool_args.get("customer_id")

        # Two failure modes share one outcome: no session bound (defensive —
        # ``configure_tools`` is the only legitimate caller), or model-supplied
        # ``customer_id`` differs from session. Today's tools read customer_id
        # from the ContextVar, never from model args; the second check is
        # prophylactic against a future tool schema that exposes the field.
        violation_reason: str | None = None
        if not session_customer_id:
            violation_reason = "no_session_customer_id"
        elif tool_arg_customer_id is not None and tool_arg_customer_id != session_customer_id:
            violation_reason = "customer_id_mismatch"

        if violation_reason is None:
            return None

        tool_name = getattr(tool, "name", None) or type(tool).__name__

        # ``tool_arg_customer_id`` value deliberately omitted from the log —
        # logging a cross-tenant identifier is the exact failure mode this
        # guardrail prevents. Presence flag is enough to triage.
        logger.warning(
            "guardrail.tenant_scope_violation",
            tool_name=tool_name,
            violation_reason=violation_reason,
            session_customer_id_bound=session_customer_id is not None,
            tool_arg_customer_id_present=tool_arg_customer_id is not None,
        )

        violation = TenantScopeViolation(
            f"tenant_scope violation on tool '{tool_name}': {violation_reason}"
        )
        record_salvage_signal(
            cause=CAUSE_TENANT_SCOPE,
            error=violation,
            extra_log_context={
                "tool_name": tool_name,
                "violation_reason": violation_reason,
                "scope": "tenant_scope_guardrail",
            },
        )

        return {
            TENANT_SCOPE_BLOCK_KEY: True,
            "error": (
                "Tool execution blocked: tenant-scope guardrail. The request "
                "was refused because the tool call did not match the session's "
                "tenant boundary. This is a server-side authorization check "
                "and cannot be retried with different arguments."
            ),
            "violation_reason": violation_reason,
        }

    # ------------------------------------------------------------------
    # Prompt-injection heuristics — `on_user_message_callback`
    # ------------------------------------------------------------------

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Scan inbound user messages for known prompt-injection patterns.
        Advisory only — logs and annotates; never raises, never substitutes
        the message. The design mandates low-false-positive bias because
        over-blocking legitimate analytics questions is a worse product
        outcome than a missed heuristic.

        Returns ``None`` unconditionally so ADK continues with the original
        message.
        """
        text_segments: list[str] = []
        if user_message and user_message.parts:
            for part in user_message.parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    text_segments.append(part_text)

        if not text_segments:
            return None

        # Join with ``\n`` so the matcher sees what the model sees when ADK
        # concatenates parts. A role-marker that lands at a part boundary
        # IS a real injection attempt (the model would interpret the join
        # the same way) — not a synthetic false positive of the join.
        combined_text = "\n".join(text_segments)
        matched: list[str] = []
        for pattern_name, pattern in INJECTION_PATTERNS:
            if pattern.search(combined_text):
                matched.append(pattern_name)

        if matched:
            # Single structured log line; the matched names are bounded and
            # safe to include. The user message itself is NOT logged — it
            # may contain tenant-identifying analytics questions and we
            # already capture user messages at the chat-service layer with
            # the right contextvars.
            logger.warning(
                "guardrail.injection_pattern_matched",
                matched_patterns=matched,
                pattern_count=len(matched),
            )

        return None
