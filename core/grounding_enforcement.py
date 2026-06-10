"""Structural grounding-enforcement gate.

This module is the "agent that won't make up numbers" seam. In the full TABI
platform the logic below lives inline in the proprietary specialist runner
(``core.specialist_runner``); for this public showcase it has been lifted into
a clean, self-contained module so the grounding behaviour can be read and
tested in isolation.

What it does
------------
After a specialist produces an answer, :func:`enforce_grounding_redaction`
runs the already-live :class:`~core.response_validator.ResponseValidator` over
the assembled prose. Every numeric claim in the answer is classified against
the actual query results:

* ``exact`` / ``approximate`` / ``derived`` — the figure is supported by the
  data and is **left untouched**.
* ``fabricated`` (no data at all) / ``no_match`` (a number with no nearby
  supporting data value) — the figure is **redacted** in place, replaced with
  the ``[unverified]`` marker.

When at least one figure is redacted, a user-facing caveat blockquote is
prepended to both answer surfaces (``result.text`` and
``result.response.answer_markdown``) and a ``grounding_enforcement.redacted``
structlog event is emitted with the redacted count.

Deliberate non-behaviours (carried over verbatim from the source):

* It does **not** set ``agent_error`` and does **not** trigger a retry — the
  answer is repaired in place so downstream scoring stays valid and no extra
  LLM pass is spent.
* It is a no-op for non-enforced agents, when the validator is absent, when
  there is no usable text, or when nothing is flagged.

Public surface
--------------
The original lived in ``core.specialist_runner`` as the module-level function
``_enforce_grounding_redaction`` plus the constants ``GROUNDING_ENFORCED_AGENTS``
and ``GROUNDING_ENFORCEMENT_CAVEAT``. To keep the existing test suite
(`tests/test_grounding_enforcement.py`) importing the exact same names, this
module re-exports all three under their original identifiers. The instance
coupling (the old function took a ``RunnerHost``) is reduced to a structural
contract: ``host`` only needs a ``.validator`` attribute, and ``result`` only
needs ``.text``, ``.tool_outputs`` and ``.response`` — so any duck-typed object
(including the tests' ``SimpleNamespace``/``SpecialistRunResult``) works.

``enforce_grounding_redaction`` is provided as a readable public alias of the
underscore-prefixed name.
"""

from __future__ import annotations

from typing import Any, Protocol

import structlog

from core.response_validator import (
    GROUNDING_REDACTION_MARKER,
    build_query_results_from_tool_outputs,
    redact_unsourced_figures,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "GROUNDING_ENFORCED_AGENTS",
    "GROUNDING_ENFORCEMENT_CAVEAT",
    "GROUNDING_REDACTION_MARKER",
    "enforce_grounding_redaction",
    "_enforce_grounding_redaction",
]


# Specialists whose prose figures are structurally grounding-checked. Scoped
# to ``pipeline_analyst`` first — the goal-attainment verifier already covers
# the planning specialists, and the other analytic specialists are a
# deliberate follow-up rather than part of the original change.
GROUNDING_ENFORCED_AGENTS: frozenset[str] = frozenset({"pipeline_analyst"})


# User-facing caveat prepended when a grounding-enforced answer had one or
# more unsourced figures redacted. Distinct cause from the planning
# cross-check caveat, hence its own constant/text.
GROUNDING_ENFORCEMENT_CAVEAT: str = (
    "> **Note:** One or more figures in this response could not be verified "
    "against the queried data and were removed. Ask a follow-up if you'd "
    "like a grounded breakdown.\n\n"
)


class _Host(Protocol):
    """Structural contract for the grounding host.

    The full runner passes a ``RunnerHost``; only its ``validator`` is read
    here, so any object exposing ``.validator`` satisfies the gate.
    """

    validator: Any


def enforce_grounding_redaction(
    result: Any,
    host: _Host,
    agent_name: str,
) -> None:
    """Redact validator-flagged unsourced figures from a specialist answer.

    Runs the live :class:`ResponseValidator` over the assembled answer,
    neutralises any ``fabricated`` / ``no_match`` figure in BOTH
    ``result.text`` and ``result.response.answer_markdown``, and prepends a
    caveat when anything was redacted.

    Deliberately does **not** set ``agent_error`` and does **not** trigger a
    retry: the answer is repaired in place so the case stays scored and the
    first cut avoids an extra LLM pass.

    No-op for non-enforced agents, when the validator is absent, when there
    is no usable text, or when nothing is flagged.

    Parameters
    ----------
    result:
        The specialist run result. Must expose ``text`` (the canonical answer
        surface), ``tool_outputs`` (the raw tool payloads used to rebuild the
        query results), and ``response`` (the structured response carrying
        ``answer_markdown``; may be ``None``).
    host:
        Any object exposing a ``validator`` attribute (a ``ResponseValidator``
        or ``None``).
    agent_name:
        The specialist that produced ``result``; gates enforcement against
        :data:`GROUNDING_ENFORCED_AGENTS`.
    """
    if agent_name not in GROUNDING_ENFORCED_AGENTS:
        return
    validator = host.validator
    if validator is None:
        return
    final_text = result.text
    if not final_text or not final_text.strip():
        return

    query_results = build_query_results_from_tool_outputs(result.tool_outputs)
    report = validator.validate(final_text, query_results)

    redacted_text, match_types = redact_unsourced_figures(final_text, report)
    if not match_types:
        return

    result.text = GROUNDING_ENFORCEMENT_CAVEAT + redacted_text
    if result.response is not None:
        answer_md = result.response.answer_markdown
        if answer_md == final_text:
            redacted_md = redacted_text
        else:
            # The legacy path strips orphaned chart refs from ``text``
            # *after* ``answer_markdown`` is set, so the two surfaces diverge
            # by those markers and ``report``'s positions (computed against
            # ``final_text``) no longer line up. Re-validate the markdown so
            # its redaction is position-correct rather than guessed across a
            # foreign offset map.
            md_report = validator.validate(answer_md, query_results)
            redacted_md, _ = redact_unsourced_figures(answer_md, md_report)
        result.response.answer_markdown = GROUNDING_ENFORCEMENT_CAVEAT + redacted_md

    logger.info(
        "grounding_enforcement.redacted",
        agent=agent_name,
        redacted_count=len(match_types),
        match_types=sorted(set(match_types)),
        claims_found=len(report.claimed_values),
        data_values=len(report.data_values),
    )


# Back-compat alias: the source module (and the test suite) refer to this gate
# by its original underscore-prefixed name. Keep both pointing at one body.
_enforce_grounding_redaction = enforce_grounding_redaction
