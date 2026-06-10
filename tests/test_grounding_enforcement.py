"""Structural grounding enforcement (the grounding design).

Fast-tier unit coverage for the pipeline_analyst redaction seam:

1. ``tool_outputs → query_results`` shaping (the dominant false-positive risk).
2. Redaction neutralises a flagged figure + caveat; grounded figures survive.
3. Scope gate — non-enforced agents are untouched, validator not invoked.
4. The redaction path never sets ``agent_error`` (the scoring invariant).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from core.grounding_enforcement import (
    GROUNDING_ENFORCEMENT_CAVEAT,
    _enforce_grounding_redaction,
)
from core.handoff import SpecialistRunResult
from core.response_validator import (
    GROUNDING_REDACTION_MARKER,
    ResponseValidator,
    build_query_results_from_tool_outputs,
)
from core.specialist_schema import SpecialistResponse


def _response(answer_markdown: str) -> SpecialistResponse:
    return SpecialistResponse(
        agent_name="pipeline_analyst",
        answer_markdown=answer_markdown,
        claims=[],
        charts=[],
        confidence=0.8,
        data_sufficient=True,
    )


def _host(validator: object | None) -> SimpleNamespace:
    return SimpleNamespace(validator=validator)


# --- tool_outputs → query_results shaping ---------------------------------


def test_shaping_multi_query_flattens_to_indexable_rows() -> None:
    tool_outputs = [
        {
            "name": "query_multiple_recruitment_metrics",
            "response": {
                "success": True,
                "results": [
                    {
                        "query_index": 0,
                        "success": True,
                        "columns": ["dept", "avg_days_in_stage"],
                        "data": [{"dept": "Engineering", "avg_days_in_stage": 7.35}],
                    }
                ],
            },
        }
    ]

    query_results = build_query_results_from_tool_outputs(tool_outputs)

    assert len(query_results) == 1
    values = ResponseValidator()._build_value_index(query_results)
    assert any(v.metric_name == "avg_days_in_stage" and v.value == 7.35 for v in values), (
        f"expected the known row to index, got {values!r}"
    )


def test_shaping_single_query_and_junk() -> None:
    tool_outputs = [
        {
            "name": "query_recruitment_metrics",
            "response": {
                "success": True,
                "columns": ["total_hires"],
                "data": [{"total_hires": 42}],
            },
        },
        # Handoff / viz / non-query payloads carry no ``data`` list — skipped.
        {"name": "request_specialist_handoff", "response": {"status": "handoff_requested"}},
        {"name": "create_visualization", "response": {"success": True, "chart_id": "c1"}},
        {"name": "broken", "response": None},
    ]

    query_results = build_query_results_from_tool_outputs(tool_outputs)

    assert len(query_results) == 1
    values = ResponseValidator()._build_value_index(query_results)
    assert any(v.metric_name == "total_hires" and v.value == 42.0 for v in values)


# --- redaction on a flagged figure ---------------------------------------


def test_redacts_fabricated_figure_keeps_grounded() -> None:
    grounded = "Engineering averages 7.35 days in stage."
    fabricated = "The bottleneck threshold is 9.94783 days."
    answer = f"{grounded} {fabricated}"

    result = SpecialistRunResult(
        agent_name="pipeline_analyst",
        text=answer,
        tool_outputs=[
            {
                "name": "query_multiple_recruitment_metrics",
                "response": {
                    "success": True,
                    "results": [
                        {
                            "query_index": 0,
                            "success": True,
                            "columns": ["avg_days_in_stage"],
                            "data": [{"avg_days_in_stage": 7.35}],
                        }
                    ],
                },
            }
        ],
        response=_response(answer),
    )

    _enforce_grounding_redaction(result, _host(ResponseValidator()), "pipeline_analyst")

    # Fabricated figure neutralised in both text surfaces.
    assert "9.94783" not in result.text
    assert "9.94783" not in result.response.answer_markdown
    assert GROUNDING_REDACTION_MARKER in result.text
    # Grounded figure left untouched (no over-redaction).
    assert "7.35 days" in result.text
    assert "7.35 days" in result.response.answer_markdown
    # Caveat prepended exactly once, to both surfaces.
    assert result.text.startswith(GROUNDING_ENFORCEMENT_CAVEAT)
    assert result.response.answer_markdown.startswith(GROUNDING_ENFORCEMENT_CAVEAT)
    # Exactly one figure removed — the grounded one is still present.
    assert result.text.count(GROUNDING_REDACTION_MARKER) == 1


def test_diverged_surfaces_each_redacted_position_correct() -> None:
    """Legacy path strips orphaned chart refs from ``text`` after
    ``answer_markdown`` is set, so the two surfaces diverge. Each must be
    redacted against its OWN validation (positions are text-specific) — the
    chart marker in ``answer_markdown`` shifts every offset after it."""
    text = "Bottleneck is 9.94783 days."
    answer_md = "Bottleneck is [chart:abc] 9.94783 days."

    result = SpecialistRunResult(
        agent_name="pipeline_analyst",
        text=text,
        tool_outputs=[
            {
                "name": "query_recruitment_metrics",
                "response": {
                    "success": True,
                    "columns": ["avg_days_in_stage"],
                    "data": [{"avg_days_in_stage": 3.2}],
                },
            }
        ],
        response=_response(answer_md),
    )

    _enforce_grounding_redaction(result, _host(ResponseValidator()), "pipeline_analyst")

    assert "9.94783" not in result.text
    assert "9.94783" not in result.response.answer_markdown
    # The chart marker (non-numeric) is preserved — only the figure goes.
    assert "[chart:abc]" in result.response.answer_markdown
    assert result.text.startswith(GROUNDING_ENFORCEMENT_CAVEAT)
    assert result.response.answer_markdown.startswith(GROUNDING_ENFORCEMENT_CAVEAT)


def test_benchmark_style_figure_absent_from_data_is_redacted() -> None:
    """Documents an ACCEPTED behaviour by design: a benchmark-style
    figure sourced from the prompt/knowledge — not from query data — and
    phrased without a benchmark keyword is `no_match` and therefore
    redacted. The credentialed run is the guardrail that this does not
    degrade benchmark-citing cases; if it does, that is the escalation
    trigger, NOT a code change here."""
    answer = "The healthy range is 5-7 days for that stage."
    result = SpecialistRunResult(
        agent_name="pipeline_analyst",
        text=answer,
        tool_outputs=[
            {
                "name": "query_recruitment_metrics",
                "response": {
                    "success": True,
                    "columns": ["avg_days_in_stage"],
                    "data": [{"avg_days_in_stage": 3.2}],
                },
            }
        ],
        response=_response(answer),
    )

    _enforce_grounding_redaction(result, _host(ResponseValidator()), "pipeline_analyst")

    assert "7 days" not in result.text
    assert GROUNDING_REDACTION_MARKER in result.text


def test_clean_answer_is_untouched() -> None:
    answer = "Engineering averages 7.35 days in stage."
    result = SpecialistRunResult(
        agent_name="pipeline_analyst",
        text=answer,
        tool_outputs=[
            {
                "name": "query_recruitment_metrics",
                "response": {
                    "success": True,
                    "columns": ["avg_days_in_stage"],
                    "data": [{"avg_days_in_stage": 7.35}],
                },
            }
        ],
        response=_response(answer),
    )

    _enforce_grounding_redaction(result, _host(ResponseValidator()), "pipeline_analyst")

    assert result.text == answer
    assert result.response.answer_markdown == answer
    assert not result.text.startswith(GROUNDING_ENFORCEMENT_CAVEAT)


# --- scope gate ----------------------------------------------------------


def test_non_enforced_agent_is_skipped() -> None:
    validator = MagicMock()
    answer = "Some answer with 9.94783 days that would be flagged for pipeline_analyst."
    result = SpecialistRunResult(
        agent_name="data_scientist",
        text=answer,
        response=_response(answer),
    )

    _enforce_grounding_redaction(result, _host(validator), "data_scientist")

    validator.validate.assert_not_called()
    assert result.text == answer


def test_missing_validator_is_skipped() -> None:
    answer = "Answer with 9.94783 days."
    result = SpecialistRunResult(
        agent_name="pipeline_analyst",
        text=answer,
        response=_response(answer),
    )

    _enforce_grounding_redaction(result, _host(None), "pipeline_analyst")

    assert result.text == answer


# --- no agent_error ------------------------------------------------------


def test_redaction_never_sets_agent_error() -> None:
    answer = "The threshold is 9.94783 days."
    result = SpecialistRunResult(
        agent_name="pipeline_analyst",
        text=answer,
        # Real but unrelated data ⇒ the figure is `no_match` ⇒ redacted.
        tool_outputs=[
            {
                "name": "query_recruitment_metrics",
                "response": {
                    "success": True,
                    "columns": ["avg_days_in_stage"],
                    "data": [{"avg_days_in_stage": 3.2}],
                },
            }
        ],
        response=_response(answer),
    )

    _enforce_grounding_redaction(result, _host(ResponseValidator()), "pipeline_analyst")

    assert "9.94783" not in result.text
    assert result.response.agent_error is False
