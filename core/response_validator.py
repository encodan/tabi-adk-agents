"""
Post-response data validation.

Extracts numerical claims from agent responses and cross-references them
against actual query results. Discrepancies are logged as structured warnings
for observability — responses are never blocked.

Includes a structured DataValue index, trend verification, exclusion
patterns, configurable tolerances, and the ResponseValidator class.
"""

from __future__ import annotations

import math
import re
import secrets
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from config import ValidationConfig
    from core.specialist_schema import SpecialistResponse
    from models.planning import PlanningContext

from core.planning_claims import (
    ACTIVE_RECRUITERS,
    ACTUAL_YTD_HIRES,
    CAPACITY_CONSTRAINED_MAX_HIRES,
    HIRING_GAP,
    HIRING_TARGET,
    MONTHS_ELAPSED,
    MONTHS_REMAINING,
    PLANNING_METRICS,
    PROJECTED_FULL_YEAR_HIRES,
    PROJECTED_FULL_YEAR_HIRES_LOWER,
    PROJECTED_FULL_YEAR_HIRES_UPPER,
    RECRUITER_CAPACITY_PER_MONTH,
    SQID_PREFIX_COMPUTE_GOAL_ATTAINMENT,
)

logger = structlog.get_logger(__name__)

# --- Extraction patterns ---

# 12.3%, 85%, 0.5%
_PCT_RE = re.compile(r"(\d+\.?\d*)\s*%")

# "14.2 days", "3 weeks", "24 hours"
_DURATION_RE = re.compile(r"(\d+\.?\d*)\s*(days?|hours?|weeks?|months?)")

# Duration unit token, used by the sub-unit guard in ``_match_claim``. Wider
# than ``_DURATION_RE`` (which gates extraction) — the guard tolerates units
# the extractor may not yet emit so the family check stays correct if the
# extractor is widened later.
_DURATION_UNIT_RE = re.compile(r"(days?|hours?|weeks?|months?|years?|minutes?)", re.IGNORECASE)

# Single source of truth for the provenance-suffix format. ``annotate_claims``
# writes it; ``_PROVENANCE_SUFFIX_RE`` strips it for user-facing copy. The
# regex is derived from the format template so the two cannot drift out of
# sync — previous incidents had the inserter and the stripper diverging on
# the leading space.
_PROVENANCE_SUFFIX_FMT = " (source: {metric})"
_PROVENANCE_SUFFIX_RE = re.compile(
    re.escape(_PROVENANCE_SUFFIX_FMT.format(metric="<METRIC>")).replace("<METRIC>", "[^)]+")
)

# Currency: "$1,234", "$12.5K", "$1.2M"
_CURRENCY_RE = re.compile(r"\$\s*([\d,]+\.?\d*)\s*([KkMm])?")

# Standalone integers in metric contexts: "152 candidates", "38 hires"
_COUNT_RE = re.compile(
    r"\b([\d,]+)\b\s*(?:candidates?|applications?|hires?|offers?|interviews?"
    r"|roles?|jobs?|openings?|positions?|employees?|people|headcount"
    r"|rejections?|referrals?)",
    re.IGNORECASE,
)

# Ratios: "3.5x", "2:1"
_RATIO_RE = re.compile(r"(\d+\.?\d*)x\b")
_RATIO_COLON_RE = re.compile(r"(\d+):(\d+)\b")

# --- Exclusion patterns ---
# Numbers to exclude from claim extraction
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QUARTER_RE = re.compile(r"\bQ[1-4]\s*\d{4}\b", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b\d+(st|nd|rd|th)\b", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
_BENCHMARK_RE = re.compile(
    r"\b(?:industry\s+(?:benchmark|average|standard)|target|goal|benchmark)\b",
    re.IGNORECASE,
)

# Trailing-window phrases — chart titles, recency clauses, lookback windows.
# A number here describes a window, not a measured value. Matches e.g.
# "Last 12 Months", "past 30 days", "trailing 4 quarters". Without this,
# "Monthly Hiring Velocity (Last 12 Months)" in answer_markdown leaks
# "12 months" into the duration matcher.
_TRAILING_WINDOW_RE = re.compile(
    r"\b(?:last|previous|past|trailing|prior|over\s+the\s+last|over\s+the\s+past)\s*$",
    re.IGNORECASE,
)

# --- Trend patterns ---
_TREND_INCREASE_RE = re.compile(
    r"(?:increased|grew|rose|improved|jumped|surged|risen|gained|higher|up)\s+"
    r"(?:by\s+)?(\d+\.?\d*)",
    re.IGNORECASE,
)
_TREND_DECREASE_RE = re.compile(
    r"(?:decreased|dropped|fell|declined|reduced|shrank|lower|down)\s+"
    r"(?:by\s+)?(\d+\.?\d*)",
    re.IGNORECASE,
)
_TREND_STABLE_RE = re.compile(
    r"\b(?:remained|stayed|stable|flat|unchanged|consistent)\b",
    re.IGNORECASE,
)


@dataclass
class ClaimedValue:
    """A numerical claim extracted from the response text."""

    text_span: str
    parsed_value: float
    value_type: str  # "percentage" | "count" | "duration" | "currency" | "ratio"
    position: int = 0
    context: str = ""


@dataclass
class DataValue:
    """A value from query results."""

    value: float
    metric_name: str
    dimensions: dict[str, str] = field(default_factory=dict)
    query_index: int = 0


@dataclass
class Discrepancy:
    """A mismatch between a claimed value and the actual query data."""

    claimed: ClaimedValue
    nearest_match: DataValue | None
    match_type: str  # "exact" | "approximate" | "no_match" | "fabricated"
    deviation: float | None


@dataclass
class TrendClaim:
    """A directional claim about data."""

    direction: str  # "increased" | "decreased" | "stable"
    text_span: str
    magnitude: float | None = None


@dataclass
class ValidationResult:
    """Complete validation result for a response."""

    is_valid: bool
    claimed_values: list[ClaimedValue] = field(default_factory=list)
    data_values: list[DataValue] = field(default_factory=list)
    matches: list[tuple[ClaimedValue, DataValue]] = field(default_factory=list)
    discrepancies: list[Discrepancy] = field(default_factory=list)
    trend_checks: list[dict[str, Any]] = field(default_factory=list)
    validation_time_ms: float = 0.0


ValidationReport = ValidationResult


@dataclass
class Deviation:
    """Public shape of a single claim-vs-data mismatch.

    Thin projection of ``Discrepancy`` with a stable field names for the
    enforcement layer. Callers that walk the decision log
    consume ``relative_error`` rather than the legacy ``deviation`` field.
    """

    claim_text: str
    claim_value: float
    actual_value: float | None
    metric_name: str | None
    relative_error: float
    position: int
    value_type: str


def _discrepancy_to_deviation(d: Discrepancy) -> Deviation:
    return Deviation(
        claim_text=d.claimed.text_span,
        claim_value=d.claimed.parsed_value,
        actual_value=d.nearest_match.value if d.nearest_match else None,
        metric_name=d.nearest_match.metric_name if d.nearest_match else None,
        relative_error=d.deviation if d.deviation is not None else 0.0,
        position=d.claimed.position,
        value_type=d.claimed.value_type,
    )


@dataclass
class StructuredDeviation:
    """A deviation between a ``Claim.value`` and the ground-truth query value.

    Thin projection of the structured-claim validation result — distinct from
    ``Deviation`` so the regex-based validator's legacy events don't co-mingle
    with the structured path in the decision log.
    """

    claim_text_fragment: str
    metric: str
    claim_value: float
    actual_value: float | None
    source_query_id: str
    relative_error: float
    reason: Literal[
        "ok",
        "unresolved_query_id",
        "query_failed",
        "value_mismatch",
        "unparseable_filter",
    ]


@dataclass
class StructuredValidationReport:
    """Outcome of ``ResponseValidator.validate_structured``.

    ``deviations`` is non-empty when any claim failed (missing query_id,
    value mismatch beyond tolerance, or unparseable filters). ``is_valid``
    collapses the list for quick gating.
    """

    deviations: list[StructuredDeviation] = field(default_factory=list)
    claims_checked: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.deviations


@dataclass
class ValidationDecision:
    """Structured outcome from ``ResponseValidator.validate_and_enforce``.

    - ``outcome="pass"``: no action needed; ``annotated_text == response_text``.
    - ``outcome="annotate"``: replace the claim in-place with visible provenance.
    - ``outcome="regenerate"``: caller issues one correction pass; ``annotated_text``
      is the *fallback* returned if regen drifts or the budget is exhausted.

    ``annotated_text`` carries ``(source: <metric>)`` markers — the form fed
    back to the regen LLM. ``user_facing_text`` is the same string with the
    markers stripped, for the validation runner's user-visible fallback.
    """

    outcome: Literal["pass", "annotate", "regenerate"]
    report: ValidationResult
    annotated_text: str
    correction_prompt: str | None
    deviations: list[Deviation]
    regenerated: bool = False

    @property
    def user_facing_text(self) -> str:
        return _PROVENANCE_SUFFIX_RE.sub("", self.annotated_text)


def _parse_currency_value(amount_str: str, suffix: str | None) -> float:
    """Parse a currency string like '1,234' or '12.5' with optional K/M suffix."""
    cleaned = amount_str.replace(",", "")
    value = float(cleaned)
    if suffix:
        suffix_lower = suffix.lower()
        if suffix_lower == "k":
            value *= 1_000
        elif suffix_lower == "m":
            value *= 1_000_000
    return value


def _is_excluded_number(match: re.Match, text: str) -> bool:
    """Check if a regex match should be excluded from claim extraction."""
    start = match.start()
    # Get surrounding context (25 chars before) — narrow window to avoid
    # excluding numbers in the same sentence but different clauses
    context_start = max(0, start - 25)
    before_text = text[context_start:start]

    # Exclude if immediately preceded by benchmark/target language
    if _BENCHMARK_RE.search(before_text):
        return True

    # Exclude trailing-window phrases ("Last 12 Months", "past 30 days").
    # The trailing keyword sits *before* the number, hence the same context
    # window as the benchmark check.
    if _TRAILING_WINDOW_RE.search(before_text):
        return True

    # Check if the number is a year
    span = match.group(0)
    if _YEAR_RE.fullmatch(span.strip()):
        return True

    return False


def _get_sentence_context(text: str, position: int, window: int = 80) -> str:
    """Get surrounding text for debugging context."""
    start = max(0, position - window)
    end = min(len(text), position + window)
    return text[start:end].strip()


# Claim type → compatible metric unit(s) from dbt metadata
_CLAIM_TYPE_TO_METRIC_UNITS: dict[str, list[str]] = {
    "duration": ["days", "minutes"],
    "percentage": ["percentage"],
    "count": ["count"],
    "currency": ["currency"],
    "ratio": ["percentage"],
}


# Duration sub-unit families. Two durations from different families never
# match each other in ``_match_claim`` — without this guard, a "39 months"
# hallucination got paired with ``time_to_hire = 39 days``, producing the
# user-visible "39 months (source: time_to_hire)" artifact in the capacity
# planner.
_DURATION_UNIT_FAMILY: dict[str, str] = {
    "minutes": "sub_day",
    "minute": "sub_day",
    "hours": "sub_day",
    "hour": "sub_day",
    "days": "day_scale",
    "day": "day_scale",
    "weeks": "day_scale",
    "week": "day_scale",
    "months": "month_scale",
    "month": "month_scale",
    "years": "month_scale",
    "year": "month_scale",
}


def _duration_unit_family(unit_token: str | None) -> str | None:
    """Resolve a duration unit token (e.g. ``months``) to its family.

    Returns ``None`` when the unit is unknown or absent — callers should
    treat that as "no guard" and fall through to the default matcher.
    """
    if not unit_token:
        return None
    return _DURATION_UNIT_FAMILY.get(unit_token.lower())


def _filter_to_duration_family(
    claim: ClaimedValue,
    candidates: list[DataValue],
    metric_units: dict[str, str | None] | None,
) -> list[DataValue]:
    """Restrict ``candidates`` to metrics in the same duration family as ``claim``.

    The guard exists to prevent cross-family numerical coincidences from
    producing user-visible artifacts like "39 months (source: time_to_hire)".

    Returns ``candidates`` unchanged when the guard doesn't apply:
        - non-duration claim
        - no recognizable unit token in the claim text
        - ``metric_units`` missing
    Returns an empty list when the guard applies but every candidate
    lives in a different family — callers MUST treat that as
    ``unverifiable`` and not fall through to the type-agnostic matcher,
    which would force a unit-wrong but numerically-near match.
    """
    if claim.value_type != "duration" or not candidates or not metric_units:
        return candidates
    unit_match = _DURATION_UNIT_RE.search(claim.text_span)
    claim_family = _duration_unit_family(unit_match.group(1) if unit_match else None)
    if claim_family is None:
        return candidates
    return [
        dv
        for dv in candidates
        if _duration_unit_family(metric_units.get(dv.metric_name)) == claim_family
    ]


class ResponseValidator:
    """Validates agent response claims against query results."""

    def __init__(
        self,
        tolerance: float = 0.05,
        absolute_tolerance: float = 0.5,
    ) -> None:
        self.tolerance = tolerance
        self.absolute_tolerance = absolute_tolerance

    def validate(
        self,
        response_text: str,
        query_results: list[dict[str, Any]],
        metric_units: dict[str, str | None] | None = None,
    ) -> ValidationResult:
        """
        Validate response claims against query results.

        Steps:
        1. Extract claimed values from response text
        2. Build value index from query results
        3. Match each claim to nearest data value
        4. Flag unmatched claims as discrepancies
        5. Verify trend claims against data ordering
        """
        start = time.perf_counter()

        claims = self._extract_claims(response_text)
        data_values = self._build_value_index(query_results)

        matches: list[tuple[ClaimedValue, DataValue]] = []
        discrepancies: list[Discrepancy] = []

        if claims and data_values:
            for claim in claims:
                nearest, match_type, deviation = self._match_claim(
                    claim,
                    data_values,
                    metric_units=metric_units,
                )
                if match_type in ("exact", "approximate", "derived"):
                    if nearest is not None:
                        matches.append((claim, nearest))
                elif match_type == "unverifiable":
                    # Claim cannot be checked against the available data
                    # (e.g., percentage-of-total when no derivable shares
                    # exist). Skip silently — neither a verified match nor
                    # a discrepancy. Avoids false positives that force the
                    # model to regenerate against bogus nearest values.
                    continue
                else:
                    discrepancies.append(
                        Discrepancy(
                            claimed=claim,
                            nearest_match=nearest,
                            match_type=match_type,
                            deviation=deviation,
                        )
                    )

        trend_checks = self._check_trends(response_text, query_results)

        # Trend discrepancies count toward validity
        trend_discrepancies = [t for t in trend_checks if not t.get("verified", True)]

        elapsed = (time.perf_counter() - start) * 1000
        return ValidationResult(
            is_valid=len(discrepancies) == 0 and len(trend_discrepancies) == 0,
            claimed_values=claims,
            data_values=data_values,
            matches=matches,
            discrepancies=discrepancies,
            trend_checks=trend_checks,
            validation_time_ms=elapsed,
        )

    def validate_structured(
        self,
        response: SpecialistResponse,
        keyed_results: Mapping[str, Any],
        *,
        config: ValidationConfig,
    ) -> StructuredValidationReport:
        """Validate a structured ``SpecialistResponse`` against per-query
        ground truth.

        Each ``Claim.source_query_id`` is resolved against ``keyed_results``
        (the per-turn dict threaded through by ``AgentSession``). A claim is
        "ok" when its ``value`` is within the configured tolerance of the
        matching value in the query result; otherwise it is flagged as a
        deviation with a stable ``reason`` so dashboards can split missing-id
        failures from query failures from numerical divergence.

        ``relative_tolerance`` is read from ``config.soft_threshold`` (default
        5%) — matches the regex-path policy so the two validators remain
        comparable.  ``absolute_tolerance`` covers near-zero values where a
        purely relative bound would explode.
        """
        rel_tol = config.soft_threshold
        abs_tol = config.absolute_tolerance
        deviations: list[StructuredDeviation] = []

        for claim in response.claims:
            result = keyed_results.get(claim.source_query_id)
            if result is None:
                deviations.append(
                    StructuredDeviation(
                        claim_text_fragment=claim.text_fragment,
                        metric=claim.metric,
                        claim_value=claim.value,
                        actual_value=None,
                        source_query_id=claim.source_query_id,
                        relative_error=1.0,
                        reason="unresolved_query_id",
                    )
                )
                continue
            if not isinstance(result, dict) or not result.get("success"):
                # The id matched, but the underlying query failed (timeout,
                # permission error, etc.). Distinct from filter problems so
                # dashboards can separate infra failures from model error.
                deviations.append(
                    StructuredDeviation(
                        claim_text_fragment=claim.text_fragment,
                        metric=claim.metric,
                        claim_value=claim.value,
                        actual_value=None,
                        source_query_id=claim.source_query_id,
                        relative_error=1.0,
                        reason="query_failed",
                    )
                )
                continue

            actual = self._lookup_actual(claim, keyed_results)
            if actual is None:
                # Query succeeded but no row resolved — either the metric name
                # disagrees with the query's columns or a filter was unparseable.
                deviations.append(
                    StructuredDeviation(
                        claim_text_fragment=claim.text_fragment,
                        metric=claim.metric,
                        claim_value=claim.value,
                        actual_value=None,
                        source_query_id=claim.source_query_id,
                        relative_error=1.0,
                        reason="unparseable_filter",
                    )
                )
                continue

            abs_err = abs(claim.value - actual)
            rel_err = abs_err / max(abs(actual), 1e-9)
            if abs_err > abs_tol and rel_err > rel_tol:
                deviations.append(
                    StructuredDeviation(
                        claim_text_fragment=claim.text_fragment,
                        metric=claim.metric,
                        claim_value=claim.value,
                        actual_value=actual,
                        source_query_id=claim.source_query_id,
                        relative_error=rel_err,
                        reason="value_mismatch",
                    )
                )

        return StructuredValidationReport(
            deviations=deviations,
            claims_checked=len(response.claims),
        )

    @staticmethod
    def _lookup_actual(
        claim: Any,  # ``Claim`` — kept as Any to avoid a cycle with specialist_schema
        keyed_results: Mapping[str, Any],
    ) -> float | None:
        """Resolve a claim's ground-truth numeric value from its source query.

        Scans the query's data rows for a column whose name matches
        ``claim.metric``. For grouped queries we apply a best-effort filter
        match on dimension columns: if every filter in ``claim.filters``
        disagrees with a row, that row is skipped. Returns the first matching
        row's ``metric`` value, or ``None`` if the query is missing, failed,
        or has no matching row/column.

        ``_row_matches_filters`` is intentionally lenient about *absent*
        dimension columns (a scalar-only query has nothing to filter on, so
        any filter trivially matches), which means the main loop already
        covers scalar queries. There is no row[0] fallback when the filter
        column is *present but disagrees* on every row — silently returning
        row[0] would compare a claim against the wrong segment and either
        miss a real deviation or invent a spurious one.
        """
        result = keyed_results.get(claim.source_query_id)
        if not isinstance(result, dict) or not result.get("success"):
            return None
        rows = result.get("data") or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not _row_matches_filters(row, claim.filters):
                continue
            value = row.get(claim.metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    def _extract_claims(self, text: str) -> list[ClaimedValue]:
        """Extract numerical claims from response text."""
        claims: list[ClaimedValue] = []
        seen_spans: set[str] = set()

        # Build a set of positions to exclude (ordinals, list markers, years, quarters)
        excluded_positions: set[int] = set()
        for pattern in [_ORDINAL_RE, _LIST_MARKER_RE, _QUARTER_RE]:
            for m in pattern.finditer(text):
                for pos in range(m.start(), m.end()):
                    excluded_positions.add(pos)

        for m in _YEAR_RE.finditer(text):
            # Only exclude standalone years (not part of larger numbers)
            before = text[m.start() - 1] if m.start() > 0 else " "
            after = text[m.end()] if m.end() < len(text) else " "
            if not before.isdigit() and not after.isdigit():
                for pos in range(m.start(), m.end()):
                    excluded_positions.add(pos)

        def _should_add(m: re.Match, span: str, value: float, vtype: str) -> bool:
            if span in seen_spans:
                return False
            if m.start() in excluded_positions:
                return False
            if _is_excluded_number(m, text):
                return False
            return True

        for m in _PCT_RE.finditer(text):
            span = m.group(0)
            if _should_add(m, span, float(m.group(1)), "percentage"):
                seen_spans.add(span)
                claims.append(
                    ClaimedValue(
                        text_span=span,
                        parsed_value=float(m.group(1)),
                        value_type="percentage",
                        position=m.start(),
                        context=_get_sentence_context(text, m.start()),
                    )
                )

        for m in _DURATION_RE.finditer(text):
            span = m.group(0)
            if _should_add(m, span, float(m.group(1)), "duration"):
                seen_spans.add(span)
                claims.append(
                    ClaimedValue(
                        text_span=span,
                        parsed_value=float(m.group(1)),
                        value_type="duration",
                        position=m.start(),
                        context=_get_sentence_context(text, m.start()),
                    )
                )

        for m in _CURRENCY_RE.finditer(text):
            span = m.group(0)
            value = _parse_currency_value(m.group(1), m.group(2))
            if _should_add(m, span, value, "currency"):
                seen_spans.add(span)
                claims.append(
                    ClaimedValue(
                        text_span=span,
                        parsed_value=value,
                        value_type="currency",
                        position=m.start(),
                        context=_get_sentence_context(text, m.start()),
                    )
                )

        for m in _COUNT_RE.finditer(text):
            span = m.group(0)
            value = float(m.group(1).replace(",", ""))
            if _should_add(m, span, value, "count"):
                seen_spans.add(span)
                claims.append(
                    ClaimedValue(
                        text_span=span,
                        parsed_value=value,
                        value_type="count",
                        position=m.start(),
                        context=_get_sentence_context(text, m.start()),
                    )
                )

        for m in _RATIO_RE.finditer(text):
            span = m.group(0)
            if _should_add(m, span, float(m.group(1)), "ratio"):
                seen_spans.add(span)
                claims.append(
                    ClaimedValue(
                        text_span=span,
                        parsed_value=float(m.group(1)),
                        value_type="ratio",
                        position=m.start(),
                        context=_get_sentence_context(text, m.start()),
                    )
                )

        for m in _RATIO_COLON_RE.finditer(text):
            span = m.group(0)
            value = float(m.group(1)) / float(m.group(2)) if float(m.group(2)) != 0 else 0.0
            if _should_add(m, span, value, "ratio"):
                seen_spans.add(span)
                claims.append(
                    ClaimedValue(
                        text_span=span,
                        parsed_value=round(value, 4),
                        value_type="ratio",
                        position=m.start(),
                        context=_get_sentence_context(text, m.start()),
                    )
                )

        return claims

    def _build_value_index(self, query_results: list[dict[str, Any]]) -> list[DataValue]:
        """Build flat value index from query results."""
        values: list[DataValue] = []
        for qi, result in enumerate(query_results):
            if not result.get("success"):
                continue
            rows = result.get("data", [])
            # Limit to first 100 rows for performance
            for row in rows[:100]:
                if not isinstance(row, dict):
                    continue
                # Separate numeric values from dimension values
                dimensions: dict[str, str] = {}
                for k, v in row.items():
                    if isinstance(v, str):
                        dimensions[k] = v

                for k, v in row.items():
                    if isinstance(v, (int, float)) and v is not True and v is not False:
                        values.append(
                            DataValue(
                                value=float(v),
                                metric_name=k,
                                dimensions=dimensions,
                                query_index=qi,
                            )
                        )
        return values

    def _classify_match(
        self,
        claim: ClaimedValue,
        nearest: DataValue,
        all_values: list[DataValue],
    ) -> tuple[DataValue, str, float]:
        """Apply tolerance logic to a claim/value pair.

        Returns (best_match, match_type, deviation).
        """
        deviation = _compute_deviation(claim.parsed_value, nearest.value)

        if abs(claim.parsed_value - nearest.value) < 0.01:
            return nearest, "exact", 0.0

        abs_diff = abs(claim.parsed_value - nearest.value)
        if abs(nearest.value) < 10 and abs_diff <= self.absolute_tolerance:
            return nearest, "approximate", deviation

        if deviation <= self.tolerance:
            return nearest, "approximate", deviation

        derived = self._check_derived_value(claim, all_values)
        if derived is not None:
            return derived, "derived", 0.0

        return nearest, "no_match", deviation

    def _match_claim(
        self,
        claim: ClaimedValue,
        values: list[DataValue],
        metric_units: dict[str, str | None] | None = None,
    ) -> tuple[DataValue | None, str, float | None]:
        """Find the closest matching data value for a claim."""
        if not values:
            return None, "fabricated", None

        # Pass 1: match within same type family (if metric units available)
        if metric_units:
            compatible_units = _CLAIM_TYPE_TO_METRIC_UNITS.get(claim.value_type, [])
            same_type_values = [
                dv for dv in values if metric_units.get(dv.metric_name) in compatible_units
            ]
            # Duration sub-unit guard: a "months" claim must not match a
            # "days" metric (see ``_filter_to_duration_family``). An empty
            # filtered pool short-circuits to ``unverifiable`` rather than
            # falling through to Pass 2 — Pass 2's type-agnostic matcher
            # would otherwise force a numerically-near but unit-wrong match.
            if claim.value_type == "duration" and same_type_values:
                filtered = _filter_to_duration_family(claim, same_type_values, metric_units)
                if not filtered:
                    return None, "unverifiable", None
                same_type_values = filtered
            if same_type_values:
                nearest = min(same_type_values, key=lambda dv: abs(dv.value - claim.parsed_value))
                return self._classify_match(claim, nearest, same_type_values)

        # Percentage claims must NEVER be compared against arbitrary raw
        # values — that's how "35%" got matched against a count of 29 and
        # forced a regeneration into wrong text. Two ordered fallbacks:
        #
        #   1. Multi-row data → derive "% of column total" and match
        #      against those shares. Authoritative for percentage-of-
        #      total claims like "Engineering: 35% of hires" (194/547 =
        #      35.47%). Raw counts in the data are explicitly excluded
        #      from the candidate pool here so a small count near the
        #      claim (e.g., Sales=58 ≈ "60%") cannot create a false
        #      match alongside a real-but-distant derived share.
        #
        #   2. No derivation possible (single-row data, zero sums) →
        #      direct match against raw values in plausible percentage
        #      range [0, 100]. Covers the legacy case where a metric
        #      like ``hire_rate=12.3`` lacks ``metric_units`` annotation.
        #
        # Otherwise: claim is unverifiable. Skip rather than manufacture
        # a false discrepancy.
        if claim.value_type == "percentage":
            derived_pcts = self._derive_percentages_from_data(values)
            if derived_pcts:
                nearest = min(derived_pcts, key=lambda dv: abs(dv.value - claim.parsed_value))
                return self._classify_match(claim, nearest, derived_pcts)
            plausible = [dv for dv in values if 0 <= dv.value <= 100]
            if plausible:
                nearest = min(plausible, key=lambda dv: abs(dv.value - claim.parsed_value))
                return self._classify_match(claim, nearest, plausible)
            return None, "unverifiable", None

        # Pass 2: fall back to type-agnostic matching (non-percentage only)
        nearest = min(values, key=lambda dv: abs(dv.value - claim.parsed_value))
        return self._classify_match(claim, nearest, values)

    def _derive_percentages_from_data(self, values: list[DataValue]) -> list[DataValue]:
        """Derive ``% of column total`` from raw numeric data.

        Groups values by ``(query_index, metric_name)``. For groups with a
        non-zero sum and ≥2 rows, emits one synthetic ``DataValue`` per row
        at ``value = row / sum * 100``. Lets the validator verify
        percentage-of-total claims like "Engineering: 35% of hires" when
        the data contains only raw counts (no percentage-typed metric).

        Synthetic values use ``metric_name = f"{col}_share_pct"`` so a
        downstream ``response_claim_discrepancy`` log naming the metric
        signals it was a derived check.
        """
        from collections import defaultdict

        grouped: defaultdict[tuple[int, str], list[DataValue]] = defaultdict(list)
        for dv in values:
            grouped[(dv.query_index, dv.metric_name)].append(dv)

        derived: list[DataValue] = []
        for (qi, metric), group in grouped.items():
            if len(group) < 2:
                continue
            total = sum(dv.value for dv in group)
            if total == 0:
                continue
            share_metric = f"{metric}_share_pct"
            for dv in group:
                derived.append(
                    DataValue(
                        value=(dv.value / total) * 100,
                        metric_name=share_metric,
                        dimensions=dv.dimensions,
                        query_index=qi,
                    )
                )
        return derived

    def _check_sum_aggregation(
        self,
        claim: ClaimedValue,
        values: list[DataValue],
    ) -> DataValue | None:
        """Check if a claimed count is a sum of consecutive values from the same metric.

        Handles cases like quarterly totals derived from monthly data rows.
        Checks window sizes 2-4 (covering bi-monthly through quarterly aggregations).
        """
        from collections import defaultdict

        by_metric: defaultdict[str, list[DataValue]] = defaultdict(list)
        for dv in values:
            by_metric[dv.metric_name].append(dv)

        for metric_values in by_metric.values():
            if len(metric_values) < 2:
                continue
            for window_size in (2, 3, 4):
                if len(metric_values) < window_size:
                    continue
                for i in range(len(metric_values) - window_size + 1):
                    window = metric_values[i : i + window_size]
                    window_sum = sum(dv.value for dv in window)
                    if abs(window_sum - claim.parsed_value) < max(1.0, window_sum * 0.02):
                        return window[0]

        return None

    def _check_derived_value(
        self,
        claim: ClaimedValue,
        values: list[DataValue],
    ) -> DataValue | None:
        """Check if a claimed value is a derived calculation (sum, average, pct change)."""
        # Check sum-based aggregations for count values (e.g. quarterly totals)
        if claim.value_type == "count":
            match = self._check_sum_aggregation(claim, values)
            if match is not None:
                return match

        if claim.value_type != "percentage":
            return None

        # Check if the claimed percentage matches a percentage change between
        # two values: (new - old) / old * 100
        # Cap at 50 values to bound the O(n²) pair comparison
        numeric_values = [dv.value for dv in values if dv.value != 0][:50]
        for i, v1 in enumerate(numeric_values):
            for v2 in numeric_values[i + 1 :]:
                pct_change = abs((v2 - v1) / v1 * 100)
                if abs(pct_change - claim.parsed_value) < 1.0:  # 1% tolerance for derived
                    return values[i]
                pct_change_rev = abs((v1 - v2) / v2 * 100)
                if abs(pct_change_rev - claim.parsed_value) < 1.0:
                    return values[i]

        return None

    # ------------------------------------------------------------------
    # Phase-2 enforcement surface
    # ------------------------------------------------------------------
    def validate_and_enforce(
        self,
        response_text: str,
        query_results: list[dict[str, Any]],
        *,
        config: ValidationConfig,
        metric_units: dict[str, str | None] | None = None,
    ) -> ValidationDecision:
        """Run validation and return a structured action decision.

        The decision tree:
        - worst relative error < ``soft_threshold`` → ``pass``
        - worst relative error < ``hard_threshold`` → ``annotate``
        - worst relative error ≥ ``hard_threshold`` → ``regenerate`` (with
          ``annotated_text`` populated as the fallback)
        """
        report = self.validate(response_text, query_results, metric_units)
        deviations = [
            _discrepancy_to_deviation(d)
            for d in report.discrepancies
            if d.nearest_match is not None and d.deviation is not None
        ]
        worst = max((d.relative_error for d in deviations), default=0.0)

        if worst < config.soft_threshold:
            return ValidationDecision(
                outcome="pass",
                report=report,
                annotated_text=response_text,
                correction_prompt=None,
                deviations=deviations,
            )

        annotated = self.annotate_claims(response_text, report)

        if worst < config.hard_threshold:
            return ValidationDecision(
                outcome="annotate",
                report=report,
                annotated_text=annotated,
                correction_prompt=None,
                deviations=deviations,
            )

        return ValidationDecision(
            outcome="regenerate",
            report=report,
            annotated_text=annotated,
            correction_prompt=self._build_correction_prompt(response_text, report),
            deviations=deviations,
        )

    def annotate_claims(
        self,
        response_text: str,
        report: ValidationResult,
    ) -> str:
        """Replace flagged claim substrings with source-true values + provenance
        marker.

        Iterates discrepancies in reverse position order so earlier positions
        stay valid while later substrings are replaced. Safe to call when
        there are no discrepancies — the text passes through unchanged.
        """
        flagged = [
            d
            for d in report.discrepancies
            if d.nearest_match is not None and d.deviation is not None
        ]
        if not flagged:
            return response_text

        flagged.sort(key=lambda d: d.claimed.position, reverse=True)
        result = response_text
        for d in flagged:
            claim = d.claimed
            actual = d.nearest_match
            if actual is None:  # narrowed above, but placates type checker
                continue
            start = claim.position
            end = start + len(claim.text_span)
            # Defensive: positions come from the pre-edit text; skip if the
            # substring doesn't match what we expect (non-unique spans
            # earlier in the text could have moved it).
            if result[start:end] != claim.text_span:
                continue
            formatted = self._format_source_value(actual.value, claim)
            replacement = formatted + _PROVENANCE_SUFFIX_FMT.format(metric=actual.metric_name)
            result = result[:start] + replacement + result[end:]
        return result

    def annotate_claims_user_facing(
        self,
        response_text: str,
        report: ValidationResult,
    ) -> str:
        """User-facing counterpart to :meth:`annotate_claims` — same span
        replacements, but with the ``(source: <metric>)`` provenance suffix
        stripped. The suffix is meaningful feedback to the regen LLM but
        leaks internal metric names if rendered to the end user."""
        return _PROVENANCE_SUFFIX_RE.sub("", self.annotate_claims(response_text, report))

    @staticmethod
    def _format_source_value(actual_value: float, claim: ClaimedValue) -> str:
        """Render a source-true numeric value in the same style as the claim span.

        Duration units (``days``, ``hours``, etc.) are recovered from the
        original claim text so the annotated span reads naturally.
        """
        if claim.value_type == "percentage":
            return f"{actual_value:.1f}%"
        if claim.value_type == "currency":
            return f"${actual_value:,.0f}"
        if claim.value_type == "count":
            return f"{actual_value:,.0f}"
        if claim.value_type == "ratio":
            return f"{actual_value:.2f}x"
        if claim.value_type == "duration":
            unit_match = re.search(r"(days?|hours?|weeks?|months?)", claim.text_span, re.IGNORECASE)
            unit = unit_match.group(1) if unit_match else ""
            if unit:
                return f"{actual_value:g} {unit}"
            return f"{actual_value:g}"
        return f"{actual_value:g}"

    def _build_correction_prompt(
        self,
        response_text: str,
        report: ValidationResult,
    ) -> str:
        """Render the explicit "change ONLY these substrings" prompt.

        Claims are listed in document order so the model can scan the prompt
        top-to-bottom against the verbatim response.
        """
        flagged = [
            d
            for d in report.discrepancies
            if d.nearest_match is not None and d.deviation is not None
        ]
        flagged.sort(key=lambda d: d.claimed.position)

        error_lines: list[str] = []
        for d in flagged:
            claim = d.claimed
            actual = d.nearest_match
            if actual is None:
                continue
            formatted_actual = self._format_source_value(actual.value, claim)
            # d.deviation is a fraction (0.071 == 7.1%); render as a percent.
            error_lines.append(
                f'  - "{claim.text_span}" → "{formatted_actual}" '
                f"(metric: {actual.metric_name}, relative error "
                f"{(d.deviation or 0.0) * 100:.1f}%)"
            )
        errors_block = (
            "\n".join(error_lines)
            if error_lines
            else "  (no specific substring errors — response looks fine)"
        )

        fence = _pick_fence(response_text)
        return (
            "The previous response contained numerically incorrect claims.\n\n"
            f"Previous response (verbatim, enclosed by {fence}):\n"
            f"{fence}\n"
            f"{response_text}\n"
            f"{fence}\n\n"
            f"Specific errors (change ONLY these substrings):\n{errors_block}\n\n"
            "Return the previous response with ONLY the listed substrings "
            "replaced. All other text — prose, structure, headings, tables, "
            "chart references — must be byte-identical to the input. Do not "
            "apologise, re-interpret, or reorder."
        )

    @staticmethod
    def drift_detected(original: str, regenerated: str) -> bool:
        """True if the regenerated response changed non-numeric prose
        (a known failure mode).

        Strategy: strip tokens that contain any digit (those are the only
        ones regen is allowed to touch), then compare the remaining word
        multisets. A mismatch means the model paraphrased outside the
        flagged claims and we should fall through to annotation.
        """
        orig_tokens = _tokens_without_digits(original)
        regen_tokens = _tokens_without_digits(regenerated)
        return Counter(orig_tokens) != Counter(regen_tokens)

    def _check_trends(
        self,
        text: str,
        query_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Verify directional claims against data."""
        trend_checks: list[dict[str, Any]] = []

        # Extract trend claims
        trend_claims: list[TrendClaim] = []
        for m in _TREND_INCREASE_RE.finditer(text):
            trend_claims.append(
                TrendClaim(
                    direction="increased",
                    text_span=m.group(0),
                    magnitude=float(m.group(1)) if m.group(1) else None,
                )
            )
        for m in _TREND_DECREASE_RE.finditer(text):
            trend_claims.append(
                TrendClaim(
                    direction="decreased",
                    text_span=m.group(0),
                    magnitude=float(m.group(1)) if m.group(1) else None,
                )
            )
        for m in _TREND_STABLE_RE.finditer(text):
            trend_claims.append(
                TrendClaim(
                    direction="stable",
                    text_span=m.group(0),
                )
            )

        if not trend_claims:
            return trend_checks

        # Pre-compute direction for every numeric column across all results
        col_directions: list[dict[str, Any]] = []
        for result in query_results:
            if not result.get("success"):
                continue
            rows = result.get("data", [])
            if len(rows) < 2 or not isinstance(rows[0], dict):
                continue

            numeric_cols = [
                k
                for k, v in rows[0].items()
                if isinstance(v, (int, float)) and v is not True and v is not False
            ]
            for col in numeric_cols:
                col_values = [
                    float(row[col])
                    for row in rows
                    if isinstance(row.get(col), (int, float))
                    and row[col] is not True
                    and row[col] is not False
                ]
                if len(col_values) < 2:
                    continue
                first_val, last_val = col_values[0], col_values[-1]
                if last_val > first_val:
                    direction = "increased"
                elif last_val < first_val:
                    direction = "decreased"
                else:
                    direction = "stable"
                col_directions.append(
                    {
                        "metric": col,
                        "direction": direction,
                        "first_value": first_val,
                        "last_value": last_val,
                    }
                )

        if not col_directions:
            return trend_checks

        # For each trend claim, find best matching column. A claim is
        # verified if ANY column's direction matches — avoids false
        # negatives from unrelated columns in the same result set.
        for tc in trend_claims:
            matched: dict[str, Any] | None = None
            first_seen: dict[str, Any] | None = None
            for cd in col_directions:
                if first_seen is None:
                    first_seen = cd
                if tc.direction == cd["direction"]:
                    matched = cd
                    break

            ref = matched or first_seen
            if ref is None:
                continue

            check: dict[str, Any] = {
                "claim_direction": tc.direction,
                "data_direction": ref["direction"],
                "metric": ref["metric"],
                "text_span": tc.text_span,
                "verified": matched is not None,
                "first_value": ref["first_value"],
                "last_value": ref["last_value"],
            }

            # Verify magnitude only when direction already matches
            if matched and tc.magnitude is not None and ref["first_value"] != 0:
                actual_change = abs(
                    (ref["last_value"] - ref["first_value"]) / ref["first_value"] * 100
                )
                check["claimed_magnitude"] = tc.magnitude
                check["actual_magnitude"] = round(actual_change, 1)
                if abs(actual_change - tc.magnitude) > tc.magnitude * self.tolerance + 1.0:
                    check["verified"] = False

            trend_checks.append(check)

        return trend_checks


def _split_pipe_alternatives(value: str) -> list[str]:
    """Split a pipe-delimited multi-value filter literal (``"Engineering|Data"``).

    The pipe convention is defined by ``specialist_schema.ClaimFilters``.
    """
    return [v.strip() for v in value.split("|") if v.strip()]


def _row_matches_filters(row: dict[str, Any], filters: Mapping[str, str]) -> bool:
    """Best-effort filter-match for structured-claim resolution.

    ``filters`` is a ``dict[str, str]`` emitted by the model. A filter matches
    if either the exact dimension name OR any column that ends with
    ``__<dim>`` carries a value that equals the filter value or appears in
    the pipe-separated alternatives (``"Engineering|Data"``). Filter keys
    that don't appear in the row at all are treated as matching — we only
    reject rows whose explicit dimension values disagree, because the row
    being returned at all implies the underlying query filter was honoured.
    """
    if not filters:
        return True
    for key, value in filters.items():
        alternatives = set(_split_pipe_alternatives(value))
        matched = None
        for col, col_value in row.items():
            if col == key or col.endswith(f"__{key}"):
                matched = col_value
                break
        if matched is None:
            continue  # column absent → can't contradict
        if str(matched) not in alternatives:
            return False
    return True


def _compute_deviation(claimed: float, actual: float) -> float:
    """Relative deviation as a fraction (0.0 = exact match)."""
    if actual == 0:
        return abs(claimed) if claimed != 0 else 0.0
    return abs(claimed - actual) / abs(actual)


def _tokens_without_digits(text: str) -> list[str]:
    """Return whitespace-split tokens, dropping any that contain a digit.

    Used by ``ResponseValidator.drift_detected``: the regen path is allowed
    to change numeric spans but nothing else, so the non-numeric token
    multiset should be stable across original and regenerated responses.

    NFC-normalized before tokenizing so responses that differ only in
    unicode composition (e.g. é as U+00E9 vs U+0065 U+0301) don't register
    as drift.
    """
    normalized = unicodedata.normalize("NFC", text)
    return [t for t in re.findall(r"\S+", normalized) if not any(c.isdigit() for c in t)]


def _pick_fence(response_text: str) -> str:
    """Return a delimiter string guaranteed to not appear in ``response_text``.

    Used by ``_build_correction_prompt`` so that adversarial response content
    (e.g. a response that literally contains ``>>>`` or other delimiter
    markers) cannot break out of the verbatim-quoted block and reshape the
    correction instruction. Loops with fresh randomness until clear — the
    collision probability on the first 64-bit token is ~2^-64.
    """
    while True:
        candidate = f"<<<TABI_FENCE_{secrets.token_hex(8)}>>>"
        if candidate not in response_text:
            return candidate


# --- Convenience functions (backward-compatible with the original free-function API) ---


def extract_claimed_values(response_text: str) -> list[ClaimedValue]:
    """Extract all numerical claims from agent response text."""
    return ResponseValidator()._extract_claims(response_text)


def validate_response(
    response_text: str,
    query_results: list[dict[str, Any]],
    deviation_threshold: float = 0.05,
    metric_units: dict[str, str | None] | None = None,
) -> ValidationResult:
    """
    Validate agent response claims against actual query results.

    Backward-compatible wrapper around ResponseValidator.validate().
    """
    validator = ResponseValidator(tolerance=deviation_threshold)
    return validator.validate(response_text, query_results, metric_units=metric_units)


# ---------------------------------------------------------------------------
# Structural grounding enforcement (per the grounding design)
# ---------------------------------------------------------------------------
#
# Pure text-surgery over a ``ValidationResult`` — kept here (not in the
# runner) so the dominant false-positive risk (tool-output → query_results
# shaping) and the redaction edit are unit-testable without an ADK session.

# Sentinel substituted for a redacted figure. Kept neutral and bracketed so
# the groundedness judge segments it as a non-numeric token rather than an
# unsupported measured value.
GROUNDING_REDACTION_MARKER: str = "[unverified]"

# Discrepancy match types eligible for redaction: invented composites
# (``fabricated`` — no data at all) and figures with no nearby data value
# (``no_match``). ``unverifiable`` is never here (dropped upstream in
# ``validate``); ``exact``/``approximate``/``derived`` are grounded and
# never reach ``discrepancies``.
_REDACTABLE_MATCH_TYPES: frozenset[str] = frozenset({"fabricated", "no_match"})


def build_query_results_from_tool_outputs(
    tool_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten specialist tool outputs into the ``query_results`` shape
    :meth:`ResponseValidator.validate` / :meth:`_build_value_index` consume.

    ``tool_outputs`` entries are ``{"name", "response"}`` dicts (see
    ``handoff.collect_tool_response``). The query tools return one of:

    - single: ``{"success": True, "data": [...], "columns": [...]}``
      (``query_recruitment_metrics``)
    - multi: ``{"success": True, "results": [{"success": True, "data":
      [...], ...}, ...]}`` (``query_multiple_recruitment_metrics``)

    Only dicts carrying a ``data`` list — directly or nested under
    ``results`` — contribute. Everything else (handoffs, viz, report
    operations, failures, non-dict payloads) is skipped. The output is a
    flat ``list[{"success", "data", ...}]`` — the exact structure
    ``_build_value_index`` indexes. Getting this shape wrong yields zero
    data values and over-redaction of legitimate figures, so it is
    deliberately conservative and unit-tested.
    """
    flattened: list[dict[str, Any]] = []
    for output in tool_outputs:
        if not isinstance(output, dict):
            continue
        response = output.get("response")
        if not isinstance(response, dict):
            continue
        nested = response.get("results")
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict) and isinstance(item.get("data"), list):
                    flattened.append(item)
            continue
        if isinstance(response.get("data"), list):
            flattened.append(response)
    return flattened


def redact_unsourced_figures(
    text: str,
    report: ValidationResult,
    *,
    marker: str = GROUNDING_REDACTION_MARKER,
) -> tuple[str, list[str]]:
    """Neutralise validator-flagged unsourced figures in ``text``.

    Targets the ``_REDACTABLE_MATCH_TYPES`` ``Discrepancy`` rows — invented
    composites / trends with no support in the query data.

    Replaces each flagged ``ClaimedValue.text_span`` at its recorded
    ``position``. Iterates in reverse text position so earlier offsets stay
    valid while later spans are rewritten. The position is authoritative —
    it was computed against this exact ``text`` by ``validate``. When it
    does not line up the span is **skipped, never substring-replaced**: a
    blind ``str.replace`` could neutralise a *grounded* twin of the same
    number or corrupt a longer number the span is a substring of (e.g.
    ``"3.6"`` inside ``"13.62"``). Mirrors ``annotate_claims``'s
    conservative skip-on-drift. Callers that need a divergent surface
    redacted must re-validate that surface (positions are text-specific),
    not pass a foreign ``report``.

    Safe to call with an empty/clean report — returns the text unchanged.

    Returns ``(redacted_text, redacted_match_types)``; the redaction count
    is ``len(redacted_match_types)``.
    """
    flagged = [d for d in report.discrepancies if d.match_type in _REDACTABLE_MATCH_TYPES]
    if not flagged:
        return text, []

    flagged.sort(key=lambda d: d.claimed.position, reverse=True)
    result = text
    match_types: list[str] = []
    for d in flagged:
        span = d.claimed.text_span
        start = d.claimed.position
        end = start + len(span)
        if result[start:end] != span:
            continue
        result = result[:start] + marker + result[end:]
        match_types.append(d.match_type)
    return result, match_types


_LOG_LEVEL_ORDER = {"debug": 0, "info": 1, "warning": 2, "error": 3}


def log_validation_result(
    result: ValidationResult,
    agent_name: str = "unknown",
    sub_intent: str | None = None,
    log_level: str = "warning",
) -> None:
    """Log validation result as structured events.

    The log_level parameter controls the minimum severity that gets emitted:
    - "debug": log everything (valid results at debug, summaries at info)
    - "info": suppress debug, emit summaries at info and discrepancies at warning
    - "warning": only emit discrepancy warnings (default)
    - "error": suppress all validation logging
    """
    min_level = _LOG_LEVEL_ORDER.get(log_level, 2)

    log_kwargs: dict[str, Any] = {
        "agent": agent_name,
        "claims_found": len(result.claimed_values),
        "claims_verified": len(result.matches),
        "discrepancies": len(result.discrepancies),
        "trend_checks": len(result.trend_checks),
        "validation_time_ms": round(result.validation_time_ms, 1),
        "is_valid": result.is_valid,
    }
    if sub_intent:
        log_kwargs["sub_intent"] = sub_intent

    if result.is_valid:
        if min_level <= 0:  # debug
            logger.debug("response_validation_complete", **log_kwargs)
    else:
        if min_level <= 1:  # info
            logger.info("response_validation_complete", **log_kwargs)
        if min_level <= 2:  # warning
            for d in result.discrepancies:
                logger.warning(
                    "response_claim_discrepancy",
                    claimed_text=d.claimed.text_span,
                    claimed_value=d.claimed.parsed_value,
                    claimed_type=d.claimed.value_type,
                    nearest_actual=d.nearest_match.value if d.nearest_match else None,
                    nearest_metric=d.nearest_match.metric_name if d.nearest_match else None,
                    deviation=d.deviation,
                    match_type=d.match_type,
                    agent=agent_name,
                )


# ---------------------------------------------------------------------------
# Goal-attainment verifier (the goal-attainment design)
# ---------------------------------------------------------------------------
#
# Cross-checks planning-derived claims (target, capacity, headcount,
# projections, gaps) against the tenant's configured PlanningContext and
# the recorded compute_goal_attainment tool trace. Dispatches on the
# *presence* of planning-derived metrics in the response's claims, not on
# the routed sub_intent, so a misroute doesn't silently skip verification.

# Tolerance for float-valued claim comparisons. The model serialises floats
# through JSON; round-trip can land a 2.0 baseline on 2.0000000000001.
_FLOAT_REL_TOL = 1e-6
_FLOAT_ABS_TOL = 1e-9

# Maps from claim.metric → key in the replayed compute_goal_attainment
# result. Nested-band entries are tuples consumed by ``_replay_value``.
_REPLAY_METRIC_KEYS: dict[str, str | tuple[str, ...]] = {
    PROJECTED_FULL_YEAR_HIRES: "projected_full_year_hires",
    PROJECTED_FULL_YEAR_HIRES_LOWER: ("projection_confidence_band", "lower"),
    PROJECTED_FULL_YEAR_HIRES_UPPER: ("projection_confidence_band", "upper"),
    HIRING_GAP: "gap",
    CAPACITY_CONSTRAINED_MAX_HIRES: "capacity_constrained_max_hires",
    MONTHS_ELAPSED: "months_elapsed",
    MONTHS_REMAINING: "months_remaining",
}


def _values_equal(claim_value: float, configured: float | int) -> bool:
    """Comparator for planning-derived numeric claims.

    Counts (int) compare exactly; rates (float) compare with isclose.
    Pydantic Claim.value is typed as float; counts still arrive as whole
    numbers so ``==`` is clearer than isclose in logs for that case.
    """
    if isinstance(configured, int) and not isinstance(configured, bool):
        return claim_value == configured
    return math.isclose(claim_value, configured, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL)


def _replay_value(replayed: dict, key: str | tuple[str, ...]) -> Any:
    if isinstance(key, tuple):
        cursor: Any = replayed
        for k in key:
            cursor = cursor[k]
        return cursor
    return replayed[key]


def _multi_value(filter_value: object) -> list[str]:
    """Decode a ``Claim.filters[k]`` value into its literal alternatives."""
    if not filter_value or not isinstance(filter_value, str):
        return []
    return _split_pipe_alternatives(filter_value)


def goal_attainment_verifier(response: SpecialistResponse | None) -> None:
    """Cross-check planning-derived claims against ground truth.

    Stamps ``response.agent_error = True`` on any mismatch and emits a
    structured-log telemetry event. Mutates ``response`` in place; the
    caller (``goal_attainment_retry.evaluate``) reads ``response.agent_error``
    and the recorded outcomes on ``TurnPlanningState`` to decide
    Emit / Retry / Salvage.

    Pre-condition guards (all no-op + DEBUG log):
      - ``response is None`` or ``response.claims`` is None.
      - ``ToolContext`` unbound (verifier called outside a session).
      - ``TurnPlanningState`` unbound (verifier called outside a turn).
    """
    # Local imports defer heavy modules until the verifier actually runs —
    # response_validator is imported eagerly by session.py at startup, but
    # the planning surface is only exercised on goal-attainment turns.
    from tools.planning_tools import (
        compute_goal_attainment_kernel,
        peek_turn_planning_state,
    )
    from tools.tool_context import get_tool_context

    if response is None or getattr(response, "claims", None) is None:
        logger.debug("goal_attainment_verifier_skipped_no_response")
        return

    tool_context = get_tool_context()
    if tool_context is None:
        logger.debug("goal_attainment_verifier_skipped_no_tool_context")
        return

    turn_state = peek_turn_planning_state()
    if turn_state is None:
        logger.debug("goal_attainment_verifier_skipped_no_turn_state")
        return

    contexts = tool_context.planning_contexts
    default_year = tool_context.current_year_provider()
    recorded_calls = turn_state.recorded_goal_attainment_calls

    # Reset the outcomes list at the top of every verifier run so a prior
    # run's results don't leak into the next dispatch.
    turn_state.last_verifier_outcomes = []
    outcomes: list[tuple[str, str]] = []

    def mark(outcome: str, detail: str = "") -> None:
        outcomes.append((outcome, detail))
        logger.debug(
            "goal_attainment_verifier_mismatch",
            outcome=outcome,
            detail=detail,
        )

    def context_for(claim: Any) -> PlanningContext | None:
        raw = claim.filters.get("year")
        if raw is None:
            return contexts.get(default_year)
        try:
            year = int(raw)
        except (TypeError, ValueError):
            return None
        return contexts.get(year)

    planning_metrics_present = any(c.metric in PLANNING_METRICS for c in response.claims)
    goal_attainment_invoked = bool(recorded_calls)
    if not planning_metrics_present and not goal_attainment_invoked:
        return  # nothing to verify; no telemetry emitted

    # (0) Claims-presence guard.
    if goal_attainment_invoked and not planning_metrics_present:
        mark(
            "claims_missing",
            "compute_goal_attainment was called this turn but the response emitted no planning-derived claims",
        )

    # (1) Value cross-checks against the year-specific PlanningContext.
    for claim in response.claims:
        context = context_for(claim)

        if claim.metric == HIRING_TARGET:
            if context is None:
                mark(
                    "unconfigured",
                    f"claimed hiring_target for year={claim.filters.get('year')!r} but no PlanningContext for that year",
                )
                continue
            role = claim.filters.get("role", "").lower()
            match = next(
                (t for t in context.targets if t.role_label.lower() == role),
                None,
            )
            if match is None:
                mark(
                    "target_role_unconfigured",
                    f"claimed target for role={role!r} but configured roles are {[t.role_label for t in context.targets]}",
                )
            elif not _values_equal(claim.value, match.count):
                mark(
                    "target_mismatch",
                    f"claimed {claim.value}, configured {match.count}",
                )

        elif claim.metric == RECRUITER_CAPACITY_PER_MONTH:
            if context and context.recruiter_capacity_per_month is not None:
                if not _values_equal(claim.value, context.recruiter_capacity_per_month):
                    mark(
                        "capacity_mismatch",
                        f"claimed {claim.value}, configured {context.recruiter_capacity_per_month}",
                    )

        elif claim.metric == ACTIVE_RECRUITERS:
            if context and context.active_recruiters is not None:
                if not _values_equal(claim.value, context.active_recruiters):
                    mark(
                        "headcount_mismatch",
                        f"claimed {claim.value}, configured {context.active_recruiters}",
                    )

    # (2) Replay compute_goal_attainment via the pure kernel. Catches
    #     fabricated inputs and determinism breaks.
    invocation_by_sqid = {inv.source_query_id: inv for inv in recorded_calls}

    for invocation in recorded_calls:
        replayed = compute_goal_attainment_kernel(**invocation.args)
        recorded_kernel_result = {
            k: v for k, v in invocation.result.items() if k != "source_query_id"
        }
        if replayed != recorded_kernel_result:
            mark(
                "replay_mismatch",
                f"kernel is non-deterministic OR record was tampered with (sqid={invocation.source_query_id})",
            )

    referenced_sqids: set[str] = set()
    for claim in response.claims:
        sqid = claim.source_query_id or ""
        if not sqid.startswith(SQID_PREFIX_COMPUTE_GOAL_ATTAINMENT):
            continue
        invocation = invocation_by_sqid.get(sqid)
        if invocation is None:
            mark(
                "claim_unknown_source",
                f"claim references source_query_id={sqid!r} but no compute_goal_attainment call this turn produced it",
            )
            continue
        referenced_sqids.add(sqid)
        key = _REPLAY_METRIC_KEYS.get(claim.metric)
        if key is None:
            continue
        recorded = _replay_value(invocation.result, key)
        if not _values_equal(claim.value, recorded):
            mark(
                "claim_replay_mismatch",
                f"metric={claim.metric}: claim={claim.value}, recorded={recorded} (sqid={sqid})",
            )

    # (2b) Input cross-check on actual_ytd_hires, scoped per (role, year).
    # Multi-scenario goal-attainment turns ("what about engineers AND
    # designers?") legitimately emit multiple compute calls. Pair each
    # ytd claim with the compute invocation(s) sharing its (role, year);
    # only flag multi-reference when the SAME scenario references >1 call.
    def _scenario_for(claim: Any) -> tuple[str, int | None]:
        role = claim.filters.get("role", "").lower()
        raw_year = claim.filters.get("year")
        try:
            year_int = int(raw_year) if raw_year is not None else None
        except (TypeError, ValueError):
            year_int = None
        return (role, year_int)

    # Build sqid → {scenario, ...} from the claims that reference each
    # compute invocation. A well-formed response references each sqid
    # from exactly one scenario; defensive ``set`` tolerates outliers.
    sqid_scenarios: dict[str, set[tuple[str, int | None]]] = {}
    for claim in response.claims:
        sqid = claim.source_query_id or ""
        if sqid in invocation_by_sqid:
            sqid_scenarios.setdefault(sqid, set()).add(_scenario_for(claim))

    ytd_claims = [c for c in response.claims if c.metric == ACTUAL_YTD_HIRES]
    for ytd_claim in ytd_claims:
        scenario = _scenario_for(ytd_claim)
        co_scenario_sqids = {
            sqid for sqid, scenarios in sqid_scenarios.items() if scenario in scenarios
        }
        if not co_scenario_sqids:
            continue  # ytd claim not tied to any compute scenario
        if len(co_scenario_sqids) > 1:
            mark(
                "compute_input_multi_reference",
                f"actual_ytd_hires claim for scenario={scenario} references "
                f"{len(co_scenario_sqids)} compute invocations "
                f"(sqids={sorted(co_scenario_sqids)}); abandon stale "
                "references on retry per the critique",
            )
            continue
        (sqid,) = co_scenario_sqids
        invocation = invocation_by_sqid[sqid]
        recorded_ytd = invocation.args.get("actual_ytd_hires")
        if recorded_ytd is not None and not _values_equal(ytd_claim.value, recorded_ytd):
            mark(
                "compute_input_mismatch",
                f"compute_goal_attainment(sqid={sqid}) was called with "
                f"actual_ytd_hires={recorded_ytd} but the actual_ytd_hires "
                f"claim reports value={ytd_claim.value}",
            )

    # (2c) Input cross-check: the target/capacity ARGS passed to
    #      compute_goal_attainment must match the configured PlanningContext.
    #      The output-only synthesized claims (projection/gap/ceiling) replay
    #      cleanly regardless of what inputs were fed in, so without this a
    #      model that substitutes a generic benchmark capacity (or a
    #      user-stated target) for the configured value sails through as
    #      "match". Capacity is year-less (overlaid on every context); the
    #      target is matched per (role, year) only when a claim pins the
    #      invocation to exactly one configured scenario — otherwise the role
    #      is ambiguous and we skip rather than risk a false positive.
    for invocation in recorded_calls:
        sqid = invocation.source_query_id
        scenarios = sqid_scenarios.get(sqid, set())
        years = {y for _role, y in scenarios if y is not None}
        inv_year = next(iter(years)) if len(years) == 1 else default_year
        context = contexts.get(inv_year)
        if context is None:
            continue

        recorded_cap = invocation.args.get("capacity_per_recruiter_per_month")
        if (
            recorded_cap is not None
            and context.recruiter_capacity_per_month is not None
            and not _values_equal(recorded_cap, context.recruiter_capacity_per_month)
        ):
            mark(
                "compute_capacity_input_mismatch",
                f"compute_goal_attainment(sqid={sqid}) was called with "
                f"capacity_per_recruiter_per_month={recorded_cap} but the configured "
                f"baseline is {context.recruiter_capacity_per_month}",
            )

        recorded_headcount = invocation.args.get("active_recruiters")
        if (
            recorded_headcount is not None
            and context.active_recruiters is not None
            and not _values_equal(recorded_headcount, context.active_recruiters)
        ):
            mark(
                "compute_headcount_input_mismatch",
                f"compute_goal_attainment(sqid={sqid}) was called with "
                f"active_recruiters={recorded_headcount} but the configured baseline "
                f"is {context.active_recruiters}",
            )

        roles = {role for role, _y in scenarios if role}
        if len(roles) == 1 and len(years) <= 1:
            role = next(iter(roles))
            target = next(
                (t for t in context.targets if t.role_label.lower() == role),
                None,
            )
            recorded_target = invocation.args.get("target")
            if (
                target is not None
                and recorded_target is not None
                and not _values_equal(recorded_target, target.count)
            ):
                mark(
                    "compute_target_input_mismatch",
                    f"compute_goal_attainment(sqid={sqid}) was called with "
                    f"target={recorded_target} but the configured target for "
                    f"role={role!r} (year={inv_year}) is {target.count}",
                )

    # (3) Filter intersection: actual_ytd query's job_name filter must
    #     intersect HiringTarget.job_name_filter.
    for claim in response.claims:
        if claim.metric != ACTUAL_YTD_HIRES:
            continue
        context = context_for(claim)
        if context is None:
            continue
        role = claim.filters.get("role", "").lower()
        target = next(
            (t for t in context.targets if t.role_label.lower() == role),
            None,
        )
        if target is None or not target.job_name_filter:
            continue
        applied_list = _multi_value(claim.filters.get("job_name", ""))
        if not applied_list:
            continue  # no job_name filter recorded — covered by other checks
        if not set(applied_list) & set(target.job_name_filter):
            mark(
                "filter_mismatch",
                f"actual_ytd query used job_name={applied_list} but target.job_name_filter={target.job_name_filter}",
            )

    # Finalise.
    if outcomes:
        response.agent_error = True

    turn_state.last_verifier_outcomes = list(outcomes)

    logger.info(
        "goal_attainment_verifier_outcome",
        # ``agent`` disambiguates which specialist's response produced this
        # outcome. Without it, two events in one turn (e.g. a first-pass
        # ``claims_missing`` then a synthesis re-verify ``match``) read as two
        # different agents — the misattribution behind an earlier
        # prose-grounding incident.
        agent=response.agent_name,
        verifier_outcomes=[o for o, _ in outcomes] if outcomes else ["match"],
        invocations=len(recorded_calls),
        claims_with_planning_metrics=sum(
            1 for c in response.claims if c.metric in PLANNING_METRICS
        ),
    )
