"""
Gemini API usage tracking for cost visibility.

Tracks:
- Input tokens (prompt_token_count)
- Output tokens (candidates_token_count)
- Total tokens
- Request counts by model
- Latency and routing decisions (for performance analysis)
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Gemini pricing per 1M tokens.
# Illustrative rates; see provider pricing for authoritative figures.
GEMINI_PRICING = {
    "gemini-3.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "thinking": 0.30,  # Thinking tokens billed at output rate
    },
}


@dataclass
class UsageRecord:
    """Record of a single Gemini API call's token usage."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    question: str | None = None
    latency_ms: int | None = None
    routing_decision: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    thinking_tokens: int = 0
    thinking_budget: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost in USD based on model pricing."""
        pricing = GEMINI_PRICING.get(self.model)
        if not pricing:
            return 0.0

        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        thinking_cost = (
            (self.thinking_tokens / 1_000_000) * pricing["thinking"]
            if self.thinking_tokens and "thinking" in pricing
            else 0.0
        )
        return input_cost + output_cost + thinking_cost


class GeminiUsageTracker:
    """
    Tracks token usage across Gemini API calls.

    Thread-safe for use in async contexts.

    Usage:
        tracker = GeminiUsageTracker()

        response = client.models.generate_content(...)
        tracker.record_usage(
            response,
            "gemini-3.1-flash-lite",
            question="What's the hire rate?",
            latency_ms=1234,
            routing_decision="fast_route:pipeline_analyst",
        )

        # At end of session
        summary = tracker.get_summary()
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"Estimated cost: ${summary['estimated_cost_usd']:.4f}")
    """

    def __init__(self):
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    def record_usage(
        self,
        response: Any,
        model: str,
        question: str | None = None,
        latency_ms: int | None = None,
        routing_decision: str | None = None,
        tool_calls: list[str] | None = None,
    ) -> UsageRecord | None:
        """
        Extract and record usage from a Gemini response.

        Args:
            response: Gemini API response object
            model: Model name (e.g., "gemini-3.1-flash-lite")
            question: Optional question/prompt for context
            latency_ms: Time taken for the API call in milliseconds
            routing_decision: How the query was routed (e.g., "fast_route:pipeline_analyst")
            tool_calls: List of tool/function names called during the interaction

        Returns:
            UsageRecord if usage data was found, None otherwise
        """
        if not hasattr(response, "usage_metadata") or not response.usage_metadata:
            logger.debug("No usage_metadata in response")
            return None

        usage = response.usage_metadata
        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        record = UsageRecord(
            model=model,
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=usage.candidates_token_count or 0,
            total_tokens=usage.total_token_count or 0,
            timestamp=datetime.now(),
            question=question[:100] if question else None,  # Truncate for storage
            latency_ms=latency_ms,
            routing_decision=routing_decision,
            tool_calls=tool_calls or [],
            thinking_tokens=thinking_tokens,
        )

        with self._lock:
            self._records.append(record)

        # Log structured Gemini event with event_type for filtering
        logger.info(
            "gemini_api_call",
            event_type="gemini",
            model=model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            thinking_tokens=record.thinking_tokens,
            total_tokens=record.total_tokens,
            estimated_cost_usd=round(record.estimated_cost_usd, 6),
            latency_ms=latency_ms,
            routing_decision=routing_decision,
            tool_calls=tool_calls or [],
            question_preview=question[:100] if question else None,
        )

        return record

    def get_summary(self) -> dict[str, Any]:
        """
        Get aggregated usage summary.

        Returns:
            Dictionary with:
                - total_requests: Number of API calls
                - total_input_tokens: Sum of input tokens
                - total_output_tokens: Sum of output tokens
                - total_tokens: Sum of all tokens
                - estimated_cost_usd: Estimated cost in USD
                - by_model: Breakdown by model name
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "by_model": {},
            }

        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.estimated_cost_usd for r in records)

        # Group by model
        by_model: dict[str, dict] = {}
        for record in records:
            if record.model not in by_model:
                by_model[record.model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            by_model[record.model]["requests"] += 1
            by_model[record.model]["input_tokens"] += record.input_tokens
            by_model[record.model]["output_tokens"] += record.output_tokens
            by_model[record.model]["total_tokens"] += record.total_tokens
            by_model[record.model]["estimated_cost_usd"] += record.estimated_cost_usd

        return {
            "total_requests": len(records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
            "by_model": by_model,
        }

    def get_session_usage(self) -> list[UsageRecord]:
        """Get all usage records for current session."""
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        """Clear all usage records."""
        with self._lock:
            self._records.clear()

    def format_summary(self) -> str:
        """Format usage summary as a human-readable string."""
        summary = self.get_summary()

        if summary["total_requests"] == 0:
            return "No Gemini API calls recorded."

        lines = [
            "",
            "=" * 50,
            "Gemini API Usage Summary",
            "=" * 50,
            f"Total requests:      {summary['total_requests']}",
            f"Input tokens:        {summary['total_input_tokens']:,}",
            f"Output tokens:       {summary['total_output_tokens']:,}",
            f"Total tokens:        {summary['total_tokens']:,}",
            f"Estimated cost:      ${summary['estimated_cost_usd']:.4f}",
        ]

        if summary["by_model"]:
            lines.append("")
            lines.append("By Model:")
            for model, stats in summary["by_model"].items():
                lines.append(f"  {model}:")
                lines.append(f"    Requests: {stats['requests']}")
                lines.append(f"    Tokens:   {stats['total_tokens']:,}")
                lines.append(f"    Cost:     ${stats['estimated_cost_usd']:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)
