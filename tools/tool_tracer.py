"""Tool trace capture for live-agent evaluation.

`capture_tool_trace()` is a context manager that binds a `ToolTrace` to the
current `ContextVar`. Tool entry points in `adk_tools.py` append `ToolCall`
records to the active trace (if any) so evaluators can inspect the trajectory
and query payloads without coupling the agent code to the eval harness.

Outside the context manager the ContextVar is None, so the tool entry points
no-op and pay only a single `ContextVar.get()` per call.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

from evaluation.evaluators import ToolCall

F = TypeVar("F", bound=Callable[..., Any])


class ToolTrace:
    """Accumulates tool calls made during a traced block.

    Holds `ToolCall` objects (the shape evaluators consume) and the raw result
    payloads from metric-query tools (needed by `eval_factuality` and
    `eval_query_plan`). Results are kept on a parallel list rather than on
    `ToolCall` itself so the evaluator's public shape doesn't change.
    """

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self._results: list[Any] = []

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        duration_ms: float | None = None,
        result: Any = None,
        result_summary: str | None = None,
    ) -> None:
        self.calls.append(
            ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                timestamp=time.time(),
                duration_ms=duration_ms,
                result_summary=result_summary,
            )
        )
        self._results.append(result)

    def to_tool_calls(self) -> list[ToolCall]:
        """Return a shallow copy of the accumulated tool calls."""
        return list(self.calls)

    def call_results(self) -> list[tuple[str, Any]]:
        """``(tool_name, raw_result)`` pairs in call order.

        The ADK ``hallucinations_v1`` judge grounds the response against the
        Invocation's ``tool_responses`` (its ``{context}``); the bridge maps
        these pairs to ``genai.types.FunctionResponse`` so groundedness is
        scored against the data the agent actually retrieved. Kept genai-free
        here so the tracer stays uncoupled from the eval harness."""
        return [(call.tool_name, result) for call, result in zip(self.calls, self._results)]

    def to_query_results(self) -> list[dict[str, Any]]:
        """Extract metric-query rows for `eval_factuality`.

        Only `query_recruitment_metrics` and `query_multiple_recruitment_metrics`
        produce data rows the factuality evaluator can score against. Others
        (chart generation, handoffs, knowledge tools) have no data payload.
        """
        flat: list[dict[str, Any]] = []
        for call, result in zip(self.calls, self._results):
            if not isinstance(result, dict) or not result.get("success"):
                continue
            if call.tool_name == "query_recruitment_metrics":
                flat.extend(row for row in (result.get("data") or []) if isinstance(row, dict))
            elif call.tool_name == "query_multiple_recruitment_metrics":
                for sub in result.get("results") or []:
                    if isinstance(sub, dict) and sub.get("success"):
                        flat.extend(row for row in (sub.get("data") or []) if isinstance(row, dict))
        return flat

    def executed_queries(self) -> list[dict[str, Any]]:
        """Return the request payloads sent to the metric query tools.

        `eval_query_plan` uses this to verify the plan asked for the right
        metrics and filters — it looks at arguments, not results.
        """
        executed: list[dict[str, Any]] = []
        for call in self.calls:
            if call.tool_name == "query_recruitment_metrics":
                executed.append(dict(call.arguments))
            elif call.tool_name == "query_multiple_recruitment_metrics":
                for sub in call.arguments.get("queries") or []:
                    if isinstance(sub, dict):
                        executed.append(dict(sub))
        return executed


_trace_var: contextvars.ContextVar[ToolTrace | None] = contextvars.ContextVar(
    "tabi_tool_trace", default=None
)


@contextmanager
def capture_tool_trace() -> Iterator[ToolTrace]:
    """Bind a fresh `ToolTrace` for the duration of the `with` block.

    Nested usage is supported: the inner context shadows the outer, and the
    outer is restored on exit. Most call sites use one trace per question.
    """
    trace = ToolTrace()
    token = _trace_var.set(trace)
    try:
        yield trace
    finally:
        _trace_var.reset(token)


def get_active_trace() -> ToolTrace | None:
    """Return the active trace if one is bound, else None."""
    return _trace_var.get()


def record_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    duration_ms: float | None = None,
    result: Any = None,
    result_summary: str | None = None,
) -> None:
    """Append a call to the active trace if one is bound; else no-op.

    This is the hook tool entry points call. Outside a `capture_tool_trace()`
    block the cost is one `ContextVar.get()`.
    """
    trace = _trace_var.get()
    if trace is None:
        return
    trace.record(
        tool_name,
        arguments,
        duration_ms=duration_ms,
        result=result,
        result_summary=result_summary,
    )


def trace_tool(tool_name: str | None = None) -> Callable[[F], F]:
    """Decorator that records the decorated tool call into the active trace.

    Use on ADK tool entry points in modules outside `adk_tools.py`
    (statistical, knowledge) where inline wiring would be noisy. The tool
    name defaults to the wrapped function's `__name__`.

    Works on sync and async functions. Adds one `ContextVar.get()` outside
    a `capture_tool_trace()` block — no other overhead.
    """

    def decorator(fn: F) -> F:
        name = tool_name or fn.__name__
        is_coro = asyncio.iscoroutinefunction(fn)
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None

        def _build_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            if signature is None:
                return dict(kwargs)
            try:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                return dict(bound.arguments)
            except TypeError:
                return dict(kwargs)

        if is_coro:

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if _trace_var.get() is None:
                    return await fn(*args, **kwargs)
                start = time.perf_counter()
                captured_args = _build_args(args, kwargs)
                result = await fn(*args, **kwargs)
                record_tool_call(
                    name,
                    captured_args,
                    duration_ms=(time.perf_counter() - start) * 1000.0,
                    result=result,
                )
                return result

            return cast(F, async_wrapper)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if _trace_var.get() is None:
                return fn(*args, **kwargs)
            start = time.perf_counter()
            captured_args = _build_args(args, kwargs)
            result = fn(*args, **kwargs)
            record_tool_call(
                name,
                captured_args,
                duration_ms=(time.perf_counter() - start) * 1000.0,
                result=result,
            )
            return result

        return cast(F, sync_wrapper)

    return decorator


__all__ = [
    "ToolTrace",
    "capture_tool_trace",
    "get_active_trace",
    "record_tool_call",
    "trace_tool",
]
