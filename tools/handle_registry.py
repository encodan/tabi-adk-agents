"""Per-turn registry of query-result handles.

Handles let the LLM reference an in-process row set by ID instead of
re-serializing rows through the tool-call payload. Lifecycle is bound
to the current turn via ContextVar, alongside the other per-turn
ContextVars in :meth:`AgentSession._bind_turn_scope`.

Eviction is FIFO — see ``register`` docstring for the rationale (a duplicate
registration must not displace older handles that the model may still
reference within the same turn).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

FieldType = Literal["temporal", "quantitative", "nominal"]
ClassifyFieldFn = Callable[[list[dict[str, Any]], str], FieldType]


class QueryResultHandle(BaseModel):
    """A registry entry — the rows plus enough metadata to render a chart.

    ``field_types`` is renamed from ``schema`` to avoid shadowing Pydantic
    v1's deprecated ``BaseModel.schema()`` method (Pydantic v2 emits a
    deprecation warning on collisions).
    """

    handle_id: str
    rows: list[dict[str, Any]]
    field_types: dict[str, FieldType]
    query_intent: str
    metric: str | None = None
    inserted_at: float


class TurnHandleRegistry:
    """In-memory, turn-scoped store mapping ``handle_id`` → ``QueryResultHandle``.

    Storage is purely in-memory and turn-scoped; identical content produces
    identical handles (dedup), so the model cannot accidentally fragment the
    registry by re-querying the same data.
    """

    DEFAULT_MAX_HANDLES = 20

    def __init__(self, max_handles: int = DEFAULT_MAX_HANDLES) -> None:
        self._handles: OrderedDict[str, QueryResultHandle] = OrderedDict()
        self._max_handles = max_handles

    def register(
        self,
        rows: list[dict[str, Any]],
        query_intent: str,
        classify_field: ClassifyFieldFn,
        metric: str | None = None,
        cache_key: str | None = None,
    ) -> str:
        """Return a stable ``handle_id``. Same rows + intent ⇒ same handle.

        ``classify_field(rows, field_name) -> FieldType`` is injected to avoid
        importing the visualization service here (circular import). Callers
        pass the module-level ``classify_field`` helper (in the public showcase
        this is the lightweight stub in ``tools.adk_tools``).

        Dedup hits return without ``move_to_end``, so eviction is strictly
        FIFO by first-insertion time, not LRU. Intentional — a re-query for
        the same data should not displace older handles that the model may
        still reference within the turn.
        """
        handle_id = cache_key or self._content_address(rows, query_intent, metric)
        if handle_id in self._handles:
            return handle_id
        if len(self._handles) >= self._max_handles:
            evicted_id, _ = self._handles.popitem(last=False)
            logger.warning("handle_registry.evicted", handle_id=evicted_id)
        field_types: dict[str, FieldType] = (
            {field: classify_field(rows, field) for field in rows[0].keys()} if rows else {}
        )
        self._handles[handle_id] = QueryResultHandle(
            handle_id=handle_id,
            rows=rows,
            field_types=field_types,
            query_intent=query_intent,
            metric=metric,
            inserted_at=time.time(),
        )
        return handle_id

    def resolve(self, handle_id: str) -> QueryResultHandle | None:
        return self._handles.get(handle_id)

    def list_ids(self) -> list[str]:
        return list(self._handles.keys())

    @staticmethod
    def _content_address(rows: list[dict[str, Any]], query_intent: str, metric: str | None) -> str:
        payload = json.dumps(
            {"rows": rows, "intent": query_intent, "metric": metric},
            sort_keys=True,
            default=str,
        )
        return "qr_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


_turn_handle_registry: contextvars.ContextVar[TurnHandleRegistry | None] = contextvars.ContextVar(
    "_turn_handle_registry", default=None
)


def set_turn_handle_registry() -> contextvars.Token[TurnHandleRegistry | None]:
    """Bind a fresh registry for the current turn.

    Returns a token to pass to :func:`reset_turn_handle_registry` at turn end.
    Bound in :meth:`AgentSession._bind_turn_scope` alongside the other
    per-turn ContextVars.
    """
    return _turn_handle_registry.set(TurnHandleRegistry())


def reset_turn_handle_registry(
    token: contextvars.Token[TurnHandleRegistry | None],
) -> None:
    _turn_handle_registry.reset(token)


def get_turn_handle_registry() -> TurnHandleRegistry | None:
    return _turn_handle_registry.get()
