"""
Base utilities for ADK agents.

Provides shared context and configuration for all agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import get_config
from core.conversation_manager import ConversationManager, ConversationTurn
from core.usage_tracker import GeminiUsageTracker


@dataclass
class AgentContext:
    """
    Context for agent execution.

    Holds tenant information, conversation state, and usage tracking.
    Passed to agents to provide shared state across the multi-agent system.

    Attributes:
        customer_id: Tenant identifier for multi-tenant isolation
        conversation_id: Unique identifier for this conversation session
        model: Default model name for agents
        conversation: ConversationManager instance for turn tracking
        tracker: GeminiUsageTracker for token/cost monitoring
        metadata: Additional context (e.g., user info, request ID)
    """

    customer_id: str
    conversation_id: str = field(
        default_factory=lambda: f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    model: str = field(default_factory=lambda: get_config().model.default_model)
    conversation: ConversationManager = field(default=None)  # type: ignore[assignment]
    tracker: GeminiUsageTracker = field(default_factory=GeminiUsageTracker)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize conversation manager if not provided."""
        if self.conversation is None:
            self.conversation = ConversationManager(
                conversation_id=self.conversation_id,
                model=self.model,
            )

    def get_stats(self) -> dict[str, Any]:
        """
        Get combined statistics for conversation and usage.

        Returns:
            Dictionary with conversation and usage summaries
        """
        return {
            "customer_id": self.customer_id,
            "conversation_id": self.conversation_id,
            "model": self.model,
            "conversation": self.conversation.get_stats(),
            "usage": self.tracker.get_summary(),
        }

    def log_user_turn(self, content: str) -> ConversationTurn:
        """Log a user turn to the conversation and return the created turn.

        The turn's metadata is an independent copy of ``self.metadata`` so
        callers can mutate it (e.g. to attach routing entities) without
        retroactively altering earlier turns that share this context.
        """
        return self.conversation.add_user_turn(content, metadata=self.metadata.copy())

    def log_model_turn(self, content: str, agent_name: str | None = None) -> None:
        """Log a model turn to the conversation."""
        meta = {**self.metadata}
        if agent_name:
            meta["agent"] = agent_name
        self.conversation.add_model_turn(content, metadata=meta)

    def log_function_turn(
        self,
        function_name: str,
        result: Any,
        agent_name: str | None = None,
    ) -> None:
        """Log a function call turn to the conversation."""
        meta = {**self.metadata}
        if agent_name:
            meta["agent"] = agent_name
        self.conversation.add_function_turn(function_name, result, metadata=meta)
