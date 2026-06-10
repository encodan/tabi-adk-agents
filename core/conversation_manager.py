"""
Conversation state management for Gemini agents.

Provides:
- Multi-turn conversation history tracking
- Context window management to prevent overflow
- Conversation persistence (optional)
- Turn counting and limits
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Approximate token limits for Gemini models
MODEL_CONTEXT_LIMITS = {
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,  # 2M tokens
    "gemini-1.5-flash": 1_000_000,
    "gemini-3-flash-preview": 1_000_000,
    "gemini-3.1-flash-lite-preview": 1_000_000,
    "gemini-3.1-flash-lite": 1_000_000,
    "gemini-3.5-flash": 1_000_000,
}

# Rough estimate: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user", "model", or "function"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // CHARS_PER_TOKEN

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_estimate=data.get("token_estimate", 0),
            metadata=data.get("metadata", {}),
        )


class ConversationManager:
    """
    Manages conversation state for multi-turn Gemini interactions.

    Features:
    - Automatic context window management
    - Turn limiting to prevent runaway conversations
    - Optional conversation persistence
    - Token usage estimation

    Usage:
        manager = ConversationManager(max_turns=10, model="gemini-3.1-flash-lite")

        # Add turns
        manager.add_user_turn("What's our hire rate?")
        manager.add_model_turn("Based on the data, your hire rate is 12%...")
        manager.add_function_turn("query_metrics", {"hire_rate": 12.5})

        # Get history for Gemini
        history = manager.get_history_for_api()

        # Check limits
        if manager.is_context_full():
            manager.prune_oldest_turns(keep_recent=5)
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens: int | None = None,
        model: str = "gemini-3.1-flash-lite",
        system_prompt: str | None = None,
        conversation_id: str | None = None,
    ):
        """
        Initialize conversation manager.

        Args:
            max_turns: Maximum number of turns before warning/pruning
            max_tokens: Maximum estimated tokens (defaults to 80% of model limit)
            model: Model name for context limit lookup
            system_prompt: Optional system prompt (not counted toward history)
            conversation_id: Optional ID for persistence
        """
        self.max_turns = max_turns
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_id = conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set token limit to 80% of model's context window
        model_limit = MODEL_CONTEXT_LIMITS.get(model, 100_000)
        self.max_tokens = max_tokens or int(model_limit * 0.8)

        self._turns: list[ConversationTurn] = []
        self._created_at = datetime.now()

    @property
    def turn_count(self) -> int:
        """Number of turns in conversation."""
        return len(self._turns)

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens in conversation history."""
        system_tokens = len(self.system_prompt) // CHARS_PER_TOKEN if self.system_prompt else 0
        history_tokens = sum(turn.token_estimate for turn in self._turns)
        return system_tokens + history_tokens

    @property
    def is_empty(self) -> bool:
        """Check if conversation has no turns."""
        return len(self._turns) == 0

    def add_user_turn(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> ConversationTurn:
        """Add a user message to the conversation."""
        turn = ConversationTurn(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        self._turns.append(turn)
        self._check_limits()
        return turn

    def add_model_turn(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> ConversationTurn:
        """Add a model response to the conversation."""
        turn = ConversationTurn(
            role="model",
            content=content,
            metadata=metadata or {},
        )
        self._turns.append(turn)
        self._check_limits()
        return turn

    def add_function_turn(
        self,
        function_name: str,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Add a function call result to the conversation."""
        content = json.dumps({"function": function_name, "result": result}, default=str)
        turn = ConversationTurn(
            role="function",
            content=content,
            metadata={"function_name": function_name, **(metadata or {})},
        )
        self._turns.append(turn)
        self._check_limits()
        return turn

    def _check_limits(self) -> None:
        """Check and warn about approaching limits."""
        if self.turn_count >= self.max_turns:
            logger.warning(
                "Conversation reached turn limit: %d/%d turns",
                self.turn_count,
                self.max_turns,
            )

        if self.total_tokens >= self.max_tokens * 0.9:
            logger.warning(
                "Conversation approaching token limit: ~%d/%d tokens (%.1f%%)",
                self.total_tokens,
                self.max_tokens,
                (self.total_tokens / self.max_tokens) * 100,
            )

    def is_turn_limit_reached(self) -> bool:
        """Check if turn limit has been reached."""
        return self.turn_count >= self.max_turns

    def is_context_full(self) -> bool:
        """Check if context window is approaching capacity."""
        return self.total_tokens >= self.max_tokens * 0.9

    def get_history(self) -> list[ConversationTurn]:
        """Get all conversation turns."""
        return list(self._turns)

    def get_history_for_api(self) -> list[dict[str, str]]:
        """
        Get conversation history in format suitable for Gemini API.

        Returns list of {"role": ..., "content": ...} dicts.
        """
        return [{"role": turn.role, "content": turn.content} for turn in self._turns]

    def get_last_n_turns(self, n: int) -> list[ConversationTurn]:
        """Get the last N turns from the conversation."""
        return self._turns[-n:] if n > 0 else []

    def prune_oldest_turns(self, keep_recent: int = 5) -> int:
        """
        Remove oldest turns, keeping the most recent ones.

        Args:
            keep_recent: Number of recent turns to keep

        Returns:
            Number of turns removed
        """
        if len(self._turns) <= keep_recent:
            return 0

        removed_count = len(self._turns) - keep_recent
        self._turns = self._turns[-keep_recent:]

        logger.info(
            "Pruned %d turns from conversation, keeping %d recent turns",
            removed_count,
            keep_recent,
        )
        return removed_count

    def summarize_and_prune(self, summary: str, keep_recent: int = 3) -> None:
        """
        Replace old history with a summary, keeping recent turns.

        Args:
            summary: Summary of the conversation so far
            keep_recent: Number of recent turns to keep after summary
        """
        recent_turns = self._turns[-keep_recent:] if keep_recent > 0 else []

        # Create a summary turn
        summary_turn = ConversationTurn(
            role="model",
            content=f"[Conversation Summary]\n{summary}",
            metadata={"is_summary": True, "summarized_turns": len(self._turns) - keep_recent},
        )

        self._turns = [summary_turn] + recent_turns

        logger.info(
            "Summarized conversation: replaced %d turns with summary + %d recent turns",
            len(self._turns) - keep_recent - 1,
            keep_recent,
        )

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns = []

    def get_stats(self) -> dict[str, Any]:
        """Get conversation statistics."""
        return {
            "conversation_id": self.conversation_id,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "turns_remaining": max(0, self.max_turns - self.turn_count),
            "token_usage_percent": (self.total_tokens / self.max_tokens) * 100,
            "created_at": self._created_at.isoformat(),
            "model": self.model,
        }

    def save_to_file(self, filepath: str | Path) -> None:
        """
        Save conversation to a JSON file.

        Args:
            filepath: Path to save the conversation
        """
        filepath = Path(filepath)
        data = {
            "conversation_id": self.conversation_id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "created_at": self._created_at.isoformat(),
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "turns": [turn.to_dict() for turn in self._turns],
        }
        filepath.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved conversation to %s", filepath)

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> "ConversationManager":
        """
        Load conversation from a JSON file.

        Args:
            filepath: Path to load the conversation from

        Returns:
            Loaded ConversationManager instance
        """
        filepath = Path(filepath)
        data = json.loads(filepath.read_text())

        manager = cls(
            max_turns=data.get("max_turns", 20),
            max_tokens=data.get("max_tokens"),
            model=data.get("model", "gemini-3.1-flash-lite"),
            system_prompt=data.get("system_prompt"),
            conversation_id=data.get("conversation_id"),
        )
        manager._created_at = datetime.fromisoformat(data["created_at"])
        manager._turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]

        logger.info("Loaded conversation from %s (%d turns)", filepath, len(manager._turns))
        return manager

    def __len__(self) -> int:
        """Return number of turns."""
        return self.turn_count

    def __repr__(self) -> str:
        return (
            f"ConversationManager(id={self.conversation_id!r}, "
            f"turns={self.turn_count}, tokens=~{self.total_tokens})"
        )
