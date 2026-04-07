"""
Short-term session memory for conversation continuity.
"""

from backend.utils.config import SHORT_TERM_LIMIT


class SessionMemory:
    """
    Maintains the last N conversation exchanges for a single session.

    Attributes:
        exchanges: List of {role, content} dicts.
        limit: Maximum number of user-assistant pairs to keep.
    """

    def __init__(self, limit: int = SHORT_TERM_LIMIT):
        self.exchanges: list[dict] = []
        self.limit = limit

    def add(self, role: str, content: str) -> None:
        """Append a role/content pair and trim to the limit."""
        self.exchanges.append({"role": role, "content": content})
        if len(self.exchanges) > self.limit * 2:
            self.exchanges = self.exchanges[-self.limit * 2:]

    def recent(self, n: int = 6) -> list[dict]:
        """Return the last n exchanges."""
        return self.exchanges[-(n * 2):]

    def as_text(self) -> str:
        """Format all exchanges as a readable transcript."""
        return "\n".join(f"{e['role'].upper()}: {e['content']}" for e in self.exchanges)

    def recent_as_text(self, n: int = 6) -> str:
        """Format the last n exchanges as text."""
        return "\n".join(f"{e['role'].upper()}: {e['content']}" for e in self.recent(n))

    def clear(self) -> None:
        """Clear all exchanges."""
        self.exchanges.clear()
