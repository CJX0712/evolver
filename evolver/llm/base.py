"""The model interface.

Evolver never talks to a vendor SDK directly -- it talks to this one method.
That keeps the engine testable (swap in a deterministic replay adapter), keeps
token accounting in a single place, and means switching providers is a config
change rather than a refactor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evolver.core.config import LLMConfig


@dataclass
class LLMResponse:
    """One completion, with the cost of obtaining it."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw: Any = None
    error: Optional[str] = None

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMAdapter(ABC):
    """Minimal contract every backend must satisfy."""

    name: str = "base"

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Return a completion for (system, user)."""

    def summarize(self, text: str, max_tokens: int = 200) -> str:
        """Condense text. Default implementation asks the same model."""
        resp = self.complete(
            "You compress agent history into dense notes. Keep every concrete "
            "value, decision, and error. No preamble.",
            f"Compress to under {max_tokens} tokens:\n\n{text}",
            max_tokens=max_tokens,
        )
        return resp.text.strip() if resp.ok else ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.config.model!r}>"
