"""Model adapters.

``replay`` is the dependency-free default: deterministic, offline, and useful
for validating the evolution machinery itself. Swap in ``openai`` or
``anthropic`` when you want real model numbers.
"""

from evolver.llm.base import LLMAdapter, LLMResponse
from evolver.llm.replay import ReplayAdapter


def build_adapter(config) -> LLMAdapter:
    """Instantiate the adapter named by ``config.llm.provider``."""
    provider = (config.llm.provider or "replay").lower()
    if provider == "replay":
        return ReplayAdapter(config=config.llm)
    if provider == "openai":
        from evolver.llm.vendors import OpenAIAdapter

        return OpenAIAdapter(config=config.llm)
    if provider == "anthropic":
        from evolver.llm.vendors import AnthropicAdapter

        return AnthropicAdapter(config=config.llm)
    raise ValueError(f"unknown provider: {provider!r}")


__all__ = ["LLMAdapter", "LLMResponse", "ReplayAdapter", "build_adapter"]
