"""Vendor adapters.

Both are thin on purpose: no retries-with-cleverness, no vendor-specific
prompting. Evolver's contribution is the layer above the model, not another
HTTP client. Install the extras you need::

    pip install evolver[openai]
    pip install evolver[anthropic]
"""

from __future__ import annotations

import os
import time
from typing import Optional

from evolver.core.config import LLMConfig
from evolver.llm.base import LLMAdapter, LLMResponse


class OpenAIAdapter(LLMAdapter):
    """Any OpenAI-compatible chat completions endpoint."""

    name = "openai"

    def __init__(self, config: Optional[LLMConfig] = None, client: object = None) -> None:
        self.config = config or LLMConfig(provider="openai", model="gpt-4o-mini")
        self._client = client
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415 - optional dependency
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "OpenAIAdapter needs the openai package: pip install evolver[openai]"
                ) from exc
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    f"No API key. Set ${self.config.api_key_env} or pass config.api_key."
                )
            self._client = OpenAI(api_key=api_key, base_url=self.config.base_url)

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=self.config.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                timeout=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            return LLMResponse(text="", error=f"{type(exc).__name__}: {exc}",
                               latency_ms=(time.perf_counter() - started) * 1000)

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self.config.model,
            raw=resp,
        )


class AnthropicAdapter(LLMAdapter):
    """Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, config: Optional[LLMConfig] = None, client: object = None) -> None:
        self.config = config or LLMConfig(
            provider="anthropic", model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        )
        self._client = client
        if self._client is None:
            try:
                from anthropic import Anthropic  # noqa: PLC0415 - optional dependency
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "AnthropicAdapter needs the anthropic package: pip install evolver[anthropic]"
                ) from exc
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    f"No API key. Set ${self.config.api_key_env} or pass config.api_key."
                )
            self._client = Anthropic(api_key=api_key, base_url=self.config.base_url)

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=self.config.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=self.config.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                timeout=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            return LLMResponse(text="", error=f"{type(exc).__name__}: {exc}",
                               latency_ms=(time.perf_counter() - started) * 1000)

        text = "".join(
            getattr(b, "text", "") for b in getattr(resp, "content", [])
        )
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "input_tokens", 0) or 0,
            tokens_out=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self.config.model,
            raw=resp,
        )
