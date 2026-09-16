"""Token accounting.

Accurate-enough estimation without a hard dependency: the heuristic is tuned
against cl100k_base, and ``tiktoken`` is used automatically when installed.
Every number the benchmark reports flows through here, so consistency matters
more than byte-perfect precision.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)

# Tuned so that mixed Chinese/English text lands within ~10% of cl100k_base.
_CHARS_PER_TOKEN_ASCII = 4.0
_TOKENS_PER_CJK = 1.0


@lru_cache(maxsize=1)
def _encoder():
    """Return a tiktoken encoding, or None if unavailable."""
    try:  # pragma: no cover - depends on optional dependency
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: Any) -> int:
    """Estimate the token cost of ``text`` (str, or anything str-able)."""
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0

    enc = _encoder()
    if enc is not None:  # pragma: no cover
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass

    cjk = len(_CJK.findall(text))
    ascii_len = len(text) - cjk
    return max(1, int(cjk * _TOKENS_PER_CJK + ascii_len / _CHARS_PER_TOKEN_ASCII))


def count_messages(messages: List[Dict[str, Any]]) -> int:
    """Count tokens across chat messages, including per-message overhead."""
    total = 0
    for m in messages:
        total += 4  # role/separator overhead, per OpenAI's accounting rule
        for k, v in m.items():
            if isinstance(v, str):
                total += count_tokens(v)
        total += 2
    return total + 2


def truncate_to_tokens(text: str, budget: int) -> str:
    """Head-truncate ``text`` so it costs at most ``budget`` tokens."""
    if count_tokens(text) <= budget:
        return text
    # Binary search on character length -- cheap and exact enough.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + " …[truncated]"


def code_ratio(text: str) -> float:
    """Fraction of text that is fenced code -- used to weight summarisation."""
    if not text:
        return 0.0
    code = sum(len(m) for m in _CODE_FENCE.findall(text))
    return min(1.0, code / len(text))
