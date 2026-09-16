"""Atomic tools available to agent-generated code.

Deliberately low-level. Each does exactly one thing, so the *composition* is
what has to be learned -- which is the whole point. If the tools were
high-level ("just solve it"), the agent would have nothing to learn and the
benchmark would measure nothing.

Several pairs here differ by a subtlety that bites the unwary:
``odds``/``evens``, ``flatten1``/``deep_flatten``, ``count_words``/
``count_words_normalized``. Those gaps are where skills earn their keep.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from evolver.core.types import ToolSpec


def sort_desc(xs: Sequence[Any]) -> List[Any]:
    """Sort descending."""
    return sorted(xs, reverse=True)


def sort_asc(xs: Sequence[Any]) -> List[Any]:
    """Sort ascending."""
    return sorted(xs)


def unique(xs: Iterable[Any]) -> List[Any]:
    """De-duplicate, preserving first-seen order."""
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def dedup_sorted_desc(xs: Iterable[Any]) -> List[Any]:
    """De-duplicate then sort descending."""
    return sorted(set(xs), reverse=True)


def nth(xs: Sequence[Any], i: int) -> Any:
    """0-indexed element access."""
    return list(xs)[i]


def evens(xs: Iterable[int]) -> List[int]:
    """Keep even numbers."""
    return [x for x in xs if x % 2 == 0]


def odds(xs: Iterable[int]) -> List[int]:
    """Keep odd numbers."""
    return [x for x in xs if x % 2 == 1]


def squares(xs: Iterable[float]) -> List[float]:
    """Square each element."""
    return [x * x for x in xs]


def total(xs: Iterable[float]) -> float:
    """Sum elements."""
    return sum(xs)


def mean(xs: Iterable[float]) -> float:
    """Arithmetic mean."""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def count_words(text: str) -> Dict[str, int]:
    """Whitespace-split word counts, case sensitive, punctuation retained."""
    out: Dict[str, int] = {}
    for w in text.split():
        out[w] = out.get(w, 0) + 1
    return out


def count_words_normalized(text: str) -> Dict[str, int]:
    """Word counts, lowercased with punctuation stripped."""
    out: Dict[str, int] = {}
    for raw in text.split():
        w = "".join(ch for ch in raw.lower() if ch.isalnum())
        if w:
            out[w] = out.get(w, 0) + 1
    return out


def max_key(d: Dict[Any, Any]) -> Any:
    """Key with the largest value."""
    return max(d, key=d.get) if d else None


def merge_sum(a: Dict[Any, float], b: Dict[Any, float]) -> Dict[Any, float]:
    """Merge two dicts, adding values of shared keys."""
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def merge_overwrite(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge two dicts, b wins on conflict. Usually the wrong answer."""
    out = dict(a)
    out.update(b)
    return out


def flatten1(xs: Iterable[Any]) -> List[Any]:
    """Flatten exactly one level."""
    out: List[Any] = []
    for x in xs:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def deep_flatten(xs: Any) -> List[Any]:
    """Flatten arbitrarily nested lists."""
    out: List[Any] = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            out.extend(deep_flatten(x))
    else:
        out.append(xs)
    return out


def windows(xs: Sequence[Any], k: int) -> List[List[Any]]:
    """All contiguous windows of length k."""
    xs = list(xs)
    return [xs[i : i + k] for i in range(len(xs) - k + 1)]


def windows_truncated(xs: Sequence[Any], k: int) -> List[List[Any]]:
    """Windows of length k, dropping the final one. Off by one, on purpose."""
    xs = list(xs)
    return [xs[i : i + k] for i in range(max(0, len(xs) - k))]


def norm(s: str) -> str:
    """Lowercase, strip non-alphanumerics."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def rev(s: str) -> str:
    """Reverse a string."""
    return s[::-1]


def group_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    """Bucket rows by a shared key value."""
    out: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r.get(key), []).append(r)
    return out


def sum_field(rows: Iterable[Dict[str, Any]], field: str) -> float:
    """Sum one numeric field across rows."""
    return sum(float(r.get(field, 0)) for r in rows)


TOOLS: List[ToolSpec] = [
    ToolSpec("sort_desc", "Sort descending.", "sort_desc(xs)", sort_desc),
    ToolSpec("sort_asc", "Sort ascending.", "sort_asc(xs)", sort_asc),
    ToolSpec("unique", "De-duplicate preserving order.", "unique(xs)", unique),
    ToolSpec("dedup_sorted_desc", "De-duplicate then sort descending.",
             "dedup_sorted_desc(xs)", dedup_sorted_desc),
    ToolSpec("nth", "0-indexed element.", "nth(xs, i)", nth),
    ToolSpec("evens", "Keep even numbers.", "evens(xs)", evens),
    ToolSpec("odds", "Keep odd numbers.", "odds(xs)", odds),
    ToolSpec("squares", "Square each element.", "squares(xs)", squares),
    ToolSpec("total", "Sum elements.", "total(xs)", total),
    ToolSpec("mean", "Arithmetic mean.", "mean(xs)", mean),
    ToolSpec("count_words", "Word counts, case sensitive.", "count_words(text)",
             count_words),
    ToolSpec("count_words_normalized", "Word counts, lowercase, punctuation stripped.",
             "count_words_normalized(text)", count_words_normalized),
    ToolSpec("max_key", "Key with largest value.", "max_key(d)", max_key),
    ToolSpec("merge_sum", "Merge dicts, adding shared keys.", "merge_sum(a, b)",
             merge_sum),
    ToolSpec("merge_overwrite", "Merge dicts, b wins.", "merge_overwrite(a, b)",
             merge_overwrite),
    ToolSpec("flatten1", "Flatten one level.", "flatten1(xs)", flatten1),
    ToolSpec("deep_flatten", "Flatten all levels.", "deep_flatten(xs)", deep_flatten),
    ToolSpec("windows", "All contiguous windows of length k.", "windows(xs, k)",
             windows),
    ToolSpec("windows_truncated", "Windows of length k, last one dropped.",
             "windows_truncated(xs, k)", windows_truncated),
    ToolSpec("norm", "Lowercase, strip non-alphanumerics.", "norm(s)", norm),
    ToolSpec("rev", "Reverse a string.", "rev(s)", rev),
    ToolSpec("group_by", "Bucket rows by key.", "group_by(rows, key)", group_by),
    ToolSpec("sum_field", "Sum one field across rows.", "sum_field(rows, field)",
             sum_field),
]


def as_dict() -> Dict[str, Any]:
    return {t.name: t.fn for t in TOOLS if t.fn is not None}
