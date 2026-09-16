"""Naming helpers for distilled artifacts.

Skill names are not cosmetic: the retrieval path matches on them, and a stable
name derived from task prose means the same task family converges on the same
skill instead of spawning near-duplicates every run.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the in of on at by for from with to and or is are was were be been
    find compute get return give what which that this it its please answer just
    after normalising normalizing case punctuation fully completely up all
    numbers number period value values task using use make show""".split()
)


_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")


def name_from_text(text: str, max_words: int = 5) -> str:
    """Derive a snake_case identifier from task prose.

    Quoted spans are stripped first: they hold *data* ("Level", a word list),
    and letting data into a skill name would give every new input its own
    skill instead of one per task family.
    """
    prose = _QUOTED.sub(" ", text or "")
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", prose)
    keep = [w.lower() for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not keep:
        keep = ["task"]
    if not keep:
        keep = ["task"]
    name = "_".join(keep[:max_words])
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "task"


def token_set(text: str) -> frozenset:
    """Content tokens, used for skill similarity."""
    words = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    return frozenset(w for w in words if w not in STOPWORDS and len(w) > 1)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
