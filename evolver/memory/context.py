"""Layered context with automatic compaction.

Most agents treat the context window as a log: append forever, and hope the
model finds the needle. That fails in two ways -- cost grows linearly with
session length, and accuracy degrades because the model is wading through
noise.

This module implements three tiers instead:

* **Resident** -- task, goal, hard constraints. Never compressed, always present.
* **Working** -- the last N steps, verbatim. This is where reasoning happens.
* **Archive** -- everything older, reduced to a one-line summary each.

When the window crosses ``compact_threshold``, the oldest working steps are
distilled into archive entries. The invariant: the agent always sees the goal
and its most recent reasoning in full, and never pays full price for history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from evolver.core.config import ContextConfig
from evolver.core.types import Step, StepKind
from evolver.memory.tokens import count_tokens, truncate_to_tokens


def heuristic_summarize(step: Step, max_tokens: int = 60) -> str:
    """Compress a step without calling a model.

    Heuristics beat an LLM call here: summarising costs tokens and latency, and
    for a single step the structural content is mostly recoverable from the
    shape of the step -- first line of thought, first meaningful line of code,
    and any error.
    """
    if step.kind is StepKind.THOUGHT:
        head = step.content.strip().splitlines()
        text = head[0] if head else ""
    elif step.kind is StepKind.CODE:
        lines = [ln.strip() for ln in step.content.splitlines() if ln.strip()]
        # Skip boilerplate: imports and comment-only lines carry little signal.
        lines = [ln for ln in lines if not ln.startswith(("#", "import ", "from "))]
        text = "; ".join(lines[:2]) if lines else "ran code"
    else:
        text = step.content.strip()

    if step.error:
        text = f"{text} [error: {step.error[:60]}]"
    elif step.observation:
        obs = step.observation.strip().replace("\n", " ")[:80]
        text = f"{text} -> {obs}"

    return truncate_to_tokens(text or "(step)", max_tokens)


@dataclass
class ContextWindow:
    """A compacting view over a sequence of steps."""

    config: ContextConfig = field(default_factory=ContextConfig)
    resident: str = ""
    steps: List[Step] = field(default_factory=list)
    archive: List[str] = field(default_factory=list)
    summarizer: Optional[Callable[[Step], str]] = None
    compactions: int = 0

    # -- mutation --------------------------------------------------------
    def set_resident(self, text: str) -> None:
        self.resident = text

    def add(self, step: Step) -> None:
        self.steps.append(step)
        self.maybe_compact()

    def clear(self) -> None:
        self.steps.clear()
        self.archive.clear()
        self.resident = ""

    # -- accounting ------------------------------------------------------
    def render(self) -> str:
        """Assemble the full window text."""
        parts: List[str] = []
        if self.resident:
            parts.append(f"## GOAL\n{self.resident}")
        if self.archive:
            parts.append("## EARLIER (compressed)\n" + "\n".join(f"- {a}" for a in self.archive))
        if self.steps:
            keep = self.config.keep_recent_steps
            recent = self.steps[-keep:] if keep > 0 else self.steps
            rendered = [self._render_step(i, s) for i, s in enumerate(recent, 1)]
            parts.append("## RECENT\n" + "\n".join(rendered))
        return "\n\n".join(parts)

    @staticmethod
    def _render_step(idx: int, s: Step) -> str:
        block = f"[{idx}] {s.kind.value}: {s.content}"
        if s.observation:
            block += f"\n    -> {s.observation}"
        if s.error:
            block += f"\n    !! {s.error}"
        return block

    @property
    def raw_tokens(self) -> int:
        """Cost if every step were expanded -- the pressure that triggers compaction.

        Must not be measured on :meth:`render`, which already hides older
        steps. Judging budget pressure by the truncated view means the window
        never looks full and compaction never fires -- a self-fulfilling lie.
        """
        parts = [self.resident or ""]
        parts += [f"- {a}" for a in self.archive]
        parts += [self._render_step(i, s) for i, s in enumerate(self.steps, 1)]
        return count_tokens("\n".join(parts))

    def usage(self) -> Tuple[int, int, float]:
        """Return (tokens_used, budget, ratio), measured on the untruncated content."""
        used = self.raw_tokens
        budget = self.config.max_tokens
        return used, budget, (used / budget if budget else 0.0)

    @property
    def tokens(self) -> int:
        """What :meth:`render` actually costs -- the real prompt payload."""
        return count_tokens(self.render())

    # -- compaction ------------------------------------------------------
    def should_compact(self) -> bool:
        _, _, ratio = self.usage()
        return ratio >= self.config.compact_threshold

    def maybe_compact(self) -> bool:
        if not self.should_compact():
            return False
        self.compact()
        return True

    def compact(self) -> int:
        """Fold all but the newest steps into archive. Returns steps folded."""
        keep = max(1, self.config.keep_recent_steps)
        if len(self.steps) <= keep:
            return 0

        fold = self.steps[:-keep]
        # Cap each entry: a 400-token "summary" is not a summary.
        budget = max(40, min(120, self.config.archive_summary_tokens // max(1, len(fold))))
        for s in fold:
            try:
                summary = (
                    self.summarizer(s)
                    if self.summarizer
                    else heuristic_summarize(s, budget)
                )
            except Exception:  # noqa: BLE001 - never lose a step to a bad summary
                summary = heuristic_summarize(s, budget)
            self.archive.append(summary)

        self.steps = self.steps[-keep:]
        self.compactions += 1

        # Merge down while over budget, then enforce a hard ceiling. Merging
        # alone cannot shrink the archive -- joining two entries keeps their
        # combined tokens -- so the ceiling is what actually bounds it.
        while self.raw_tokens > self.config.max_tokens and len(self.archive) > 1:
            self._collapse_archive()
        self._enforce_archive_budget()
        return len(fold)

    def _enforce_archive_budget(self) -> None:
        """Drop the oldest archive entries until the archive fits its share."""
        cap = max(200, int(self.config.max_tokens * 0.35))
        while self.archive and count_tokens("\n".join(self.archive)) > cap:
            drop = max(1, len(self.archive) // 2)
            self.archive = self.archive[drop:]

    def _collapse_archive(self) -> None:
        """Halve the archive by pairing adjacent entries."""
        merged: List[str] = []
        for i in range(0, len(self.archive), 2):
            chunk = self.archive[i : i + 2]
            merged.append(" | ".join(chunk))
        if len(merged) >= len(self.archive):
            # Can't shrink further without discarding; drop the oldest quarter.
            drop = max(1, len(self.archive) // 4)
            merged = self.archive[drop:]
        self.archive = merged

    # -- introspection ---------------------------------------------------
    def stats(self) -> dict:
        used, budget, ratio = self.usage()
        return {
            "tokens": self.tokens,
            "raw_tokens": used,
            "budget": budget,
            "ratio": round(ratio, 4),
            "steps": len(self.steps),
            "archive_entries": len(self.archive),
            "compactions": self.compactions,
        }
