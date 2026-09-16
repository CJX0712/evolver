"""Core data model for Evolver.

Design notes
------------
The whole engine is built around one idea: **an agent run is a trajectory, and
every trajectory is training data**. Successes are distilled into reusable
``Skill`` objects; failures are distilled into ``Pitfall`` records so the agent
learns what *not* to do. Most frameworks only harvest the success half of the
signal -- this module models both.

Everything here is a plain dataclass: serialisable, diffable, dependency-free.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class StepKind(str, Enum):
    """What the agent emitted for a single step."""

    THOUGHT = "thought"
    CODE = "code"          # CodeAct: a Python block that calls tools
    TOOL_CALL = "tool_call"  # legacy/atomic tool invocation
    FINAL = "final"


class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"

    @property
    def is_success(self) -> bool:
        return self is RunStatus.SUCCESS


@dataclass(slots=True)
class ToolSpec:
    """A callable exposed to the agent's generated code."""

    name: str
    description: str
    signature: str = ""
    fn: Optional[Callable[..., Any]] = None

    def render(self) -> str:
        sig = self.signature or f"{self.name}(*args, **kwargs)"
        return f"{sig}\n    \"\"\"{self.description}\"\"\""


@dataclass(slots=True)
class Step:
    """One turn of an agent run."""

    kind: StepKind
    content: str
    observation: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(slots=True)
class Trajectory:
    """A complete attempt at a task -- the unit of learning."""

    task: str
    task_id: str = ""
    steps: List[Step] = field(default_factory=list)
    status: RunStatus = RunStatus.SUCCESS
    final_answer: str = ""
    failure_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory_id: str = field(default_factory=lambda: _new_id("traj"))
    created_at: float = field(default_factory=time.time)

    # -- derived metrics -------------------------------------------------
    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def transcript(self, include_observations: bool = True) -> str:
        """Flatten to text -- the raw material for distillation."""
        lines: List[str] = [f"TASK: {self.task}"]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"\n[{i}] {s.kind.value}: {s.content}")
            if include_observations and s.observation:
                lines.append(f"    -> {s.observation}")
            if s.error:
                lines.append(f"    !! {s.error}")
        lines.append(f"\nSTATUS: {self.status.value}")
        if self.final_answer:
            lines.append(f"ANSWER: {self.final_answer}")
        if self.failure_reason:
            lines.append(f"REASON: {self.failure_reason}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        for s in d["steps"]:
            s["kind"] = StepKind(s["kind"]).value
        return d


@dataclass(slots=True)
class SkillStats:
    """Utility tracking -- drives curation and decay."""

    uses: int = 0
    successes: int = 0
    failures: int = 0
    tokens_saved: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses else 0.0

    def record(self, success: bool, tokens_saved: int = 0) -> None:
        self.uses += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.tokens_saved += max(0, tokens_saved)
        self.last_used = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillStats":
        return cls(**d)


@dataclass(slots=True)
class Skill:
    """A reusable capability distilled from a successful trajectory.

    A skill is *not* a prompt snippet -- it is executable Python with a typed
    signature, so it can be injected into the CodeAct sandbox and called
    directly, skipping re-reasoning entirely.
    """

    name: str
    description: str
    signature: str
    body: str
    trigger: str = ""                      # natural-language match hint
    tags: List[str] = field(default_factory=list)
    source_trajectory: str = ""
    stats: SkillStats = field(default_factory=SkillStats)
    skill_id: str = field(default_factory=lambda: _new_id("skill"))
    created_at: float = field(default_factory=time.time)
    generation: int = 0                    # bumped when merged with another
    parents: List[str] = field(default_factory=list)

    @property
    def code(self) -> str:
        """The skill as executable source.

        ``signature`` already carries the parameter list, so the colon lives
        here -- and ``validate_source`` on the way in must see a real function
        definition, which it would not if the colon were missing.
        """
        return f"def {self.signature}:\n{self.body}"

    @property
    def score(self) -> float:
        """Curation score: usefulness, tempered by age and reliability.

        An unused skill is dead weight in the context window, so recency is a
        first-class term. Reliability matters more than raw use count.
        """
        if self.stats.uses == 0:
            return 0.0
        age_days = max(0.0, (time.time() - self.stats.last_used) / 86400.0)
        recency = 1.0 / (1.0 + 0.15 * age_days)          # half-life ~ 4.6 days
        reliability = 0.35 + 0.65 * self.stats.success_rate
        volume = min(1.0, self.stats.uses / 8.0)          # saturates at 8 uses
        efficiency = min(1.0, self.stats.tokens_saved / 4000.0)
        return round(recency * reliability * (0.55 + 0.30 * volume + 0.15 * efficiency), 6)

    def matches(self, text: str) -> float:
        """Cheap lexical relevance in [0, 1] -- used to pick skills for a task.

        Deliberately does *not* search ``self.description``: that text quotes
        the original task, so it matches the originating task perfectly and
        bleeds onto unrelated ones that happen to share a word. Matching on the
        trigger keywords alone keeps retrieval from firing on the wrong task.
        """
        if not self.trigger:
            return 0.0
        hay = text.lower()
        needle_tokens = [t for t in self.trigger.split() if len(t) > 2]
        if not needle_tokens:
            return 0.0
        hits = sum(1 for t in needle_tokens if t in hay)
        return hits / len(needle_tokens)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stats"] = self.stats.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        d = dict(d)
        d["stats"] = SkillStats.from_dict(d.get("stats", {}))
        return cls(**d)


@dataclass(slots=True)
class Pitfall:
    """A lesson distilled from a *failed* trajectory.

    Skills tell the agent what to do. Pitfalls tell it what to avoid. Storing
    both halves of the signal is what makes the difference between an agent
    that repeats its mistakes and one that stops making them.
    """

    description: str
    context: str = ""
    wrong_action: str = ""
    correct_action: str = ""
    tags: List[str] = field(default_factory=list)
    source_trajectory: str = ""
    hits: int = 0          # how often it was surfaced
    avoids: int = 0        # how often the run avoided the failure after being warned
    pitfall_id: str = field(default_factory=lambda: _new_id("pfall"))
    created_at: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        if self.hits == 0:
            return 0.0
        return round(self.avoids / self.hits, 6)

    def matches(self, text: str) -> float:
        """Relevance to ``text``, scored over the pitfall's keyword tags.

        Tags only -- not ``description``, which is a full sentence and would
        match almost any task on common words like "the" or "agent".
        """
        if not self.tags:
            return 0.0
        hay = text.lower()
        hits = sum(1 for t in self.tags if t in hay)
        return hits / len(self.tags)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pitfall":
        return cls(**d)


@dataclass(slots=True)
class EvolutionReport:
    """Per-epoch benchmark measurements."""

    epoch: int
    success_rate: float
    avg_steps: float
    avg_tokens: float
    skill_count: int
    skill_hit_rate: float
    pitfalls_active: int
    wall_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
