"""Evolver -- the self-evolving layer for AI agents.

Four mechanisms, composed:

1. **CodeAct execution** -- the model writes one Python block that calls many
   tools, instead of one tool call per reasoning turn.
2. **Skill distillation** -- successful trajectories become reusable skills;
   failed ones become pitfalls.
3. **Autonomous curation** -- a background curator scores, prunes, and merges
   the skill library so it never bloats.
4. **Layered context** -- a 30K budget with automatic compaction, so cost and
   hallucination rate fall instead of rising with session length.

The benchmark in :mod:`evolver.bench` measures whether all of that actually
makes the agent better over time -- which is the only claim worth making.
"""

from evolver.core.config import Config
from evolver.core.types import (
    EvolutionReport,
    Pitfall,
    RunStatus,
    Skill,
    SkillStats,
    Step,
    StepKind,
    ToolSpec,
    Trajectory,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "EvolutionReport",
    "Pitfall",
    "RunStatus",
    "Skill",
    "SkillStats",
    "Step",
    "StepKind",
    "ToolSpec",
    "Trajectory",
    "Evolver",
]


def __getattr__(name: str):  # lazy import keeps `import evolver` dependency-free
    if name == "Evolver":
        from evolver.core.engine import Evolver

        return Evolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
