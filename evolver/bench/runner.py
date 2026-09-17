"""The evolution benchmark.

What it measures
----------------
Run the same fixed task set for several epochs, letting the skill library
accumulate between them, and record three curves:

* **success rate** -- should rise. Pitfalls drive most of this: an agent that
  has been warned about a trap stops walking into it.
* **average steps** -- should fall. Skills drive this: a distilled skill
  collapses a multi-turn derivation into one call.
* **average tokens** -- should fall, as a consequence of the above plus the
  bounded context window.

The two mechanisms are separable, which is the point. If success rises but
steps don't fall, pitfalls are working and skills aren't -- a much more useful
diagnosis than "the agent got better".

Honesty note
------------
With the default ``replay`` adapter these numbers validate the *mechanism* on a
scripted agent, not a real model. See :mod:`evolver.llm.replay`.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evolver.bench.tasks import BENCH_TASKS, BenchTask
from evolver.core.config import Config
from evolver.core.engine import Evolver
from evolver.core.types import EvolutionReport, Trajectory


@dataclass
class EpochRecord:
    epoch: int
    trajectories: List[Trajectory] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(t.succeeded for t in self.trajectories) / max(1, len(self.trajectories))

    @property
    def avg_steps(self) -> float:
        return statistics.fmean([t.n_steps for t in self.trajectories]) if self.trajectories else 0.0

    @property
    def avg_tokens(self) -> float:
        return statistics.fmean([t.total_tokens for t in self.trajectories]) if self.trajectories else 0.0

    @property
    def skill_hit_rate(self) -> float:
        hits = sum(1 for t in self.trajectories if t.metadata.get("skill_hit"))
        return hits / max(1, len(self.trajectories))


@dataclass
class BenchmarkResult:
    epochs: List[EvolutionReport] = field(default_factory=list)
    records: List[EpochRecord] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    skill_names: List[str] = field(default_factory=list)
    pitfall_count: int = 0
    wall_time_s: float = 0.0
    # Where the learned library landed on disk ("" when not persisted), plus a
    # non-empty message when persistence was attempted and failed.
    store_path: str = ""
    persistence_error: str = ""

    # -- derived ---------------------------------------------------------
    @property
    def first(self) -> Optional[EvolutionReport]:
        return self.epochs[0] if self.epochs else None

    @property
    def last(self) -> Optional[EvolutionReport]:
        return self.epochs[-1] if self.epochs else None

    def delta(self, key: str) -> float:
        if not self.first or not self.last:
            return 0.0
        return getattr(self.last, key) - getattr(self.first, key)

    def summary(self) -> Dict[str, Any]:
        f, l = self.first, self.last
        if not f or not l:
            return {}
        return {
            "epochs": len(self.epochs),
            "tasks_per_epoch": len(self.task_ids),
            "success_rate": f"{f.success_rate:.0%} -> {l.success_rate:.0%}",
            "avg_steps": f"{f.avg_steps:.2f} -> {l.avg_steps:.2f}",
            "avg_tokens": f"{f.avg_tokens:.0f} -> {l.avg_tokens:.0f}",
            "token_reduction": f"{(1 - l.avg_tokens / f.avg_tokens):.0%}" if f.avg_tokens else "n/a",
            "step_reduction": f"{(1 - l.avg_steps / f.avg_steps):.0%}" if f.avg_steps else "n/a",
            "skills_learned": len(self.skill_names),
            "pitfalls_learned": self.pitfall_count,
            "wall_time_s": round(self.wall_time_s, 2),
            "store_path": self.store_path or "-",
            "persistence_error": self.persistence_error or "-",
        }


def run_benchmark(
    config: Optional[Config] = None,
    *,
    tasks: Optional[List[BenchTask]] = None,
    epochs: int = 5,
    seed: int = 42,
    max_steps: int = 8,
    verbose: bool = False,
    evolver: Optional[Evolver] = None,
) -> BenchmarkResult:
    """Run the fixed task set for ``epochs`` rounds, accumulating skills."""
    cfg = config or Config()
    task_set = tasks or list(BENCH_TASKS)

    engine = evolver or Evolver(config=cfg, tools=_default_tools(), max_steps=max_steps)
    if hasattr(engine.adapter, "seed"):
        engine.adapter.seed = seed
        engine.adapter.rng.seed(seed)

    result = BenchmarkResult(task_ids=[t.task_id for t in task_set], config=cfg.to_dict())
    started = time.perf_counter()

    for epoch in range(epochs):
        rec = EpochRecord(epoch=epoch)
        for spec in task_set:
            traj = engine.run(
                spec.prompt,
                task_id=spec.task_id,
                verify=spec.verify,
                max_steps=max_steps,
            )
            rec.trajectories.append(traj)
            if verbose:
                mark = "ok " if traj.succeeded else "FAIL"
                print(f"  [e{epoch}] {mark} {spec.task_id:20s} "
                      f"steps={traj.n_steps} tok={traj.total_tokens} "
                      f"ans={str(traj.final_answer)[:24]!r}")

        result.records.append(rec)
        result.epochs.append(
            EvolutionReport(
                epoch=epoch,
                success_rate=round(rec.success_rate, 4),
                avg_steps=round(rec.avg_steps, 3),
                avg_tokens=round(rec.avg_tokens, 1),
                skill_count=len(engine.store.skills),
                skill_hit_rate=round(rec.skill_hit_rate, 4),
                pitfalls_active=len(engine.store.pitfalls),
                wall_time_s=round(time.perf_counter() - started, 3),
            )
        )
        if verbose:
            e = result.epochs[-1]
            print(f"epoch {epoch}: success={e.success_rate:.0%} steps={e.avg_steps:.2f} "
                  f"tokens={e.avg_tokens:.0f} skills={e.skill_count} hits={e.skill_hit_rate:.0%}")

    result.wall_time_s = time.perf_counter() - started
    result.skill_names = sorted(engine.store.skills.keys())
    result.pitfall_count = len(engine.store.pitfalls)
    result.engine = engine  # type: ignore[attr-defined]

    # A run that learns skills and then drops them on the floor is not an
    # evolution system, it is a very expensive way to compute a number.
    # Persist so the library survives the process and can be packed/shipped.
    result.store_path = ""
    store_path = getattr(cfg, "store_path", "")
    if store_path:
        try:
            engine.store.save(store_path)
            result.store_path = store_path
        except OSError as exc:  # pragma: no cover - disk issues are environmental
            result.persistence_error = f"{type(exc).__name__}: {exc}"
    return result


def _default_tools():
    from evolver.bench.tools import TOOLS

    return list(TOOLS)
