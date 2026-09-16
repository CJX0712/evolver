"""The Evolver engine: one loop that runs tasks and gets better at them.

Per step: select skills -> surface pitfalls -> render a bounded context ->
ask the model for a code block -> execute it in the sandbox -> record.

Per run: verify the answer, then distill -- a success becomes a skill, a
failure becomes a pitfall. Every N runs, the curator prunes and merges.

The loop is deliberately small. Everything interesting lives in the components
it calls, which is what makes each mechanism independently testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from evolver.act.codeact import CodeActExecutor, extract_code
from evolver.core.config import Config
from evolver.core.types import (
    EvolutionReport,
    RunStatus,
    Skill,
    Step,
    StepKind,
    ToolSpec,
    Trajectory,
)
from evolver.evolve.curator import Curator
from evolver.evolve.distill import Distiller
from evolver.evolve.store import SkillStore
from evolver.llm import build_adapter
from evolver.llm.base import LLMAdapter
from evolver.memory.context import ContextWindow


@dataclass
class Evolver:
    """An agent that improves with use."""

    config: Config = field(default_factory=Config)
    tools: List[ToolSpec] = field(default_factory=list)
    adapter: Optional[LLMAdapter] = None
    store: Optional[SkillStore] = None
    max_steps: int = 8

    def __post_init__(self) -> None:
        self.adapter = self.adapter or build_adapter(self.config)
        self.store = self.store or SkillStore.load(self.config.store_path)
        self.executor = CodeActExecutor(tools=self.tools, config=self.config.sandbox)
        self.distiller = Distiller(
            adapter=self.adapter if self.config.llm.provider != "replay" else None,
            tools=", ".join(t.name for t in self.tools),
        )
        self.curator = Curator(config=self.config.evolve)
        self._baseline_tokens: Dict[str, int] = {}
        self.run_history: List[Trajectory] = []

    # -- skill plumbing --------------------------------------------------
    def _pick_skills(self, task: str) -> List[Skill]:
        if not self.config.evolve.distill_enabled:
            return []
        return self.store.select(task, k=self.config.evolve.top_k_skills,
                                 threshold=self.config.evolve.skill_match_threshold)

    def _pick_warnings(self, task: str) -> List[str]:
        if not self.config.evolve.distill_enabled:
            return []
        return [p.description for p in self.store.warnings(task, k=2)]

    # -- the loop --------------------------------------------------------
    def run(
        self,
        task: str,
        *,
        task_id: str = "",
        verify: Optional[Callable[[Any], bool]] = None,
        max_steps: Optional[int] = None,
    ) -> Trajectory:
        """Execute one task and learn from it."""
        limit = max_steps or self.max_steps
        traj = Trajectory(task=task, task_id=task_id or task[:48])

        ctx = ContextWindow(config=self.config.context)
        ctx.set_resident(task)

        started = time.perf_counter()
        status = RunStatus.FAILURE
        failure_reason = ""
        injected: List[str] = []

        for turn in range(limit):
            skills = self._pick_skills(task)
            warnings = self._pick_warnings(task)
            injected = self.executor.inject_skills(skills) if skills else []

            prompt = self.executor.build_prompt(
                ctx.render(),
                skills=skills,
                state={"turn": turn},
                warnings=warnings,
            )
            resp = self.adapter.complete(self.executor.system_prompt(), prompt)

            code = extract_code(resp.text)
            record = Step(
                kind=StepKind.CODE,
                content=code,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                latency_ms=resp.latency_ms,
            )

            if not resp.ok:
                record.error = resp.error
                ctx.add(record)
                traj.add(record)
                status = RunStatus.ERROR
                failure_reason = f"model error: {resp.error}"
                break

            result = self.executor.execute(code)
            record.observation = self.executor.observe(result)
            if result.error:
                record.error = result.error
            ctx.add(record)
            traj.add(record)

            if result.final_answer is not None:
                traj.final_answer = str(result.final_answer)
                ok = verify(traj.final_answer) if verify else bool(traj.final_answer.strip())
                if ok:
                    status = RunStatus.SUCCESS
                else:
                    status = RunStatus.FAILURE
                    failure_reason = f"wrong answer: {traj.final_answer!r}"
                break
        else:
            status = RunStatus.TIMEOUT
            failure_reason = f"no final answer within {limit} steps"

        traj.status = status
        traj.failure_reason = failure_reason
        traj.metadata["wall_time_s"] = round(time.perf_counter() - started, 4)
        traj.metadata["injected_skills"] = list(injected)
        traj.metadata["context"] = ctx.stats()

        self._credit_skills(traj, task, injected)
        self._learn(traj)

        self.run_history.append(traj)
        self.store.runs += 1
        return traj

    # -- learning --------------------------------------------------------
    def _credit_skills(self, traj: Trajectory, task: str, injected: List[str]) -> None:
        """Attribute outcome to the skills that were actually invoked."""
        if not injected:
            return
        used = [
            name for name in injected
            if any(name in s.content for s in traj.steps if s.kind is StepKind.CODE)
        ]
        if not used:
            return

        baseline = self._baseline_tokens.setdefault(task, traj.total_tokens)
        saved = max(0, baseline - traj.total_tokens) if traj.succeeded else 0
        for name in used:
            self.store.record_skill_use(name, traj.succeeded, saved // max(1, len(used)))
        traj.metadata["skills_used"] = used
        traj.metadata["skill_hit"] = True

    def _learn(self, traj: Trajectory) -> None:
        if not self.config.evolve.distill_enabled:
            return
        outcome = self.distiller.distill(traj)
        if outcome.skill is not None:
            self.store.add_skill(outcome.skill)
        if outcome.pitfall is not None:
            self.store.add_pitfall(outcome.pitfall)
        traj.metadata["distilled"] = outcome.reason

        if self.config.evolve.curate_every_n_runs > 0 and \
           self.store.runs % self.config.evolve.curate_every_n_runs == 0:
            actions = self.curator.curate(self.store)
            if actions:
                traj.metadata["curated"] = [str(a) for a in actions]

    # -- persistence -----------------------------------------------------
    def save(self, path: Optional[str] = None) -> str:
        return self.store.save(path or self.config.store_path)

    def stats(self) -> Dict[str, Any]:
        return self.store.stats()

    def report(self, epochs: List[EvolutionReport]) -> List[EvolutionReport]:
        return epochs
