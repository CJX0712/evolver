"""A deterministic synthetic agent for mechanism validation.

Read this before trusting any benchmark number it produces
---------------------------------------------------------
This adapter does **not** call a language model. It is a scripted behavioural
simulation with three distinct competence levels, so the benchmark can
attribute improvement to a specific mechanism:

1. **No skill, no warning** -- fumbles through ``naive_turns`` turns and then
   emits the answer a naive composition produces. For eight of the ten tasks
   that answer is wrong, because the task contains a trap.
2. **Pitfall warning, no skill** -- same turn count, but routes around the
   known trap. This isolates the value of learning from *failures*.
3. **Skill available** -- calls the distilled skill in a single turn. This
   isolates the value of learning from *successes*.

So the benchmark answers a precise question: *if an agent can accumulate skills
and pitfalls, do success rate, step count, and token cost improve?* It is
**not** evidence about any particular frontier model -- that needs a real
adapter (see :mod:`evolver.llm.vendors`).

The upside is that the entire pipeline runs offline, in CI, in a reviewer's
hands, with no API key and byte-identical reruns from a fixed seed.
"""

from __future__ import annotations

import ast
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from evolver.bench.tasks import BENCH_TASKS, BenchTask, find_task
from evolver.core.config import LLMConfig
from evolver.llm.base import LLMAdapter, LLMResponse

_TASK_RE = re.compile(r"TASK:\s*(.*?)(?:\n\nTOOLS:|\Z)", re.DOTALL)
_SKILL_RE = re.compile(r"REUSABLE SKILLS.*?(?:\n\nAVAILABLE|\Z)", re.DOTALL)
_PITFALL_RE = re.compile(r"KNOWN PITFALLS.*?\Z", re.DOTALL)
_TURN_RE = re.compile(r"^\[\d+\] code:", re.MULTILINE)
_STATE_TURN_RE = re.compile(r"^turn\s*=\s*(\d+)", re.MULTILINE)


@dataclass
class ReplayAdapter(LLMAdapter):
    """Deterministic stand-in for a real model."""

    tasks: List[BenchTask] = field(default_factory=lambda: list(BENCH_TASKS))
    seed: int = 42
    jitter: float = 0.0          # set >0 to model a flaky model
    config: LLMConfig = field(default_factory=lambda: LLMConfig(provider="replay"))

    name: str = "replay"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.calls = 0
        self.skill_hits = 0
        self.warned_hits = 0

    # -- prompt parsing --------------------------------------------------
    @staticmethod
    def _task_text(user: str) -> str:
        m = _TASK_RE.search(user or "")
        return (m.group(1) if m else (user or "")).strip()

    @staticmethod
    def _has_warnings(user: str) -> bool:
        return bool(_PITFALL_RE.search(user or ""))

    @staticmethod
    def _skills_block(user: str) -> str:
        m = _SKILL_RE.search(user or "")
        return m.group(0) if m else ""

    def _turn(self, user: str) -> int:
        m = _STATE_TURN_RE.search(user or "")
        if m:
            return int(m.group(1))
        return len(_TURN_RE.findall(user or ""))

    def _find_skill(self, block: str, marker: str) -> Optional[Tuple[str, int]]:
        """Locate an injected skill whose name contains ``marker``.

        Returns (function_name, arity) or None.
        """
        if not block or not marker:
            return None
        try:
            tree = ast.parse(block)
        except SyntaxError:
            # The block has prose around the code; fall back to a regex.
            m = re.search(rf"def\s+(\w*{re.escape(marker)}\w*)\s*\(([^)]*)\)", block)
            if not m:
                return None
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            return m.group(1), len(params)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and marker in node.name:
                return node.name, len(node.args.args)
        return None

    # -- the model -------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        self.calls += 1

        task_text = self._task_text(user)
        spec: Optional[BenchTask] = find_task(task_text)
        turn = self._turn(user)
        warned = self._has_warnings(user)

        if spec is None:
            code = "# no strategy matches this task\nfinish(None)\n"
        else:
            found = self._find_skill(self._skills_block(user), spec.skill_marker)
            if found is not None:
                fname, arity = found
                args = ", ".join((spec.args + ["[]"] * arity)[:arity])
                code = f"result = {fname}({args})\nfinish(result)\n"
                self.skill_hits += 1
            else:
                if warned:
                    self.warned_hits += 1
                if turn + 1 >= spec.naive_turns:
                    code = spec.render_final(warned=warned)
                    if self.jitter and self.rng.random() < self.jitter:
                        code = "result = None\nfinish(result)\n"
                else:
                    code = spec.render_naive(turn + 1)

        text = f"```python\n{code}```"
        return LLMResponse(
            text=text,
            tokens_in=max(1, len(system + user) // 4),
            tokens_out=max(1, len(text) // 4),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self.config.model,
        )

    def reset_stats(self) -> None:
        self.calls = 0
        self.skill_hits = 0
        self.warned_hits = 0

    def stats(self) -> Dict[str, int]:
        return {"calls": self.calls, "skill_hits": self.skill_hits,
                "warned_hits": self.warned_hits}
