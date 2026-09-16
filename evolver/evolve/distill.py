"""Distillation: turning trajectories into reusable capability.

Two channels, because a trajectory carries signal whether it succeeded or not:

* **Success -> Skill.** The final expression is lifted out of the code,
  literals are promoted to parameters, and the result is a callable function the
  agent can invoke next time instead of re-deriving it.
* **Failure -> Pitfall.** The task context and the wrong answer are recorded so
  the agent gets warned before repeating itself.

Most frameworks only implement the first. That throws away the cheaper half of
the learning signal: failures are usually more informative than successes,
because they are specific about what does *not* work.

Both channels have a heuristic implementation (no model needed) and an
LLM-driven one for real deployments.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evolver.core.types import Pitfall, Skill, StepKind, Trajectory
from evolver.evolve.naming import name_from_text, token_set

_SKILL_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class DistillResult:
    skill: Optional[Skill] = None
    pitfall: Optional[Pitfall] = None
    reason: str = ""

    @property
    def produced(self) -> bool:
        return self.skill is not None or self.pitfall is not None


# ---------------------------------------------------------------------------
# Heuristic distillation
# ---------------------------------------------------------------------------


class _ParamPromoter(ast.NodeTransformer):
    """Replace data literals with parameter references.

    Only genuine *data* is promoted. Short strings stay inline because they are
    usually field names or flags (``'k'``, ``'v'``), not inputs -- promoting
    those would produce skills with nonsense arity.
    """

    _SEQ = ("xs", "ys", "zs", "ws")
    _MAP = ("d", "d2", "d3")
    _TXT = ("text", "text2", "text3")

    def __init__(self) -> None:
        self.names: List[str] = []
        self._seen: Dict[str, str] = {}
        self._count = {"seq": 0, "map": 0, "txt": 0}

    def _next(self, kind: str) -> str:
        pool = {"seq": self._SEQ, "map": self._MAP, "txt": self._TXT}[kind]
        i = self._count[kind]
        self._count[kind] += 1
        return pool[i] if i < len(pool) else f"{pool[0]}{i}"

    def _promote(self, node: ast.AST, kind: str) -> ast.Name:
        key = ast.dump(node)
        if key in self._seen:
            return ast.Name(id=self._seen[key], ctx=ast.Load())
        name = self._next(kind)
        self._seen[key] = name
        self.names.append(name)
        return ast.Name(id=name, ctx=ast.Load())

    def visit_List(self, node: ast.List) -> ast.AST:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            return self._promote(node, "seq")
        return self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:  # noqa: N802
        # Store-context tuples are comprehension targets (`for k, v in ...`),
        # not data. Promoting those would rewrite the loop variable.
        if isinstance(node.ctx, ast.Load):
            return self._promote(node, "seq")
        return self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> ast.AST:  # noqa: N802
        return self._promote(node, "seq")

    def visit_Dict(self, node: ast.Dict) -> ast.AST:  # noqa: N802
        return self._promote(node, "map")

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str) and len(node.value) >= 2:
            return self._promote(node, "txt")
        return node


def _final_expression(source: str) -> Optional[ast.AST]:
    """Find the expression the agent ultimately produced.

    Looks for ``result = <expr>`` or ``finish(<expr>)``, last one wins.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    found: Optional[ast.AST] = None
    from_finish: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "result":
                    found = node.value
        elif isinstance(node, ast.Call):
            fname = getattr(node.func, "id", "")
            if fname in ("finish", "final_answer") and node.args:
                arg = node.args[0]
                # `finish(result)` is just a pointer to the assignment above;
                # only take it when it carries an expression of its own.
                if not (isinstance(arg, ast.Name) and arg.id == "result"):
                    from_finish = arg
    return found if found is not None else from_finish


def _has_call(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


def heuristic_distill_skill(
    trajectory: Trajectory, max_tokens: int = 1200
) -> Optional[Skill]:
    """Lift a callable skill out of a successful trajectory."""
    if not trajectory.succeeded:
        return None

    code_steps = [s for s in trajectory.steps if s.kind is StepKind.CODE]
    if not code_steps:
        return None

    expr = None
    for step in reversed(code_steps):
        expr = _final_expression(step.content)
        if expr is not None:
            break
    if expr is None or not _has_call(expr):
        return None

    promoter = _ParamPromoter()
    try:
        promoted = promoter.visit(ast.parse(ast.unparse(expr)).body[0].value)
    except Exception:  # noqa: BLE001 - unparse can fail on exotic nodes
        return None

    try:
        body_expr = ast.unparse(promoted)
    except Exception:  # noqa: BLE001
        return None

    name = name_from_text(trajectory.task)
    params = promoter.names
    signature = f"{name}({', '.join(params)})"
    body = f"    return {body_expr}"

    if len(signature) + len(body) > max_tokens * 4:
        return None

    return Skill(
        name=name,
        description=f"Distilled from a successful run: {trajectory.task[:120]}",
        signature=signature,
        body=body,
        trigger=" ".join(sorted(token_set(trajectory.task))),
        tags=sorted(token_set(trajectory.task))[:8],
        source_trajectory=trajectory.trajectory_id,
    )


def heuristic_distill_pitfall(trajectory: Trajectory) -> Optional[Pitfall]:
    """Record what went wrong, so the next attempt gets warned."""
    if trajectory.succeeded:
        return None

    wrong = trajectory.final_answer or "(no answer)"
    reason = trajectory.failure_reason or "produced an incorrect result"
    return Pitfall(
        description=f"On '{trajectory.task[:100]}' the agent answered {wrong!r} "
                    f"and failed: {reason}",
        context=trajectory.task[:200],
        wrong_action=str(wrong)[:200],
        correct_action="",
        tags=sorted(token_set(trajectory.task))[:8],
        source_trajectory=trajectory.trajectory_id,
    )


# ---------------------------------------------------------------------------
# LLM-driven distillation
# ---------------------------------------------------------------------------

SKILL_PROMPT = """Extract a reusable Python function from this successful agent run.

Return ONLY a JSON object with keys: name, description, signature, body, trigger.
- name: snake_case identifier
- signature: e.g. "second_largest(xs)"
- body: indented function body (4 spaces), must return the answer
- trigger: 3-6 lowercase keywords that should match future similar tasks
- Use only these helpers if needed: {tools}
- No imports, no I/O, no side effects.

RUN:
{transcript}
"""

PITFALL_PROMPT = """This agent run failed. Extract a concise lesson.

Return ONLY a JSON object with keys: description, context, wrong_action,
correct_action, trigger.
- description: one sentence on what went wrong
- wrong_action: the specific mistake
- correct_action: what should have been done, if inferable
- trigger: 3-6 lowercase keywords matching this kind of task

RUN:
{transcript}
"""


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    m = _SKILL_JSON.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def llm_distill_skill(trajectory: Trajectory, adapter, tools: str = "") -> Optional[Skill]:
    """Ask a model to generalise a success into a function."""
    if not trajectory.succeeded:
        return None
    resp = adapter.complete(
        "You extract reusable functions from agent trajectories. Output JSON only.",
        SKILL_PROMPT.format(tools=tools or "(builtins)",
                            transcript=trajectory.transcript()[-6000:]),
        max_tokens=800,
    )
    if not resp.ok:
        return None
    data = _parse_json_object(resp.text)
    if not data or not all(k in data for k in ("name", "signature", "body")):
        return None
    try:
        ast.parse(f"def {data['signature']}\n{data['body']}")
    except SyntaxError:
        return None
    return Skill(
        name=str(data["name"]),
        description=str(data.get("description", ""))[:300],
        signature=str(data["signature"]),
        body=str(data["body"]),
        trigger=str(data.get("trigger", ""))[:200],
        source_trajectory=trajectory.trajectory_id,
    )


def llm_distill_pitfall(trajectory: Trajectory, adapter) -> Optional[Pitfall]:
    """Ask a model to articulate why a run failed."""
    if trajectory.succeeded:
        return None
    resp = adapter.complete(
        "You diagnose failed agent runs. Output JSON only.",
        PITFALL_PROMPT.format(transcript=trajectory.transcript()[-6000:]),
        max_tokens=500,
    )
    if not resp.ok:
        return None
    data = _parse_json_object(resp.text)
    if not data or "description" not in data:
        return None
    return Pitfall(
        description=str(data["description"])[:300],
        context=str(data.get("context", trajectory.task))[:200],
        wrong_action=str(data.get("wrong_action", ""))[:200],
        correct_action=str(data.get("correct_action", ""))[:200],
        tags=sorted(token_set(str(data.get("trigger", "")) or trajectory.task))[:8],
        source_trajectory=trajectory.trajectory_id,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class Distiller:
    """Chooses the heuristic or model-driven path automatically."""

    adapter: Optional[Any] = None
    tools: str = ""
    use_llm: bool = True

    def distill(self, trajectory: Trajectory) -> DistillResult:
        if trajectory.succeeded:
            skill = None
            if self.use_llm and self.adapter is not None:
                skill = llm_distill_skill(trajectory, self.adapter, self.tools)
            if skill is None:
                skill = heuristic_distill_skill(trajectory)
            return DistillResult(skill=skill, reason="success" if skill else "no extractable expression")

        pitfall = None
        if self.use_llm and self.adapter is not None:
            pitfall = llm_distill_pitfall(trajectory, self.adapter)
        if pitfall is None:
            pitfall = heuristic_distill_pitfall(trajectory)
        return DistillResult(pitfall=pitfall, reason="failure" if pitfall else "uninformative failure")
