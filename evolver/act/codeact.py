"""CodeAct: express a whole tool-using plan as one Python block.

The standard agent loop is a token-hungry ping-pong match:

    model -> pick one tool -> wait -> observe -> model -> pick next tool -> ...

Every arrow is a full model turn, and each turn re-sends the entire history.
When a task needs six tool calls, you pay for six round trips and six copies of
the context.

CodeAct inverts it. The model writes a short program that calls the tools it
needs, the program runs once in the sandbox, and one consolidated result comes
back. Six tool calls become one turn. This is the single largest lever on both
latency and token cost, and it composes cleanly with skills -- a distilled skill
is just a function the generated code can call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evolver.act.sandbox import ExecutionResult, Sandbox, validate_source
from evolver.core.config import SandboxConfig
from evolver.core.types import Skill, ToolSpec

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)

SYSTEM_PROMPT = """You are an agent that solves tasks by writing Python code.

Write a single Python block that calls the available tools. Rules:
- Use the provided tool functions directly. Do not redefine them.
- Call `finish(value)` with the final answer when done. That ends the episode.
- Prefer computing over narrating: loops, conditionals, and aggregation are free.
- Print intermediate results only if they help you decide what to do next.
- Available stdlib imports: {imports}
- Never use: open, eval, exec, import of anything not listed above.

Respond with exactly one fenced code block:

```python
# your code here
```
"""

TASK_TEMPLATE = """TASK:
{task}

TOOLS:
{tools}

{skills_block}
AVAILABLE STATE:
{state}

Write the Python block that accomplishes the task."""


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a model response.

    Falls back to the raw text when the model forgets the fence -- a common and
    cheap-to-handle failure.
    """
    m = _CODE_FENCE.search(text or "")
    if m:
        return m.group(1).strip()
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def render_tools(tools: List[ToolSpec]) -> str:
    return "\n".join(t.render() for t in tools)


def render_skills(skills: List[Skill]) -> str:
    if not skills:
        return ""
    blocks = [
        "REUSABLE SKILLS (call these directly instead of re-reasoning):",
        *[f"\n# {s.name} — {s.description}\n{s.code}\n" for s in skills],
    ]
    return "\n".join(blocks)


@dataclass
class CodeActExecutor:
    """Runs one CodeAct turn: render prompt -> model -> sandbox -> result."""

    sandbox: Sandbox = field(default_factory=Sandbox)
    config: SandboxConfig = field(default_factory=SandboxConfig)

    def __init__(self, tools: Optional[List[ToolSpec]] = None,
                 config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self.sandbox = Sandbox(config=self.config)
        self.tools: List[ToolSpec] = list(tools or [])
        for t in self.tools:
            if t.fn is not None:
                self.sandbox.register(t.name, t.fn)

    # -- prompt ----------------------------------------------------------
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(imports=", ".join(self.config.allow_imports))

    def build_prompt(self, task: str, skills: Optional[List[Skill]] = None,
                     state: Optional[Dict[str, Any]] = None,
                     warnings: Optional[List[str]] = None) -> str:
        prompt = TASK_TEMPLATE.format(
            task=task,
            tools=render_tools(self.tools),
            skills_block=render_skills(skills or []),
            state=self._render_state(state or {}),
        )
        if warnings:
            prompt += "\n\nKNOWN PITFALLS FOR THIS KIND OF TASK:\n" + "\n".join(
                f"- {w}" for w in warnings
            )
        return prompt

    @staticmethod
    def _render_state(state: Dict[str, Any]) -> str:
        if not state:
            return "(empty)"
        return "\n".join(f"{k} = {v!r}" for k, v in state.items())

    # -- execution -------------------------------------------------------
    def inject_skills(self, skills: List[Skill]) -> List[str]:
        """Bind distilled skills into the namespace as callable functions.

        Skill bodies originate from model output, so they go through the same
        static vetting as any other generated code. Returns the names bound.
        """
        bound: List[str] = []
        for s in skills:
            try:
                validate_source(s.code, self.config)
                # Skills are defined into a namespace that already holds the
                # tools. A function resolves free names through the globals it
                # was *defined* in, not the namespace it is later called from --
                # so defining skills in an empty namespace would leave every
                # tool call inside a skill raising NameError.
                ns: Dict[str, Any] = dict(self.sandbox.tools)
                ns["__builtins__"] = __builtins__
                shadowed = {n: ns[n] for n in self.sandbox.tools if n in ns and n == s.name}
                exec(compile(s.code, f"<skill:{s.name}>", "exec"), ns)  # noqa: S102
                # A skill can be distilled with the same name as a tool
                # (`total`). Left alone, its body would call itself forever.
                # Restore the tool inside the skill's own globals so the body
                # resolves to the primitive.
                for n, fn in shadowed.items():
                    ns[n] = fn
                fn2 = ns.get(s.name)
                fn = fn2
                if callable(fn):
                    self.sandbox.register(s.name, fn)
                    bound.append(s.name)
            except Exception:  # noqa: BLE001 - a broken skill must not kill the run
                continue
        return bound

    def execute(self, code: str) -> ExecutionResult:
        return self.sandbox.run(extract_code(code))

    def observe(self, result: ExecutionResult) -> str:
        """Turn an execution result into the text the model sees next."""
        parts: List[str] = []
        if result.output:
            parts.append(result.output.rstrip())
        if result.value is not None:
            parts.append(f"result = {result.value!r}")
        if result.error:
            parts.append(f"ERROR: {result.error}")
        if result.timed_out:
            parts.append("ERROR: execution timed out")
        if not parts:
            parts.append("(no output)")
        return "\n".join(parts)[: self.config.max_output_chars]
