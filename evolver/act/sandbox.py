"""A constrained execution sandbox for model-generated Python.

Threat model
------------
The code being executed is written by a language model. It is not trusted, and
it is not merely "possibly buggy" -- a prompt-injected document can steer the
model into emitting hostile code. So execution is gated by four layers:

1. **Static AST allowlist** -- the module is parsed and every node checked
   against an explicit allowlist before a single byte runs. Dunder attribute
   access (the classic ``().__class__.__bases__`` sandbox escape) is rejected,
   as are ``__import__``, ``eval``, ``exec``, ``open``, ``compile``, and
   friends.
2. **Restricted builtins** -- globals get a curated builtin dict, not the real
   one. There is no ``__builtins__`` to climb back out through.
3. **Wall-clock timeout** -- ``SIGALRM`` on POSIX aborts runaway loops.
4. **Address-space rlimit** -- caps memory so an allocator bomb can't take the
   host down.

Defence in depth matters: any single layer has known bypasses, but the
combination makes escape impractical rather than merely unlikely.
"""

from __future__ import annotations

import ast
import io
import resource
import signal
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from evolver.core.config import SandboxConfig
from evolver.core.types import ToolSpec


class SandboxError(Exception):
    """Raised when code is rejected or misbehaves inside the sandbox."""


class SecurityViolation(SandboxError):
    """Static analysis rejected the code before execution."""


class TimeoutError_(SandboxError):
    """Execution exceeded the wall-clock budget."""


# Names that must never appear, even as an attribute or a bare identifier.
BANNED_NAMES = frozenset(
    {
        "__import__",
        "__builtins__",
        "__globals__",
        "__dict__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__code__",
        "__reduce__",
        "__loader__",
        "__spec__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "memoryview",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "super",
        "object",
        "type",
        "help",
        "exit",
        "quit",
        "__enter__",
        "__exit__",
        "mro",
    }
)

# Statement / expression node types we permit.
ALLOWED_NODES = frozenset(
    {
        # module & statements
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Return,
        ast.Pass,
        ast.Break,
        ast.Continue,
        ast.Assert,
        ast.Raise,
        ast.Delete,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.Match,
        ast.match_case,
        ast.MatchValue,
        ast.MatchAs,
        ast.MatchOr,
        ast.MatchClass,
        ast.MatchStar,
        ast.MatchMapping,
        ast.MatchSequence,
        # expressions
        ast.BoolOp,
        ast.NamedExpr,
        ast.BinOp,
        ast.UnaryOp,
        ast.Lambda,
        ast.IfExp,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Compare,
        ast.Call,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.Constant,
        ast.Attribute,
        ast.Subscript,
        ast.Starred,
        ast.Name,
        ast.List,
        ast.Tuple,
        ast.Slice,
        # operators
        ast.Load,
        ast.Store,
        ast.Del,
        ast.And,
        ast.Or,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.MatMult,
        ast.Invert,
        ast.Not,
        ast.UAdd,
        ast.USub,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        # comprehension / arg plumbing
        ast.comprehension,
        ast.arguments,
        ast.arg,
        ast.keyword,
        ast.alias,
        ast.withitem,
        # py3.12+ type params (harmless, but must be allowlisted if present)
        getattr(ast, "TypeVar", type(None)),
        getattr(ast, "TypeVarTuple", type(None)),
        getattr(ast, "ParamSpec", type(None)),
        getattr(ast, "TypeAlias", type(None)),
    }
)

# Builtins safe to hand to model code.
SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


@dataclass
class ExecutionResult:
    """Outcome of running one block of model-generated code."""

    output: str = ""
    value: Any = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    timed_out: bool = False
    tool_calls: int = 0
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out

    @property
    def finished(self) -> bool:
        return self.final_answer is not None


def validate_source(source: str, cfg: SandboxConfig) -> ast.Module:
    """Parse and statically vet ``source``. Raises :class:`SecurityViolation`."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SecurityViolation(f"syntax error: {exc}") from exc

    allowed_imports = set(cfg.allow_imports)

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise SecurityViolation(
                f"disallowed syntax: {type(node).__name__} "
                f"(line {getattr(node, 'lineno', '?')})"
            )

        # Bare identifiers: catch `eval`, `__import__`, ...
        if isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                raise SecurityViolation(f"disallowed name: {node.id!r}")

        # Attribute access: block dunders and private internals.
        if isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES or node.attr.startswith("__"):
                raise SecurityViolation(f"disallowed attribute: {node.attr!r}")

        # Function names shadowing banned builtins.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in BANNED_NAMES or node.name.startswith("__"):
                raise SecurityViolation(f"disallowed def: {node.name!r}")

        # Imports: allowlist only, and `from x import *` is never allowed.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_imports:
                    raise SecurityViolation(f"disallowed import: {alias.name!r}")
        if isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".")[0] not in allowed_imports:
                raise SecurityViolation(f"disallowed import from: {node.module!r}")
            for alias in node.names:
                if alias.name == "*":
                    raise SecurityViolation("star imports are not allowed")

    return tree


def make_safe_import(allowed: Iterable[str]):
    """Build an ``__import__`` that enforces the allowlist at runtime too.

    Static analysis already rejects bad imports, but code can also reach
    ``__import__`` indirectly. Enforcing here as well means one layer being
    bypassed does not open the door.
    """
    permitted = set(allowed)

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = (name or "").split(".")[0]
        if root not in permitted:
            raise ImportError(f"import of {name!r} is not allowed in the sandbox")
        return __import__(name, globals, locals, fromlist, level)

    return _safe_import


class _Timeout:
    """POSIX SIGALRM-based wall clock. No-op where unsupported."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._supported = hasattr(signal, "SIGALRM")

    def __enter__(self) -> "_Timeout":
        if self._supported and self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc) -> bool:
        if self._supported:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return False


def _install_alarm_handler() -> None:
    if not hasattr(signal, "SIGALRM"):
        return

    def _handler(signum, frame):  # pragma: no cover - timing dependent
        raise TimeoutError_("execution timed out")

    signal.signal(signal.SIGALRM, _handler)


_install_alarm_handler()


@dataclass
class Sandbox:
    """Executes untrusted Python with tools injected as globals."""

    config: SandboxConfig = field(default_factory=SandboxConfig)
    tools: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._call_counter = 0

    # -- tool plumbing ---------------------------------------------------
    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.tools[name] = fn

    def register_many(self, tools: Iterable[ToolSpec]) -> None:
        for t in tools:
            if t.fn is not None:
                self.register(t.name, t.fn)

    def _wrap_tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Instrument tools so we can count CodeAct batching wins."""

        def _counted(*args: Any, **kwargs: Any) -> Any:
            self._call_counter += 1
            return fn(*args, **kwargs)

        return _counted

    # -- execution -------------------------------------------------------
    def run(self, source: str, extra_globals: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        import time

        started = time.perf_counter()
        result = ExecutionResult()
        self._call_counter = 0

        if not self.config.enabled:
            result.error = "sandbox disabled"
            return result

        try:
            validate_source(source, self.config)
        except SecurityViolation as exc:
            result.error = f"SecurityViolation: {exc}"
            result.duration_ms = (time.perf_counter() - started) * 1000
            return result

        # Build the execution namespace.
        builtins_ns = dict(SAFE_BUILTINS)
        builtins_ns["__import__"] = make_safe_import(self.config.allow_imports)
        sandbox_globals: Dict[str, Any] = {
            "__builtins__": builtins_ns,
            "__name__": "sandbox",
        }
        for name, fn in self.tools.items():
            sandbox_globals[name] = self._wrap_tool(fn)
        if extra_globals:
            for k, v in extra_globals.items():
                if not k.startswith("__"):
                    sandbox_globals[k] = v

        captured: Dict[str, Any] = {}

        def _finish(value: Any = None) -> None:
            captured["final"] = "" if value is None else str(value)

        sandbox_globals["finish"] = _finish
        sandbox_globals["final_answer"] = _finish

        buf = io.StringIO()
        try:
            with redirect_stdout(buf), _Timeout(self.config.timeout_s):
                try:
                    resource.setrlimit(
                        resource.RLIMIT_AS,
                        (self.config.max_memory_mb * 1024 * 1024,) * 2,
                    )
                except (ValueError, OSError, AttributeError):
                    pass  # best effort; not all platforms honour this
                try:
                    exec(compile(source, "<evolver>", "exec"), sandbox_globals)  # noqa: S102
                except TimeoutError_ as exc:
                    result.timed_out = True
                    result.error = str(exc)
                except Exception as exc:  # noqa: BLE001 - reported to the model
                    result.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            output = buf.getvalue()
            if len(output) > self.config.max_output_chars:
                cut = self.config.max_output_chars
                output = output[:cut] + f"\n...[truncated {len(output) - cut} chars]"
            result.output = output
            result.tool_calls = self._call_counter
            result.duration_ms = (time.perf_counter() - started) * 1000

        if "final" in captured:
            result.final_answer = captured["final"]
        else:
            result.value = sandbox_globals.get("result")

        return result
