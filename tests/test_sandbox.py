"""Sandbox security is the load-bearing wall -- if this fails, nothing else matters."""

from __future__ import annotations

import pytest

from evolver.act.sandbox import (
    ExecutionResult,
    Sandbox,
    SandboxConfig,
    SecurityViolation,
    validate_source,
)


@pytest.fixture
def sandbox() -> Sandbox:
    sb = Sandbox(config=SandboxConfig())
    sb.register("double", lambda x: x * 2)
    return sb


# -- happy path ----------------------------------------------------------


def test_executes_code_and_captures_output(sandbox):
    r = sandbox.run("print('hi')\nresult = double(21)")
    assert r.ok
    assert "hi" in r.output
    assert r.value == 42


def test_finish_sets_final_answer(sandbox):
    r = sandbox.run("finish(double(4))")
    assert r.final_answer == "8"


def test_counts_tool_calls(sandbox):
    r = sandbox.run("a = double(1)\nb = double(2)\nc = double(3)")
    assert r.tool_calls == 3


def test_reports_runtime_errors(sandbox):
    r = sandbox.run("1 / 0")
    assert not r.ok
    assert "ZeroDivisionError" in r.error


def test_allows_allowlisted_imports(sandbox):
    r = sandbox.run("import math\nresult = math.sqrt(16)")
    assert r.ok and r.value == 4.0


def test_truncates_excessive_output():
    sb = Sandbox(config=SandboxConfig(max_output_chars=100))
    r = sb.run("print('x' * 5000)")
    assert len(r.output) < 200
    assert "truncated" in r.output


# -- static rejection ----------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "print(().__class__.__bases__)",
        "print(''.__class__.__mro__)",
        "import os",
        "import subprocess",
        "from os import path",
        "from json import *",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', '<s>', 'eval')",
        "__import__('os')",
        "print(globals())",
        "print(locals())",
        "x = getattr([], 'append')",
        "print([].__dict__)",
        "def __dunder(): pass",
        "print((1).__class__)",
    ],
)
def test_rejects_dangerous_code(sandbox, code):
    r = sandbox.run(code)
    assert r.error is not None
    assert "SecurityViolation" in r.error


def test_rejects_syntax_errors(sandbox):
    with pytest.raises(SecurityViolation):
        validate_source("def (:", SandboxConfig())


def test_validate_accepts_reasonable_code():
    tree = validate_source(
        "import math\n"
        "def f(xs):\n"
        "    return [math.sqrt(x) for x in xs]\n"
        "result = f([1, 4, 9])\n",
        SandboxConfig(),
    )
    assert tree is not None


# -- resource limits -----------------------------------------------------


def test_kills_infinite_loop():
    sb = Sandbox(config=SandboxConfig(timeout_s=0.5))
    r = sb.run("while True:\n    pass")
    assert r.timed_out or "timed out" in (r.error or "")


def test_disabled_sandbox_refuses():
    sb = Sandbox(config=SandboxConfig(enabled=False))
    r = sb.run("print(1)")
    assert not r.ok


# -- isolation -----------------------------------------------------------


def test_no_state_leaks_between_runs(sandbox):
    sandbox.run("leaky = 123")
    r = sandbox.run("print(leaky)")
    assert not r.ok  # NameError, not the previous value


def test_tools_are_callable_but_not_inspectable(sandbox):
    r = sandbox.run("print(double.__globals__)")
    assert not r.ok
