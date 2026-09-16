"""Context must stay bounded without losing the goal or recent reasoning."""

from __future__ import annotations

import pytest

from evolver.core.config import ContextConfig
from evolver.core.types import Step, StepKind
from evolver.memory.context import ContextWindow, heuristic_summarize
from evolver.memory.tokens import count_tokens, truncate_to_tokens


def _step(i: int, text: str = "x" * 400) -> Step:
    return Step(kind=StepKind.CODE, content=f"# step {i}\n{text}",
                observation="ok", tokens_in=100, tokens_out=20)


def test_resident_always_survives_compaction():
    cfg = ContextConfig(max_tokens=800, keep_recent_steps=1)
    ctx = ContextWindow(config=cfg)
    ctx.set_resident("GOAL: find the answer")
    for i in range(12):
        ctx.add(_step(i))
    assert "find the answer" in ctx.render()


def test_compaction_keeps_recent_steps_verbatim():
    cfg = ContextConfig(max_tokens=800, keep_recent_steps=2)
    ctx = ContextWindow(config=cfg)
    for i in range(10):
        ctx.add(_step(i))
    rendered = ctx.render()
    assert "step 9" in rendered
    assert "step 8" in rendered
    # Compaction is threshold-driven, not per-step, so the working set hovers
    # just above `keep` rather than being pinned to it exactly.
    assert len(ctx.steps) <= cfg.keep_recent_steps + 2
    assert ctx.archive, "older steps should have been archived, not dropped"
    assert ctx.compactions > 0


def test_token_budget_is_respected():
    cfg = ContextConfig(max_tokens=1500, keep_recent_steps=3)
    ctx = ContextWindow(config=cfg)
    ctx.set_resident("short goal")
    for i in range(40):
        ctx.add(_step(i))
    assert ctx.tokens <= cfg.max_tokens


def test_does_not_compact_below_threshold():
    cfg = ContextConfig(max_tokens=100_000, keep_recent_steps=10)
    ctx = ContextWindow(config=cfg)
    for i in range(5):
        ctx.add(_step(i, text="tiny"))
    assert ctx.compactions == 0
    assert len(ctx.steps) == 5


def test_usage_reports_ratio():
    ctx = ContextWindow(config=ContextConfig(max_tokens=1000))
    ctx.set_resident("goal")
    used, budget, ratio = ctx.usage()
    assert budget == 1000
    assert 0 <= ratio <= 1


def test_summarize_captures_errors():
    s = Step(kind=StepKind.CODE, content="result = 1/0", error="ZeroDivisionError")
    assert "ZeroDivisionError" in heuristic_summarize(s)


def test_summarize_skips_boilerplate():
    s = Step(kind=StepKind.CODE, content="import json\n# a comment\nresult = merge(a, b)")
    out = heuristic_summarize(s)
    assert "import" not in out and "comment" not in out
    assert "merge(a, b)" in out


def test_token_counting_is_monotonic():
    assert count_tokens("") == 0
    assert count_tokens("hello") < count_tokens("hello world, this is longer")
    assert count_tokens("中文字符") >= 4


def test_truncate_respects_budget():
    text = "word " * 500
    out = truncate_to_tokens(text, 50)
    assert count_tokens(out) <= 60
