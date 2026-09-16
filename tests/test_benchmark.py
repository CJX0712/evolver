"""End-to-end: does the agent actually get better, and is that attributable?"""

from __future__ import annotations

import pytest

from evolver.bench.report import render_html
from evolver.bench.runner import run_benchmark
from evolver.bench.tasks import BENCH_TASKS, find_task
from evolver.core.config import Config
from evolver.core.types import RunStatus


def _cfg(evolve: bool = True) -> Config:
    c = Config()
    c.evolve.distill_enabled = evolve
    return c


def test_task_set_internally_consistent():
    """Ground truth must actually be reachable with the provided tools."""
    from evolver.bench import tools

    ns = tools.as_dict()
    for t in BENCH_TASKS:
        # Every task's warned (trap-aware) path must produce the expected answer.
        code = t.warned_final.format(**{f"a{i}": a for i, a in enumerate(t.args)})
        local: dict = {}
        exec(compile(code.replace("finish(", "local.setdefault('answer', "),
                     "<t>", "exec"), {**ns, "local": local})
        assert t.verify(local.get("answer")), f"{t.task_id}: warned path is wrong"


def test_each_task_matches_its_own_spec():
    for t in BENCH_TASKS:
        assert find_task(t.prompt) is t, f"{t.task_id} did not match itself"


def test_task_markers_are_unique():
    """A shared marker would let one skill be misapplied to another task."""
    markers = [t.skill_marker for t in BENCH_TASKS]
    for m in markers:
        owners = [s for s in BENCH_TASKS if m in s.skill_name]
        assert len(owners) == 1, f"marker {m!r} is ambiguous: {[o.task_id for o in owners]}"


def test_evolution_improves_success_and_cost():
    r = run_benchmark(_cfg(evolve=True), epochs=4)
    assert r.first.success_rate < r.last.success_rate
    assert r.last.avg_steps < r.first.avg_steps
    assert r.last.avg_tokens < r.first.avg_tokens
    assert r.skill_names, "evolution should have produced skills"


def test_control_run_does_not_improve():
    """Without evolution the curve is flat -- the gain is the mechanism, not repetition."""
    r = run_benchmark(_cfg(evolve=False), epochs=4)
    assert r.first.success_rate == r.last.success_rate
    assert abs(r.last.avg_tokens - r.first.avg_tokens) < 1


def test_skill_hit_rate_reaches_saturation():
    r = run_benchmark(_cfg(evolve=True), epochs=4)
    assert r.last.skill_hit_rate > 0.8


def test_benchmark_is_deterministic():
    a = run_benchmark(_cfg(), epochs=3, seed=7)
    b = run_benchmark(_cfg(), epochs=3, seed=7)
    assert [e.avg_tokens for e in a.epochs] == [e.avg_tokens for e in b.epochs]


def test_html_report_is_self_contained(tmp_path):
    r = run_benchmark(_cfg(), epochs=3)
    html = render_html(r)
    assert html.startswith("<!DOCTYPE html>")
    assert "http://" not in html and "https://" not in html, "no external resources"
    assert "<script" not in html
    assert "SIMULATED" in html
    out = tmp_path / "report.html"
    out.write_text(html, encoding="utf-8")
    assert out.stat().st_size > 2000


def test_context_stays_within_budget():
    r = run_benchmark(_cfg(), epochs=3)
    budget = Config().context.max_tokens
    for rec in r.records:
        for t in rec.trajectories:
            ctx = t.metadata.get("context", {})
            if ctx:
                assert ctx.get("tokens", 0) <= budget * 1.5
