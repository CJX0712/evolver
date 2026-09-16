"""Benchmark runner and HTML reporting."""

from evolver.bench.runner import BenchmarkResult, EpochRecord, run_benchmark
from evolver.bench.tasks import BENCH_TASKS, BenchTask, by_id, find_task
from evolver.bench.tools import TOOLS

__all__ = [
    "BENCH_TASKS",
    "BenchmarkResult",
    "BenchTask",
    "EpochRecord",
    "TOOLS",
    "by_id",
    "find_task",
    "run_benchmark",
]
