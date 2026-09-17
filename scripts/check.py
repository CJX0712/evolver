#!/usr/bin/env python3.11
"""Local quality gate -- the same checks the dropped CI workflow ran.

Run before every push:  python3.11 scripts/check.py

Three stages, cheapest first:

  1. import     -- the package imports and exposes its public surface
  2. tests      -- the full pytest suite
  3. regression -- evolution still improves, measured against a control run

Stage 3 is the one that matters. A test suite proves the code does what the
tests say; it cannot notice that the *learning* stopped working. The benchmark
asserts two things that must hold if the mechanism is alive:

    success rate rises   (pitfalls are being learned)
    token cost falls     (skills are being learned)

and asserts them against a no-evolve control, so a rise caused by repetition
alone is not enough to pass.

Exit code is the number of failed stages, so this is usable as a pre-push hook.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable or "python3.11"

# Epochs are deliberately modest. This runs on every commit; a gate that takes
# ten minutes is a gate people learn to skip.
EPOCHS = 4
TASKS = 10


class Stage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok = False
        self.detail = ""
        self.seconds = 0.0


def _run(cmd: list[str], timeout: int = 900) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def stage_import() -> Stage:
    s = Stage("import")
    code, out = _run([
        PY, "-c",
        "import evolver; from evolver import Evolver, Config; "
        "from evolver.evolve.pack import build_pack, import_pack; "
        "print('public surface ok', evolver.__version__ if hasattr(evolver,'__version__') else '')",
    ])
    s.ok = code == 0
    s.detail = out.splitlines()[-1] if out else ""
    return s


def stage_tests() -> Stage:
    s = Stage("tests")
    # COLUMNS is pinned wide so pytest does not wrap the progress line: with the
    # default 80 columns, 152 tests wrap and the final line carries only a
    # handful of dots, which makes counting them a lie.
    env = dict(os.environ, COLUMNS="400")
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         "--tb=short", "-W", "ignore::UserWarning"],
        cwd=ROOT, capture_output=True, text=True, timeout=900, env=env,
    )
    out = proc.stdout + proc.stderr
    s.ok = proc.returncode == 0
    if s.ok:
        line = next((ln for ln in out.splitlines() if "[100%]" in ln), "")
        s.detail = f"{sum(c in '.sx' for c in line.split('[')[0])} passed"
    else:
        bad = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("FAILED")]
        s.detail = "\n      ".join(bad[:5]) or "pytest exited non-zero"
    return s


def stage_regression() -> Stage:
    """Evolution must still improve, and must beat a no-evolve control.

    Written as a subprocess rather than an import so a crash inside the
    benchmark cannot take the gate down with it -- and so the failure reads
    as a stack trace on stderr, not a swallowed exception.
    """
    s = Stage("regression")
    script = f"""
from evolver.bench.runner import run_benchmark
from evolver.bench.tasks import BENCH_TASKS
from evolver.core.config import Config, EvolveConfig

tasks = list(BENCH_TASKS)[:{TASKS}]
N = {EPOCHS}

# store_path="" keeps the gate from reading or writing .evolver/skills.json.
# A gate that silently resumes a previous run would measure nothing.
evo = run_benchmark(Config(store_path=""), tasks=tasks, epochs=N,
                    max_steps=6, verbose=False)
ctl = run_benchmark(Config(store_path="",
                           evolve=EvolveConfig(distill_enabled=False)),
                    tasks=tasks, epochs=N, max_steps=6, verbose=False)

f, l = evo.first, evo.last
cf, cl = ctl.first, ctl.last

print(f"  evolve   e0 {{f.success_rate:5.0%}} {{f.avg_tokens:6.0f}}tok "
      f"-> e{{N-1}} {{l.success_rate:5.0%}} {{l.avg_tokens:6.0f}}tok")
print(f"  control  e0 {{cf.success_rate:5.0%}} {{cf.avg_tokens:6.0f}}tok "
      f"-> e{{N-1}} {{cl.success_rate:5.0%}} {{cl.avg_tokens:6.0f}}tok")

failures = []
if not l.success_rate > f.success_rate:
    failures.append(
        f"success rate did not improve: {{f.success_rate:.0%}} -> {{l.success_rate:.0%}}")
if not l.avg_tokens < f.avg_tokens:
    failures.append(
        f"token cost did not fall: {{f.avg_tokens:.0f}} -> {{l.avg_tokens:.0f}}")
if not l.success_rate >= cl.success_rate:
    failures.append(
        f"evolve ({{l.success_rate:.0%}}) did not beat control ({{cl.success_rate:.0%}}) -- "
        f"any gain may be repetition, not learning")

if failures:
    for x in failures:
        print("  FAIL:", x)
    raise SystemExit(1)
print("  evolution intact; matched or beat the control")
"""
    code, out = _run([PY, "-c", script])
    s.ok = code == 0
    s.detail = out.replace("\n", "\n      ")
    return s


def main() -> int:
    print(f"evolver check -- {ROOT}\n")
    stages = [stage_import, stage_tests, stage_regression]
    failed = 0

    for make in stages:
        t0 = time.perf_counter()
        s = make()
        s.seconds = time.perf_counter() - t0
        mark = "PASS" if s.ok else "FAIL"
        print(f"[{mark}] {s.name:<11} {s.seconds:6.1f}s")
        if s.detail:
            print(f"      {s.detail}")
        print()
        if not s.ok:
            failed += 1
            # Stop at the first failure: later stages assume earlier ones held,
            # so their output would be noise about a problem already reported.
            break

    if failed:
        print("check FAILED -- do not push")
    else:
        print("check PASSED -- safe to push")
    return failed


if __name__ == "__main__":
    sys.exit(main())
