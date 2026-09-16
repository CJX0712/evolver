"""Minimal end-to-end example: three runs over the same task family.

    python examples/quickstart.py

No API key needed -- the default replay adapter is deterministic and offline.
"""

from __future__ import annotations

from evolver.bench.tools import TOOLS
from evolver.core.config import Config
from evolver.core.engine import Evolver


def main() -> None:
    cfg = Config()
    cfg.store_path = ".evolver/example-skills.json"

    engine = Evolver(config=cfg, tools=list(TOOLS))

    tasks = [
        "Find the second largest value in [5, 5, 4, 3].",
        "Find the second largest value in [9, 9, 8, 2, 7].",
        "Find the second largest value in [4, 4, 4, 1].",
    ]

    print(f"{'task':52s} {'status':>9s} {'steps':>6s} {'tokens':>7s}")
    print("-" * 78)

    for i, task in enumerate(tasks, 1):
        traj = engine.run(task, task_id=f"demo-{i}")
        used = traj.metadata.get("skills_used", [])
        print(f"{task[:52]:52s} {traj.status.value:>9s} {traj.n_steps:6d} "
              f"{traj.total_tokens:7d}")
        if used:
            print(f"{'':52s}   reused skill: {', '.join(used)}")

    print("-" * 78)
    stats = engine.stats()
    print(f"skills={stats['skills']}  pitfalls={stats['pitfalls']}  "
          f"tokens_saved={stats['tokens_saved']}")
    engine.save()
    print(f"\nlibrary written to {cfg.store_path}")


if __name__ == "__main__":
    main()
