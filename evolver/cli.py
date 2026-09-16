"""Command line interface.

    evolver bench                       # run the built-in benchmark
    evolver bench --provider openai     # ... against a real model
    evolver run "task text"             # one task through the engine
    evolver skills                      # inspect the learned library
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


PROVIDERS = ["replay", "openai", "anthropic", "kimi", "openrouter"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evolver",
        description="Evolver -- the self-evolving layer for AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- bench -----------------------------------------------------------
    b = sub.add_parser("bench", help="run the evolution benchmark")
    b.add_argument("--epochs", type=int, default=5)
    b.add_argument("--tasks", type=int, default=0, help="limit task count (0 = all)")
    b.add_argument("--provider", default="replay", choices=PROVIDERS)
    b.add_argument("--model", default=None)
    b.add_argument("--base-url", default=None,
                   help="custom OpenAI-compatible endpoint")
    b.add_argument("--api-key-env", default=None,
                   help="env var holding the API key")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--max-steps", type=int, default=8)
    b.add_argument("--store", default=".evolver/skills.json")
    b.add_argument("--out", default="evolver-report.html")
    b.add_argument("--json", dest="json_out", default=None)
    b.add_argument("--no-evolve", action="store_true",
                   help="disable distillation (control run)")
    b.add_argument("--verbose", "-v", action="store_true")

    # -- run -------------------------------------------------------------
    r = sub.add_parser("run", help="run a single task through the engine")
    r.add_argument("task")
    r.add_argument("--provider", default="replay", choices=PROVIDERS)
    r.add_argument("--model", default=None)
    r.add_argument("--base-url", default=None)
    r.add_argument("--api-key-env", default=None)
    r.add_argument("--store", default=".evolver/skills.json")
    r.add_argument("--max-steps", type=int, default=8)
    r.add_argument("--verbose", "-v", action="store_true")

    # -- skills ----------------------------------------------------------
    s = sub.add_parser("skills", help="inspect the learned library")
    s.add_argument("--store", default=".evolver/skills.json")
    s.add_argument("--top", type=int, default=20)

    # -- init ------------------------------------------------------------
    i = sub.add_parser("init", help="write a default config file")
    i.add_argument("--path", default="evolver.json")
    return p


def _config_from_args(args) -> "Config":
    from evolver.core.config import Config

    cfg = Config.for_provider(args.provider, model=getattr(args, "model", None))
    cfg.store_path = getattr(args, "store", ".evolver/skills.json")
    if getattr(args, "base_url", None):
        cfg.llm.base_url = args.base_url
    if getattr(args, "api_key_env", None):
        cfg.llm.api_key_env = args.api_key_env
    if getattr(args, "no_evolve", False):
        cfg.evolve.distill_enabled = False
    return cfg


def _cmd_bench(args) -> int:
    from evolver.bench.report import write_report
    from evolver.bench.runner import run_benchmark
    from evolver.bench.tasks import BENCH_TASKS

    cfg = _config_from_args(args)
    tasks = list(BENCH_TASKS)
    if args.tasks:
        tasks = tasks[: args.tasks]

    print(f"evolver: {len(tasks)} tasks x {args.epochs} epochs "
          f"via {args.provider}" + (" (evolution OFF)" if args.no_evolve else ""))

    result = run_benchmark(cfg, tasks=tasks, epochs=args.epochs, seed=args.seed,
                           max_steps=args.max_steps, verbose=args.verbose)

    print("\n--- summary ---")
    for k, v in result.summary().items():
        print(f"  {k:20s} {v}")

    path = write_report(result, args.out)
    print(f"\nreport -> {path}")

    if args.json_out:
        payload = {
            "summary": result.summary(),
            "epochs": [e.to_dict() for e in result.epochs],
            "skills": result.skill_names,
            "config": result.config,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"json    -> {args.json_out}")
    return 0


def _cmd_run(args) -> int:
    from evolver.bench.tools import TOOLS
    from evolver.core.engine import Evolver

    cfg = _config_from_args(args)
    engine = Evolver(config=cfg, tools=list(TOOLS), max_steps=args.max_steps)
    traj = engine.run(args.task, max_steps=args.max_steps)

    print(f"status : {traj.status.value}")
    print(f"answer : {traj.final_answer!r}")
    if traj.failure_reason:
        print(f"reason : {traj.failure_reason}")
    print(f"steps  : {traj.n_steps}   tokens: {traj.total_tokens}")
    if args.verbose:
        print("\n" + traj.transcript())
    engine.save()
    return 0 if traj.succeeded else 1


def _cmd_skills(args) -> int:
    from evolver.evolve.store import SkillStore

    store = SkillStore.load(args.store)
    stats = store.stats()
    print(f"runs={stats['runs']}  skills={stats['skills']}  "
          f"pitfalls={stats['pitfalls']}  tokens_saved={stats['tokens_saved']}")
    if not store.skills:
        print("(empty -- run `evolver bench` to grow a library)")
        return 0
    print(f"\n{'skill':34s} {'uses':>5s} {'succ':>6s} {'score':>7s}  signature")
    for s in store.top(args.top):
        print(f"{s.name[:34]:34s} {s.stats.uses:5d} {s.stats.success_rate:6.0%} "
              f"{s.score:7.3f}  {s.signature}")
    return 0


def _cmd_init(args) -> int:
    from evolver.core.config import Config

    cfg = Config()
    cfg.to_json(args.path)
    print(f"wrote {args.path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {"bench": _cmd_bench, "run": _cmd_run,
                "skills": _cmd_skills, "init": _cmd_init}
    return handlers[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
