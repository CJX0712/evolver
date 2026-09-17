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
    b.add_argument("--resume", action="store_true",
                   help="start from the library already at --store instead of empty "
                        "(without this, a stale store cannot leak into the baseline)")
    b.add_argument("--no-save", action="store_true",
                   help="do not write the learned library back to --store")
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

    # -- pack ------------------------------------------------------------
    pk = sub.add_parser(
        "pack", help="export/import a portable skill pack",
        description="Move an evolved skill library between machines or people.",
    )
    pk_sub = pk.add_subparsers(dest="pack_cmd", required=True)

    ex = pk_sub.add_parser("export", help="write a skill pack")
    ex.add_argument("--store", default=".evolver/skills.json")
    ex.add_argument("--out", default="skillpack.json")
    ex.add_argument("--name", default="skillpack")
    ex.add_argument("--author", default="")
    ex.add_argument("--description", default="")
    ex.add_argument("--min-uses", type=int, default=0,
                    help="only export skills with at least this many uses")
    ex.add_argument("--only-proven", action="store_true",
                    help="only export skills with off-task wins (real transfer)")

    im = pk_sub.add_parser("import", help="merge a skill pack into a store")
    im.add_argument("path")
    im.add_argument("--store", default=".evolver/skills.json")
    im.add_argument("--no-discount", action="store_true",
                    help="trust imported stats at face value (not recommended)")
    im.add_argument("--dry-run", action="store_true",
                    help="report what would happen without writing")

    ins = pk_sub.add_parser("inspect", help="summarise a pack without importing")
    ins.add_argument("path")

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
    if args.resume and args.no_save:
        # These two are contradictory: --resume means "read the library at
        # --store", --no-save means "write nothing there". Silently blanking
        # store_path first would turn --resume into a no-op and quietly
        # report a cold run as if it had resumed, so refuse instead.
        print("--resume and --no-save cannot be combined: --no-save clears "
              "the store path that --resume would read", file=sys.stderr)
        return 2
    cfg.load_existing_store = bool(args.resume)
    no_save_path = cfg.store_path
    if args.no_save:
        cfg.store_path = ""
    tasks = list(BENCH_TASKS)
    if args.tasks:
        tasks = tasks[: args.tasks]

    note = ""
    if args.no_evolve:
        note = " (evolution OFF)"
    elif args.resume:
        note = f" (resuming {args.store})"
    elif args.no_save:
        note = f" (not saving to {no_save_path})"
    print(f"evolver: {len(tasks)} tasks x {args.epochs} epochs "
          f"via {args.provider}{note}")

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


def _cmd_pack(args) -> int:
    from evolver.evolve.pack import (
        build_pack, describe_pack, import_pack, load_pack, write_pack,
    )
    from evolver.evolve.store import SkillStore

    if args.pack_cmd == "export":
        store = SkillStore.load(args.store)
        if not store.skills:
            print(f"no skills in {args.store} -- nothing to export")
            return 1
        pack = build_pack(
            store, name=args.name, author=args.author,
            description=args.description, min_uses=args.min_uses,
            only_proven=args.only_proven,
        )
        m = pack["manifest"]
        if not pack["skills"]:
            print("filters excluded every skill -- pack would be empty")
            return 1
        path = write_pack(pack, args.out)
        print(f"exported {m['skill_count']} skills, "
              f"{m['pitfall_count']} pitfalls -> {path}")
        if args.only_proven:
            print(f"  ({m['proven_skill_count']} proven skills in the source library)")
        return 0

    if args.pack_cmd == "import":
        store = SkillStore.load(args.store)
        try:
            pack = load_pack(args.path)
        except (OSError, ValueError) as exc:
            print(f"cannot read pack: {exc}")
            return 1
        report = import_pack(store, pack, discount_stats=not args.no_discount,
                             dry_run=args.dry_run)
        prefix = "[dry-run] " if args.dry_run else ""
        print(f"{prefix}{report.summary()}")
        for name in report.added:
            print(f"  + {name}")
        for name in report.merged:
            print(f"  ~ {name}")
        if report.rejected:
            print("  REJECTED -- pack not imported:")
            for r in report.rejected[:10]:
                print(f"    ! {r}")
        if not args.dry_run and report.ok:
            store.save(args.store)
            print(f"saved -> {args.store}")
        return 0 if report.ok else 1

    if args.pack_cmd == "inspect":
        try:
            d = describe_pack(load_pack(args.path))
        except (OSError, ValueError) as exc:
            print(f"cannot read pack: {exc}")
            return 1
        for k, v in d.items():
            if k == "problems":
                continue
            print(f"  {k:16s} {v}")
        if d["problems"]:
            print("\n  PROBLEMS:")
            for p in d["problems"]:
                print(f"    ! {p}")
            return 1
        print("\n  pack is valid and loadable")
        return 0

    return 1


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
                "skills": _cmd_skills, "pack": _cmd_pack, "init": _cmd_init}
    return handlers[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
