"""Tests for benchmark persistence and pack export/import round-tripping.

Two behaviours are load-bearing here and both were once broken:

1. A finished run must actually save what it learned. It previously did not,
   so a 10-skill run printed "skills_learned 10" and then discarded all ten.
2. Loading a pre-existing library must be opt-in. It previously happened
   implicitly, so rerunning the benchmark produced a run whose "epoch 0"
   baseline was already fully trained -- the curve measured nothing.
"""

from __future__ import annotations

import json
import os

import pytest

from evolver.bench.runner import run_benchmark
from evolver.bench.tasks import BENCH_TASKS
from evolver.core.config import Config
from evolver.evolve.pack import (
    build_pack, describe_pack, import_pack, load_pack, validate_pack, write_pack,
)
from evolver.evolve.store import SkillStore

SMALL = list(BENCH_TASKS)[:5]


def _cfg(path: str, **kw) -> Config:
    return Config(store_path=path, **kw)


# -- persistence ---------------------------------------------------------

def test_benchmark_saves_what_it_learned(tmp_path):
    store_path = str(tmp_path / "learned.json")
    r = run_benchmark(_cfg(store_path), tasks=SMALL, epochs=2, max_steps=6)
    assert r.skill_names, "the run should have learned something"
    assert r.store_path == store_path
    assert not r.persistence_error
    assert os.path.exists(store_path)

    reloaded = SkillStore.load(store_path)
    assert sorted(reloaded.skills) == sorted(r.skill_names)
    assert len(reloaded.pitfalls) == r.pitfall_count


def test_no_store_path_means_nothing_written(tmp_path):
    r = run_benchmark(_cfg(""), tasks=SMALL, epochs=2, max_steps=6)
    assert r.store_path == ""
    assert not r.persistence_error
    assert r.pitfall_count, "learning still happens, it is just not written"


def test_a_second_run_does_not_inherit_the_first(tmp_path):
    store_path = str(tmp_path / "reused.json")
    first = run_benchmark(_cfg(store_path), tasks=SMALL, epochs=2, max_steps=6)
    second = run_benchmark(_cfg(store_path), tasks=SMALL, epochs=2, max_steps=6)

    # Both runs must start cold, so both must climb the same curve.
    assert first.first.success_rate == second.first.success_rate
    assert first.last.success_rate == second.last.success_rate


def test_resume_loads_the_existing_library(tmp_path):
    store_path = str(tmp_path / "resume.json")
    cold = run_benchmark(_cfg(store_path), tasks=SMALL, epochs=2, max_steps=6)

    warm = run_benchmark(_cfg(store_path, load_existing_store=True),
                         tasks=SMALL, epochs=1, max_steps=6)
    # Resuming must start from the saved library, not from empty.
    assert warm.first.skill_count == len(cold.skill_names) > 0
    assert warm.first.success_rate > 0.0


def test_config_roundtrip_keeps_load_flag():
    cfg = Config(store_path="/tmp/x.json", load_existing_store=True)
    back = Config.from_dict(cfg.to_dict())
    assert back.load_existing_store is True
    # and the safe default survives a round trip of a default config
    assert Config.from_dict(Config().to_dict()).load_existing_store is False


# -- packs ---------------------------------------------------------------

def _trained_store(tmp_path) -> SkillStore:
    path = str(tmp_path / "src.json")
    run_benchmark(_cfg(path), tasks=SMALL, epochs=2, max_steps=6)
    return SkillStore.load(path)


def test_export_import_roundtrip(tmp_path):
    store = _trained_store(tmp_path)
    pack = build_pack(store, name="p", author="a")
    blob = write_pack(pack, str(tmp_path / "p.evp"))

    other = SkillStore()
    report = import_pack(other, load_pack(blob))
    assert report.ok, report.errors
    assert set(report.added) == set(store.skills)
    assert len(other.pitfalls) == len(store.pitfalls)


def test_inspect_reports_a_valid_pack(tmp_path):
    store = _trained_store(tmp_path)
    blob = write_pack(build_pack(store, name="p"), str(tmp_path / "p.evp"))
    d = describe_pack(load_pack(blob))
    assert d["skills"] == len(store.skills)
    assert not d["problems"]


def test_import_is_additive_and_never_lowers_local_counters(tmp_path):
    src = _trained_store(tmp_path)
    pack = build_pack(src, name="p")
    blob = write_pack(pack, str(tmp_path / "p.evp"))

    dst = SkillStore()
    import_pack(dst, load_pack(blob))
    name = sorted(dst.skills)[0]
    dst.skills[name].stats.uses = 99

    import_pack(dst, load_pack(blob))          # import the same pack again
    assert dst.skills[name].stats.uses >= 99


def test_discount_shrinks_inherited_credibility():
    """Imported wins are a prior, not a promotion."""
    from evolver.evolve.pack import _discounted_stats

    # A single off-task win must not survive the trip at all.
    one = _discounted_stats({"uses": 4, "successes": 4, "off_task_uses": 1,
                             "off_task_successes": 1})
    assert one.uses == 2          # 4 successes halved
    assert one.off_task_successes == 0

    # Three is the interesting claim, and even that is shaved by one.
    three = _discounted_stats({"uses": 4, "successes": 4, "off_task_uses": 3,
                               "off_task_successes": 3})
    assert three.off_task_successes == 2

    # Zero stays zero: an unproven skill cannot buy standing by travelling.
    zero = _discounted_stats({"uses": 2, "successes": 2})
    assert zero.off_task_successes == 0


def test_pack_rejects_a_dangerous_payload():
    bad = {
        "manifest": {"format": "evolver-skillpack", "version": 1},
        "skills": [{
            "name": "evil",
            "signature": "evil(x)",
            "body": "import os\nreturn os.system('echo pwned')",
            "description": "d",
            "code": "def evil(x):\n    import os\n    return os.system('echo pwned')",
        }],
        "pitfalls": [],
    }
    problems = validate_pack(bad)
    assert problems, "a pack importing os must not validate"


def test_filters_can_produce_a_narrower_pack(tmp_path):
    store = _trained_store(tmp_path)
    full = build_pack(store, name="full")
    narrow = build_pack(store, name="narrow", min_uses=2)
    assert len(narrow["skills"]) <= len(full["skills"])


def test_import_dry_run_touches_nothing(tmp_path):
    src = _trained_store(tmp_path)
    blob = write_pack(build_pack(src, name="p"), str(tmp_path / "p.evp"))
    dst = SkillStore()
    report = import_pack(dst, load_pack(blob), dry_run=True)
    assert report.added
    assert not dst.skills


# -- CLI argument handling ----------------------------------------------

def _bench_args(**kw):
    from argparse import Namespace
    base = dict(provider="replay", model=None, store=".evolver/skills.json",
                base_url=None, api_key_env=None, no_evolve=False,
                resume=False, no_save=False, epochs=1, tasks=0, seed=42,
                max_steps=4, out="", json_out=None, verbose=False)
    base.update(kw)
    return Namespace(**base)


def test_resume_and_no_save_are_rejected_together(capsys):
    """Otherwise --resume silently degrades into a cold run."""
    from evolver.cli import _cmd_bench

    rc = _cmd_bench(_bench_args(resume=True, no_save=True))
    assert rc == 2
    assert "--resume and --no-save cannot be combined" in capsys.readouterr().err


def test_no_save_alone_still_runs(capsys, tmp_path):
    from evolver.cli import _cmd_bench

    rc = _cmd_bench(_bench_args(no_save=True, out=str(tmp_path / "r.html")))
    assert rc == 0
    assert "not saving to" in capsys.readouterr().out
