"""Retrieval, curation, and persistence of the skill library."""

from __future__ import annotations

import json

import pytest

from evolver.core.config import EvolveConfig
from evolver.core.types import Pitfall, Skill, SkillStats
from evolver.evolve.curator import Curator
from evolver.evolve.store import SkillStore


def _skill(name: str, trigger: str, body: str = "    return total(xs)",
           uses: int = 0, successes: int = 0) -> Skill:
    s = Skill(name=name, description=f"does {name}", signature=f"{name}(xs)",
              body=body, trigger=trigger)
    s.stats.uses = uses
    s.stats.successes = successes
    return s


def test_select_prefers_relevant_skill():
    store = SkillStore()
    store.add_skill(_skill("second_largest", "largest second"))
    store.add_skill(_skill("merge_adding_shared_keys", "merge adding shared keys"))
    picked = store.select("Find the second largest value in [1, 2].")
    assert picked and picked[0].name == "second_largest"


def test_select_ignores_unrelated_skills():
    store = SkillStore()
    store.add_skill(_skill("merge_adding_shared_keys", "merge adding shared keys"))
    assert store.select("Compute the 3-period moving average of [1, 2, 3].") == []


def test_fresh_skill_is_still_selectable():
    """Cold start: a brand new skill has no stats, but must be usable."""
    store = SkillStore()
    store.add_skill(_skill("second_largest", "largest second"))
    assert store.select("Find the second largest value in [1, 2].")


def test_warnings_surface_relevant_pitfall():
    store = SkillStore()
    pf = Pitfall(description="sorting without dedup is wrong",
                 context="second largest", tags=["second", "largest"])
    store.add_pitfall(pf)
    assert store.warnings("Find the second largest value in [1, 2].")


def test_pitfall_deduplicates_by_description():
    store = SkillStore()
    for _ in range(3):
        store.add_pitfall(Pitfall(description="same mistake", tags=["x"]))
    assert len(store.pitfalls) == 1


def test_adding_same_name_merges_instead_of_duplicating():
    store = SkillStore()
    store.add_skill(_skill("total", "total"))
    store.add_skill(_skill("total", "total numbers"))
    assert len(store.skills) == 1


def test_curator_merges_near_duplicates():
    store = SkillStore()
    store.add_skill(_skill("sum_even_squares", "even squares sum",
                           body="    return total(squares(evens(xs)))"))
    store.add_skill(_skill("squares_of_even_sum", "squares even sum",
                           body="    return total(squares(evens(xs)))"))
    curator = Curator(config=EvolveConfig(merge_similarity=0.6))
    actions = curator.curate(store)
    assert any(a.kind == "merge" for a in actions)
    assert len(store.skills) == 1


def test_curator_prunes_dead_skills():
    store = SkillStore()
    dead = _skill("stale", "stale old", uses=10, successes=0)
    dead.stats.last_used = 0  # ancient
    store.add_skill(dead)
    curator = Curator(config=EvolveConfig(min_score_to_keep=0.5))
    curator.curate(store)
    assert "stale" not in store.skills


def test_curator_enforces_cap():
    store = SkillStore()
    for i in range(12):
        store.add_skill(_skill(f"skill_{i}", f"trigger_{i}"))
    curator = Curator(config=EvolveConfig(max_skills=5))
    curator.curate(store)
    assert len(store.skills) == 5


def test_usage_tracking_updates_stats():
    store = SkillStore()
    store.add_skill(_skill("total", "total"))
    store.record_skill_use("total", True, tokens_saved=100)
    store.record_skill_use("total", False)
    s = store.get("total")
    assert s.stats.uses == 2
    assert s.stats.successes == 1
    assert s.stats.tokens_saved == 100


def test_persistence_roundtrip(tmp_path):
    store = SkillStore(path=str(tmp_path / "skills.json"))
    store.add_skill(_skill("second_largest", "largest second"))
    store.add_pitfall(Pitfall(description="a lesson", tags=["a"]))
    store.save()

    reloaded = SkillStore.load(str(tmp_path / "skills.json"))
    assert list(reloaded.skills) == ["second_largest"]
    assert len(reloaded.pitfalls) == 1
    assert reloaded.get("second_largest").signature == "second_largest(xs)"


def test_load_missing_file_returns_empty(tmp_path):
    store = SkillStore.load(str(tmp_path / "nope.json"))
    assert store.skills == {}


def test_load_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    store = SkillStore.load(str(p))
    assert store.skills == {}


def test_stats_summary():
    store = SkillStore()
    store.add_skill(_skill("a", "a", uses=4, successes=4))
    store.add_skill(_skill("b", "b", uses=4, successes=0))
    s = store.stats()
    assert s["skills"] == 2
    assert s["skill_success_rate"] == 0.5
