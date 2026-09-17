"""Tests for generalisation tracking.

A skill library's headline number -- "80% of skill uses succeeded" -- is the
least interesting number it produces. Almost every use is the skill replaying
its own home task, which it was built from and will always win. That inflates
confidence without measuring anything.

These tests pin the counter that actually matters: success on tasks the skill
was *not* distilled from. It is what separates a library that generalises from
a lookup table with extra steps.
"""

from __future__ import annotations

import unittest

from evolver.core.types import Pitfall, Skill, SkillStats


def _skill(name: str = "total", source: str = "") -> Skill:
    return Skill(name=name, description="d", signature="total(xs)",
                 body="    return sum(xs)", trigger="total sum", source_trajectory=source)


class OffTaskAccounting(unittest.TestCase):
    def test_off_task_success_rate_is_zero_before_any_transfer(self):
        s = SkillStats()
        s.record(True)
        s.record(True)
        self.assertEqual(s.success_rate, 0.5 + 0.5)  # home wins
        self.assertEqual(s.off_task_uses, 0)
        self.assertEqual(s.off_task_success_rate, 0.0)

    def test_off_task_uses_counted_separately(self):
        s = SkillStats()
        s.record(True)                  # home, 1 use
        s.record(True)                  # home, 2 uses
        s.record_off_task(False)        # carried elsewhere, failed
        s.record_off_task(True)         # carried elsewhere, worked
        self.assertEqual(s.uses, 2)
        self.assertEqual(s.off_task_uses, 2)
        self.assertEqual(s.off_task_success_rate, 0.5)
    def test_score_prefers_proven_generalist_over_home_specialist(self):
        """Two skills with identical home records differ once one transfers."""
        specialist = _skill("specialist")
        generalist = _skill("generalist")
        for s in (specialist, generalist):
            for _ in range(4):
                s.stats.record(True)

        # The generalist has been carried twice, winning twice.
        generalist.stats.record_off_task(True)
        generalist.stats.record_off_task(True)

        self.assertGreater(generalist.score, specialist.score)

    def test_failed_transfer_is_penalised(self):
        """A skill that travels and loses should not outrank a stay-at-home."""
        stay_home = _skill("stay_home")
        drifter = _skill("drifter")
        for s in (stay_home, drifter):
            for _ in range(4):
                s.stats.record(True)
        for _ in range(3):
            drifter.stats.record_off_task(False)

        self.assertGreater(stay_home.score, drifter.score)


class BackwardCompatibility(unittest.TestCase):
    def test_old_payload_without_off_task_fields_still_loads(self):
        """Libraries saved before this feature must not break the loader."""
        legacy = {
            "uses": 3, "successes": 2, "failures": 1,
            "tokens_saved": 100, "last_used": 0.0, "created_at": 0.0,
        }
        stats = SkillStats.from_dict(legacy)
        self.assertEqual(stats.uses, 3)
        self.assertEqual(stats.off_task_uses, 0)

    def test_old_skill_payload_without_new_keys_still_loads(self):
        legacy = {
            "name": "old", "description": "d", "signature": "old(xs)",
            "body": "    return xs", "trigger": "old",
            "stats": {"uses": 1, "successes": 1, "failures": 0,
                      "tokens_saved": 0, "last_used": 0.0, "created_at": 0.0},
        }
        skill = Skill.from_dict(legacy)
        self.assertEqual(skill.name, "old")
        self.assertEqual(skill.stats.successes, 1)

    def test_unknown_future_keys_are_ignored_not_fatal(self):
        payload = {
            "name": "future", "description": "d", "signature": "f(x)",
            "body": "    return x", "stats": {"uses": 0, "successes": 0,
            "failures": 0, "tokens_saved": 0, "last_used": 0.0, "created_at": 0.0,
            "some_future_counter": 99},
            "a_field_from_the_future": True,
        }
        skill = Skill.from_dict(payload)
        self.assertEqual(skill.name, "future")


class StoreIntegration(unittest.TestCase):
    def test_store_records_off_task_use(self):
        from evolver.evolve.store import SkillStore

        store = SkillStore()
        store.add_skill(_skill("total"))
        store.record_skill_use("total", True, off_task=False)
        store.record_skill_use("total", False, off_task=True)
        store.record_skill_use("total", True, off_task=True)

        s = store.get("total")
        self.assertEqual(s.stats.uses, 3)
        self.assertEqual(s.stats.off_task_uses, 2)
        self.assertEqual(s.stats.off_task_success_rate, 0.5)

    def test_touch_all_propagates_off_task_flag(self):
        from evolver.evolve.store import SkillStore

        store = SkillStore()
        store.add_skill(_skill("total"))
        store.touch_all(["total"], success=True, off_task=True)
        self.assertEqual(store.get("total").stats.off_task_uses, 1)

    def test_roundtrip_preserves_off_task_stats(self):
        from evolver.evolve.store import SkillStore

        store = SkillStore()
        store.add_skill(_skill("total"))
        store.record_skill_use("total", True, off_task=True)
        revived = SkillStore.from_dict(store.to_dict())
        self.assertEqual(revived.get("total").stats.off_task_uses, 1)

    def test_stats_surface_off_task_aggregate(self):
        from evolver.evolve.store import SkillStore

        store = SkillStore()
        store.add_skill(_skill("total"))
        store.record_skill_use("total", True, off_task=True)
        store.record_skill_use("total", False, off_task=True)
        s = store.stats()
        self.assertIn("off_task_uses", s)
        self.assertEqual(s["off_task_uses"], 2)


class PitfallScoreUnchanged(unittest.TestCase):
    def test_pitfall_score_is_still_avoid_rate(self):
        p = Pitfall(description="d", tags=["trap"])
        p.hits, p.avoids = 4, 3
        self.assertAlmostEqual(p.score, 0.75)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
