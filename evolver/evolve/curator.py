"""Autonomous curation of the skill library.

An unattended skill library rots in three specific ways, and each needs a
different remedy:

1. **Dead weight** -- skills that stopped being useful still occupy context
   budget every single turn. Fix: prune by score, which decays with disuse.
2. **Fragmentation** -- the same capability distilled twice under slightly
   different names, so neither accumulates enough usage to look reliable.
   Fix: merge near-duplicates and pool their statistics.
3. **Bloat** -- past a few hundred entries, even *selecting* skills costs real
   tokens. Fix: hard cap, keeping the highest scorers.

Without this, distillation is a slow leak: every run adds a skill and nothing
ever removes one, until the context budget is mostly skill catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from evolver.core.config import EvolveConfig
from evolver.core.types import Skill
from evolver.evolve.naming import jaccard, token_set
from evolver.evolve.store import SkillStore


@dataclass
class CurationAction:
    """One thing the curator did, for auditability."""

    kind: str          # prune | merge | cap | keep
    detail: str
    names: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass
class Curator:
    """Scores, prunes, and merges a skill library."""

    config: EvolveConfig = field(default_factory=EvolveConfig)
    actions: List[CurationAction] = field(default_factory=list)

    # -- similarity ------------------------------------------------------
    @staticmethod
    def similarity(a: Skill, b: Skill) -> float:
        """Blend trigger-token overlap with code-shape overlap."""
        trig = jaccard(token_set(a.trigger), token_set(b.trigger))
        code = jaccard(token_set(a.body + a.signature), token_set(b.body + b.signature))
        return 0.6 * trig + 0.4 * code

    def _find_merge_pair(self, skills: List[Skill]) -> Tuple[Skill, Skill, float] | None:
        best = None
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                sim = self.similarity(skills[i], skills[j])
                if sim >= self.config.merge_similarity and (best is None or sim > best[2]):
                    best = (skills[i], skills[j], sim)
        return best

    @staticmethod
    def merge(a: Skill, b: Skill) -> Skill:
        """Combine two near-duplicate skills, pooling their statistics."""
        primary, secondary = (a, b) if a.score >= b.score else (b, a)
        merged = Skill(
            name=primary.name,
            description=primary.description,
            signature=primary.signature,
            body=primary.body,
            trigger=" ".join(sorted(set(primary.trigger.split()) | set(secondary.trigger.split()))),
            tags=sorted(set(primary.tags) | set(secondary.tags)),
            generation=max(primary.generation, secondary.generation) + 1,
            parents=sorted(set(primary.parents + secondary.parents
                               + [primary.skill_id, secondary.skill_id])),
        )
        merged.stats.uses = primary.stats.uses + secondary.stats.uses
        merged.stats.successes = primary.stats.successes + secondary.stats.successes
        merged.stats.failures = primary.stats.failures + secondary.stats.failures
        merged.stats.tokens_saved = primary.stats.tokens_saved + secondary.stats.tokens_saved
        merged.stats.created_at = min(primary.stats.created_at, secondary.stats.created_at)
        merged.stats.last_used = max(primary.stats.last_used, secondary.stats.last_used)
        return merged

    # -- main entry ------------------------------------------------------
    def curate(self, store: SkillStore) -> List[CurationAction]:
        self.actions = []
        self._merge(store)
        self._prune(store)
        self._cap(store)
        return list(self.actions)

    def _merge(self, store: SkillStore) -> None:
        for _ in range(len(store.skills)):
            pair = self._find_merge_pair(list(store.skills.values()))
            if pair is None:
                break
            a, b, sim = pair
            merged = self.merge(a, b)
            del store.skills[a.name]
            del store.skills[b.name]
            store.skills[merged.name] = merged
            self.actions.append(
                CurationAction("merge", f"{a.name} + {b.name} -> {merged.name} (sim={sim:.2f})",
                               [a.name, b.name, merged.name])
            )

    def _prune(self, store: SkillStore) -> None:
        doomed = [
            name for name, s in store.skills.items()
            if s.stats.uses > 0 and s.score < self.config.min_score_to_keep
        ]
        for name in doomed:
            del store.skills[name]
        if doomed:
            self.actions.append(
                CurationAction("prune", f"removed {len(doomed)} low-value skill(s)", doomed)
            )

    def _cap(self, store: SkillStore) -> None:
        if len(store.skills) <= self.config.max_skills:
            return
        ranked = sorted(store.skills.values(), key=lambda s: s.score)
        drop = len(store.skills) - self.config.max_skills
        for s in ranked[:drop]:
            del store.skills[s.name]
        self.actions.append(
            CurationAction("cap", f"dropped {drop} skill(s) over the {self.config.max_skills} cap",
                           [s.name for s in ranked[:drop]])
        )

    def report(self) -> str:
        if not self.actions:
            return "curator: no changes needed"
        return "\n".join(f"  - {a}" for a in self.actions)
