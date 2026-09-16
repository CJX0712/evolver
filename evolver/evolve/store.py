"""Persistent skill and pitfall library.

Retrieval is the part that decides whether any of this works. A skill that is
never surfaced might as well not exist, and a skill that is surfaced at the
wrong time is worse than nothing -- it burns context budget and misleads.

Selection balances three terms: lexical match to the current task, historical
reliability, and familiarity. Freshly distilled skills get an optimistic prior
on reliability, otherwise they could never be tried even once -- the classic
cold-start trap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolver.core.types import Pitfall, Skill
from evolver.evolve.naming import jaccard, token_set


@dataclass
class SkillStore:
    """An evolvable library of skills and pitfalls."""

    path: Optional[str] = None
    skills: Dict[str, Skill] = field(default_factory=dict)
    pitfalls: Dict[str, Pitfall] = field(default_factory=dict)
    runs: int = 0

    # -- mutation --------------------------------------------------------
    def add_skill(self, skill: Skill) -> str:
        """Insert a skill, merging into an existing one with the same name."""
        existing = self.skills.get(skill.name)
        if existing is None:
            self.skills[skill.name] = skill
            return "added"

        # Same name: keep the better-performing body, accumulate the stats.
        if skill.stats.uses == 0 and existing.stats.uses > 0:
            # A fresh distillation of a known skill: keep proven code, refresh
            # the trigger vocabulary so retrieval can improve.
            existing.trigger = " ".join(sorted(set(existing.trigger.split()) | set(skill.trigger.split())))
            return "refreshed"
        if skill.score >= existing.score:
            skill.stats.uses += existing.stats.uses
            skill.stats.successes += existing.stats.successes
            skill.stats.failures += existing.stats.failures
            skill.stats.tokens_saved += existing.stats.tokens_saved
            skill.generation = max(skill.generation, existing.generation) + 1
            skill.parents = sorted(set(skill.parents + [existing.skill_id]))
            self.skills[skill.name] = skill
        return "merged"

    def add_pitfall(self, pitfall: Pitfall) -> str:
        key = pitfall.description[:80]
        for p in self.pitfalls.values():
            if p.description[:80] == key:
                p.hits += 1
                return "deduped"
        self.pitfalls[pitfall.pitfall_id] = pitfall
        return "added"

    # -- retrieval -------------------------------------------------------
    def _select_score(self, skill: Skill, task_text: str) -> float:
        m = skill.matches(task_text)
        if m <= 0:
            return 0.0
        reliability = skill.stats.success_rate if skill.stats.uses else 0.80
        experience = min(1.0, skill.stats.uses / 5.0)
        return round(m * (0.45 + 0.35 * reliability + 0.20 * experience), 6)

    def select(self, task_text: str, k: int = 3, threshold: float = 0.20) -> List[Skill]:
        scored = [(self._select_score(s, task_text), s) for s in self.skills.values()]
        scored = [(sc, s) for sc, s in scored if sc >= threshold]
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:k]]

    def warnings(self, task_text: str, k: int = 2, threshold: float = 0.25) -> List[Pitfall]:
        scored = [(p.matches(task_text), p) for p in self.pitfalls.values()]
        scored = [(sc, p) for sc, p in scored if sc >= threshold]
        scored.sort(key=lambda x: (-x[0], x[1].created_at))
        return [p for _, p in scored[:k]]

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    # -- outcome tracking ------------------------------------------------
    def record_skill_use(self, name: str, success: bool, tokens_saved: int = 0) -> None:
        s = self.skills.get(name)
        if s:
            s.stats.record(success, tokens_saved)

    def record_pitfall_outcome(self, pitfall_id: str, avoided: bool) -> None:
        p = self.pitfalls.get(pitfall_id)
        if p:
            p.hits += 1
            if avoided:
                p.avoids += 1

    def touch_all(self, names: List[str], success: bool) -> None:
        for n in names:
            self.record_skill_use(n, success)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "runs": self.runs,
            "skills": [s.to_dict() for s in self.skills.values()],
            "pitfalls": [p.to_dict() for p in self.pitfalls.values()],
        }

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if not target:
            return ""
        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)  # atomic: never leave a half-written library
        return str(p)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], path: Optional[str] = None) -> "SkillStore":
        store = cls(path=path)
        store.runs = int(d.get("runs", 0))
        for s in d.get("skills", []):
            try:
                sk = Skill.from_dict(s)
                store.skills[sk.name] = sk
            except Exception:  # noqa: BLE001 - skip corrupt entries
                continue
        for p in d.get("pitfalls", []):
            try:
                pf = Pitfall.from_dict(p)
                store.pitfalls[pf.pitfall_id] = pf
            except Exception:  # noqa: BLE001
                continue
        return store

    @classmethod
    def load(cls, path: str) -> "SkillStore":
        p = Path(path)
        if not p.exists():
            return cls(path=path)
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), path=path)
        except (json.JSONDecodeError, OSError):
            return cls(path=path)

    # -- introspection ---------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        skills = list(self.skills.values())
        used = [s for s in skills if s.stats.uses]
        return {
            "runs": self.runs,
            "skills": len(skills),
            "skills_used": len(used),
            "skill_success_rate": (sum(s.stats.successes for s in skills)
                                   / max(1, sum(s.stats.uses for s in skills))),
            "pitfalls": len(self.pitfalls),
            "tokens_saved": sum(s.stats.tokens_saved for s in skills),
            "avg_generation": (sum(s.generation for s in skills) / len(skills)) if skills else 0.0,
        }

    def top(self, n: int = 5) -> List[Skill]:
        return sorted(self.skills.values(), key=lambda s: -s.score)[:n]
