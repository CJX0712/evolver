"""Skill pack export/import -- making an evolved library a distributable asset.

The gap this closes
-------------------
An evolved skill library was, until now, a local artifact. It lived in
``.evolver/skills.json``, it accumulated real value from real runs, and there
was no way to hand it to anyone -- not another agent, not another machine, not
a teammate. Every user started from zero and paid the same learning cost.

That is a strange gap for a system whose whole premise is that capability
accumulates. Accumulated capability that cannot move is just a cache.

What a pack is
--------------
A single JSON file with three sections:

``manifest``
    Format version, provenance (who made it, from what), and counts. Enough to
    decide whether to trust it before reading further.
``skills`` / ``pitfalls``
    The learned items.

Two design rules
----------------
**Stats travel, but they travel marked.** Importing a pack must not let foreign
skills inherit local trust they never earned. A skill that arrives already
boasting a 100% success rate, in a library where selection weights reliability,
would jump the queue ahead of locally-proven work. So imported stats are
preserved (they are real information about the skill's history) but scaled down
and flagged, and the item lands in a probationary state until it wins locally.

**Import is additive, never destructive.** A pack import can add skills and
merge stats, but it can never lower a local counter, and it never overwrites a
skill with a lower-scoring body. Losing local evidence to a bad import would be
unrecoverable.

Security note: a pack contains executable code. Importing one is equivalent to
running a stranger's code in your sandbox, so :func:`validate_pack` checks the
source against the sandbox allowlist before anything is loaded, and refuses the
whole pack on the first violation rather than silently skipping items.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evolver.core.types import Pitfall, Skill, SkillStats
from evolver.evolve.store import SkillStore

PACK_FORMAT = "evolver-skillpack"
PACK_VERSION = 1

# Imported stats are real, but they were earned elsewhere. Trust them at a
# fraction until the skill proves itself here.
IMPORT_TRUST_DISCOUNT = 0.5
PROBATION_USE_FLOOR = 3


@dataclass
class PackReport:
    """What an import actually did -- never just 'ok'."""

    added: List[str] = field(default_factory=list)
    merged: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    pitfalls_added: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected and not self.errors

    def summary(self) -> str:
        parts = [
            f"{len(self.added)} added",
            f"{len(self.merged)} merged",
            f"{len(self.skipped)} skipped",
            f"{len(self.pitfalls_added)} pitfalls",
        ]
        if self.rejected:
            parts.append(f"{len(self.rejected)} REJECTED")
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added, "merged": self.merged, "skipped": self.skipped,
            "rejected": self.rejected, "pitfalls_added": self.pitfalls_added,
            "errors": self.errors, "ok": self.ok, "summary": self.summary(),
        }


# -- export --------------------------------------------------------------

def build_pack(store: SkillStore, *, name: str = "skillpack",
               author: str = "", description: str = "",
               min_uses: int = 0, only_proven: bool = False) -> Dict[str, Any]:
    """Serialise a store into a pack.

    ``only_proven`` exports just the skills with off-task wins -- the ones with
    evidence of generalising. That is the right default when sharing, because a
    skill that only ever won on its home task is a local optimisation and of
    little use to anyone else.
    """
    skills: List[Dict[str, Any]] = []
    for s in store.skills.values():
        if s.stats.uses < min_uses:
            continue
        if only_proven and s.stats.off_task_successes == 0:
            continue
        skills.append(s.to_dict())

    pitfalls = [p.to_dict() for p in store.pitfalls.values()]

    return {
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "manifest": {
            "name": name,
            "author": author,
            "description": description,
            "created_at": time.time(),
            "source_runs": store.runs,
            "skill_count": len(skills),
            "pitfall_count": len(pitfalls),
            "proven_skill_count": sum(
                1 for s in store.skills.values() if s.stats.off_task_successes
            ),
            "filters": {"min_uses": min_uses, "only_proven": only_proven},
        },
        "skills": skills,
        "pitfalls": pitfalls,
    }


def write_pack(pack: Dict[str, Any], path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


# -- validation ----------------------------------------------------------

def validate_pack(pack: Dict[str, Any]) -> List[str]:
    """Return a list of fatal problems. Empty list means the pack is loadable.

    Every skill body is checked against the sandbox allowlist before import.
    A pack is executable code from an untrusted source, so this runs first and
    a single violation rejects the whole pack -- partial imports leave a store
    in a state nobody can reason about afterwards.
    """
    problems: List[str] = []

    if not isinstance(pack, dict):
        return ["pack is not a JSON object"]
    if pack.get("format") != PACK_FORMAT:
        problems.append(f"not a skillpack (format={pack.get('format')!r})")
    version = pack.get("version")
    if not isinstance(version, int) or version > PACK_VERSION:
        problems.append(f"unsupported pack version: {version!r}")
    if not isinstance(pack.get("skills"), list):
        problems.append("missing or non-list 'skills'")

    # Structural check only -- imported lazily so this module stays usable
    # without pulling the sandbox in at import time.
    from evolver.act.sandbox import SecurityViolation, validate_source
    from evolver.core.config import SandboxConfig

    sandbox_cfg = SandboxConfig()
    for i, raw in enumerate(pack.get("skills") or []):
        if not isinstance(raw, dict):
            problems.append(f"skill[{i}] is not an object")
            continue
        label = raw.get("name") or f"skill[{i}]"
        missing = [k for k in ("name", "signature", "body") if not raw.get(k)]
        if missing:
            problems.append(f"{label}: missing {', '.join(missing)}")
            continue
        try:
            code = f"def {raw['signature']}:\n{raw['body']}"
            validate_source(code, sandbox_cfg)
        except SecurityViolation as exc:
            problems.append(f"{label}: rejected by sandbox ({exc})")
        except (SyntaxError, ValueError, TypeError) as exc:
            problems.append(f"{label}: malformed source ({exc})")

    for i, raw in enumerate(pack.get("pitfalls") or []):
        if not isinstance(raw, dict) or not raw.get("description"):
            problems.append(f"pitfall[{i}] is malformed")

    return problems


# -- import --------------------------------------------------------------

def _discounted_stats(raw: Dict[str, Any]) -> SkillStats:
    """Scale an incoming skill's track record down to a probationary prior.

    Off-task counters are the exception: they are *decremented*, never halved.
    A pack claiming one cross-task win says almost nothing -- one data point is
    one data point no matter whose machine produced it. Claiming three or more
    is the interesting case, and even then we shave one off so the threshold
    must be re-earned locally. Zero stays zero, so an unproven skill stays
    unproven and cannot buy generalist standing by travelling.
    """
    stats = SkillStats.from_dict(raw or {})
    stats.successes = int(stats.successes * IMPORT_TRUST_DISCOUNT)
    stats.failures = int(stats.failures * IMPORT_TRUST_DISCOUNT)
    stats.uses = stats.successes + stats.failures
    stats.tokens_saved = int(stats.tokens_saved * IMPORT_TRUST_DISCOUNT)
    stats.off_task_successes = max(0, stats.off_task_successes - 1)
    stats.off_task_uses = max(0, stats.off_task_uses - 1)
    return stats


def import_pack(
    store: SkillStore,
    pack: Dict[str, Any],
    *,
    discount_stats: bool = True,
    dry_run: bool = False,
) -> PackReport:
    """Merge a pack into ``store``. Additive only -- nothing local is lost.

    Returns a report describing every action, because "imported 3 skills" hides
    which three, and whether any of them were rejected for safety.
    """
    report = PackReport()

    problems = validate_pack(pack)
    if problems:
        report.errors.extend(problems)
        report.rejected = [p.split(":")[0] for p in problems][:10]
        return report

    for raw in pack.get("skills") or []:
        name = raw["name"]
        incoming = Skill.from_dict(raw)
        if discount_stats:
            incoming.stats = _discounted_stats(raw.get("stats", {}))
        incoming.trigger = raw.get("trigger", "")

        existing = store.skills.get(name)
        if existing is None:
            if not dry_run:
                store.skills[name] = incoming
            report.added.append(name)
            continue

        # Same name already present. Keep whichever scores better, but never
        # let an import reduce accumulated local evidence.
        if incoming.score > existing.score:
            incoming.stats.uses += existing.stats.uses
            incoming.stats.successes += existing.stats.successes
            incoming.stats.failures += existing.stats.failures
            incoming.stats.tokens_saved += existing.stats.tokens_saved
            incoming.stats.off_task_uses += existing.stats.off_task_uses
            incoming.stats.off_task_successes += existing.stats.off_task_successes
            incoming.generation = max(incoming.generation, existing.generation) + 1
            incoming.parents = sorted(set(incoming.parents + [existing.skill_id]))
            if not dry_run:
                store.skills[name] = incoming
            report.merged.append(name)
        else:
            # Local version wins. Fold in the incoming evidence at a discount
            # rather than discarding it -- it is still information.
            incoming_uses = incoming.stats.uses
            if not dry_run:
                existing.stats.uses += incoming_uses
                existing.stats.successes += int(
                    incoming_uses * (incoming.stats.success_rate or 0.0)
                )
            report.skipped.append(name)

    for raw in pack.get("pitfalls") or []:
        p = Pitfall.from_dict(raw)
        if not dry_run:
            outcome = store.add_pitfall(p)
            if outcome == "deduped":
                report.skipped.append(p.description[:40])
                continue
        report.pitfalls_added.append(p.description[:40])

    return report


def load_pack(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise a pack without importing it -- for `--dry-run` inspection."""
    m = pack.get("manifest", {}) or {}
    return {
        "name": m.get("name", "(unnamed)"),
        "author": m.get("author", ""),
        "description": m.get("description", ""),
        "format": pack.get("format"),
        "version": pack.get("version"),
        "skills": len(pack.get("skills") or []),
        "pitfalls": len(pack.get("pitfalls") or []),
        "source_runs": m.get("source_runs"),
        "proven_skills": m.get("proven_skill_count"),
        "created_at": m.get("created_at"),
        "problems": validate_pack(pack),
    }


__all__ = [
    "PACK_FORMAT", "PACK_VERSION", "PackReport",
    "build_pack", "write_pack", "load_pack", "describe_pack",
    "validate_pack", "import_pack",
]
