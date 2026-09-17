"""Tests for the retrieval-specificity rule and honest off-task accounting.

These exist because the library previously reported a +20% "generalisation
win" that was entirely an artifact:

  * ``sum_squares_even`` (trigger ``"even squares sum"``) matched the task
    "Group [...] and sum 'v' within each group" on the single shared word
    ``sum``, because any keyword hit counted regardless of how generic it was.
  * A distilled pitfall about *even squares* did the same thing, and its
    warning flipped ``group_and_sum`` from a failure into a success.

Both perturbations were then booked as cross-task transfer. The scoreboard
looked like generalisation; it was two dictionary lookups colliding.
"""

from __future__ import annotations

from evolver.core.types import Pitfall, Skill, _GENERIC_TERMS

EVEN_SQUARES_TASK = "Compute the sum of squares of the even numbers in [1, 2, 3, 4, 5, 6]."
GROUP_AND_SUM_TASK = ("Group [{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}] by key 'k' "
                      "and sum 'v' within each group.")


def _skill(name: str, trigger: str) -> Skill:
    return Skill(name=name, signature=f"{name}(d)", body="return d",
                 description="", trigger=trigger)


def _pitfall(tags) -> Pitfall:
    return Pitfall(description="", tags=list(tags))


# -- the false positive --------------------------------------------------

def test_generic_only_match_is_refused():
    """`sum` alone must not be enough to inject an unrelated skill."""
    s = _skill("sum_squares_even", "even squares sum")
    assert s.matches(GROUP_AND_SUM_TASK) == 0.0


def test_generic_term_cannot_carry_a_match():
    """The specific terms must hit; generic words only ever break ties."""
    s = _skill("sum_squares_even", "even squares sum")
    assert s.matches(EVEN_SQUARES_TASK) > 0.0


def test_specific_matches_survive_the_rule():
    """Guarding against false positives must not blind the home field."""
    cases = [
        ("group_key_sum_within_each", "group key sum within each", GROUP_AND_SUM_TASK),
        ("total", "10 20 30 40 total", "Total all the numbers in [10, 20, 30, 40]."),
        ("palindrome", "level palindrome", "Is \"Level\" a palindrome?"),
        ("second_largest", "largest second", "Find the second largest value in [5, 5, 4, 3]."),
    ]
    for name, trigger, text in cases:
        assert _skill(name, trigger).matches(text) > 0.0, f"{name} lost its own task"


def test_trigger_of_only_generic_words_never_fires():
    s = _skill("whatever", "sum value number")
    assert s.matches("Total the sum of all values and numbers.") == 0.0


def test_shares_specific_term_is_strict():
    s = _skill("sum_squares_even", "even squares sum")
    assert not s.shares_specific_term(GROUP_AND_SUM_TASK)
    assert s.shares_specific_term(EVEN_SQUARES_TASK)


def test_distinguishing_words_are_not_treated_as_generic():
    """`group`/`key`/`total` look generic but are exactly what separates tasks."""
    for word in ("group", "key", "total", "second", "largest", "palindrome"):
        assert word not in _GENERIC_TERMS, f"{word} must stay discriminative"


# -- the same flaw in pitfalls -------------------------------------------

def test_pitfall_generic_match_is_refused():
    p = _pitfall(["even", "squares", "sum"])
    assert p.matches(GROUP_AND_SUM_TASK) == 0.0


def test_pitfall_still_fires_on_its_own_task():
    p = _pitfall(["even", "squares", "sum"])
    assert p.matches(EVEN_SQUARES_TASK) > 0.0


def test_pitfall_with_only_generic_tags_never_fires():
    p = _pitfall(["sum", "value", "number"])
    assert p.matches("Total the sum of all values.") == 0.0


# -- end-to-end ----------------------------------------------------------

def test_held_out_library_does_not_inject_a_foreign_skill():
    """Seeding with train-half skills must not fire on held-out tasks."""
    from evolver.bench.runner import _default_tools
    from evolver.bench.tasks import BENCH_TASKS
    from evolver.core.config import Config
    from evolver.core.engine import Evolver
    from evolver.core.types import SkillStats

    held = [t for i, t in enumerate(BENCH_TASKS) if i % 2 == 1]
    engine = Evolver(config=Config(store_path=""), tools=_default_tools(), max_steps=6)

    # Plant the exact skill that used to misfire.
    engine.store.add_skill(_skill("sum_squares_even", "even squares sum"))
    engine.store.skills["sum_squares_even"].stats = SkillStats(uses=3, successes=3)

    offenders = []
    for t in held:
        picked = engine.store.select(t.prompt, k=3, threshold=0.20)
        offenders.extend(s.name for s in picked if t.task_id != "sum_even_squares")
    assert not offenders, f"foreign skills injected: {sorted(set(offenders))}"


def test_a_stale_store_cannot_contaminate_a_fresh_run():
    """Rerunning must not silently start from the previous run's library."""
    from evolver.bench.runner import _default_tools
    from evolver.core.config import Config
    from evolver.core.engine import Evolver

    cfg = Config(store_path=".evolver/skills.json")   # path that may exist on disk
    engine = Evolver(config=cfg, tools=_default_tools(), max_steps=4)
    assert len(engine.store.skills) == 0, "engine resumed a store the caller never asked for"
    assert len(engine.store.pitfalls) == 0
