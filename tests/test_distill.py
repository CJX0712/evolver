"""Distillation: successes become skills, failures become pitfalls."""

from __future__ import annotations

import pytest

from evolver.core.types import RunStatus, Step, StepKind, Trajectory
from evolver.evolve.distill import (
    Distiller,
    heuristic_distill_pitfall,
    heuristic_distill_skill,
)


def _success(task: str, code: str) -> Trajectory:
    t = Trajectory(task=task, status=RunStatus.SUCCESS)
    t.add(Step(kind=StepKind.CODE, content=code))
    return t


def _failure(task: str, answer: str) -> Trajectory:
    return Trajectory(task=task, status=RunStatus.FAILURE,
                      final_answer=answer, failure_reason="wrong answer")


def test_promotes_list_literal_to_parameter():
    t = _success("Find the second largest value in [5, 5, 4, 3].",
                 "result = nth(dedup_sorted_desc([5, 5, 4, 3]), 1)\nfinish(result)")
    sk = heuristic_distill_skill(t)
    assert sk is not None
    assert sk.name == "second_largest"
    assert sk.signature == "second_largest(xs)"
    assert "[5, 5, 4, 3]" not in sk.body
    assert "nth(dedup_sorted_desc(xs), 1)" in sk.body


def test_promotes_dict_literals_to_two_parameters():
    t = _success("Merge {'a': 1} with {'b': 2}, adding shared keys.",
                 "result = merge_sum({'a': 1}, {'b': 2})\nfinish(result)")
    sk = heuristic_distill_skill(t)
    assert sk is not None
    assert sk.signature.count(",") == 1  # two parameters


def test_leaves_short_field_names_inline():
    """'k' and 'v' are field names, not data -- promoting them is a bug."""
    t = _success("Group rows by key and sum.",
                 "result = {k: sum_field(v, 'v') for k, v in group_by([{'k': 'a'}], 'k').items()}\n"
                 "finish(result)")
    sk = heuristic_distill_skill(t)
    assert sk is not None
    assert sk.signature.endswith("(xs)"), sk.signature
    assert "'k'" in sk.body and "'v'" in sk.body


def test_comprehension_target_not_promoted():
    """`for k, v in ...` must survive: it is a binding, not a literal."""
    t = _success("Compute the 3-period moving average of [1, 2, 3, 4, 5].",
                 "result = [mean(w) for w in windows([1, 2, 3, 4, 5], 3)]\nfinish(result)")
    sk = heuristic_distill_skill(t)
    assert sk is not None
    assert "for w in windows(xs, 3)" in sk.body
    assert sk.signature == "moving_average(xs)"


def test_generated_skill_is_valid_python():
    t = _success("Compute the sum of squares of the even numbers in [1, 2, 3, 4].",
                 "result = total(squares(evens([1, 2, 3, 4])))\nfinish(result)")
    sk = heuristic_distill_skill(t)
    assert sk is not None
    compile(sk.code, "<test>", "exec")  # raises SyntaxError if malformed


def test_quoted_data_does_not_pollute_skill_name():
    t = _success('Is "Level" a palindrome after normalising case?',
                 'result = norm("Level") == rev(norm("Level"))\nfinish(result)')
    sk = heuristic_distill_skill(t)
    assert sk is not None
    assert "level" not in sk.name
    assert sk.name == "palindrome"


def test_no_skill_from_failure():
    t = _failure("Find the second largest value in [5, 5, 4, 3].", "5")
    assert heuristic_distill_skill(t) is None


def test_pitfall_from_failure():
    t = _failure("Find the second largest value in [5, 5, 4, 3].", "5")
    pf = heuristic_distill_pitfall(t)
    assert pf is not None
    assert "5" in pf.description
    assert pf.tags, "pitfalls need tags to be retrievable"


def test_no_pitfall_from_success():
    t = _success("Total all the numbers in [1, 2].", "result = total([1, 2])\nfinish(result)")
    assert heuristic_distill_pitfall(t) is None


def test_constant_expression_not_worth_a_skill():
    """`result = 5` has no tool call -- nothing generalisable here."""
    t = _success("Return five.", "result = 5\nfinish(result)")
    assert heuristic_distill_skill(t) is None


def test_distiller_routes_by_outcome():
    d = Distiller(adapter=None)
    ok = _success("Find the second largest value in [5, 5, 4, 3].",
                  "result = nth(dedup_sorted_desc([5, 5, 4, 3]), 1)\nfinish(result)")
    bad = _failure("Find the second largest value in [5, 5, 4, 3].", "5")

    r1 = d.distill(ok)
    assert r1.skill is not None and r1.pitfall is None

    r2 = d.distill(bad)
    assert r2.pitfall is not None and r2.skill is None
