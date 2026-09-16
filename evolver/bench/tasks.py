"""The benchmark task set.

Each task carries three things: the problem, the ground truth, and a
description of how an *unskilled* agent behaves on it -- how many turns it
fumbles through, what wrong answer it lands on, and what trap it fell into.

That third part is what makes the benchmark meaningful. Every task here has a
concrete trap that a naive tool composition walks into:

* sorting without de-duplicating, so a duplicate wins "second largest"
* ``odds`` instead of ``evens``
* ``merge_overwrite`` instead of ``merge_sum``
* ``flatten1`` on doubly-nested input
* ``windows_truncated`` (off by one, drops the last window)
* comparing a string to its reverse without normalising case

A skill is only worth having if it encodes the composition that avoids the
trap. That is exactly what gets distilled from the trajectories that succeed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

STOPWORDS = frozenset(
    """a an the in of on at by for from with to and or is are was were be been
    find compute get return give what which that this it its please answer just
    after normalising normalizing case punctuation fully completely up all
    numbers number period""".split()
)


def _name_from(text: str) -> str:
    """Derive a skill function name from task prose."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text)
    keep = [w.lower() for w in words if w.lower() not in STOPWORDS]
    keep = keep[:5] or ["task"]
    name = "_".join(keep)
    return re.sub(r"_+", "_", name).strip("_") or "task"


def _normalise(value: Any) -> Any:
    """Loose canonical form so '4' == 4.0 == '4.0'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return str(value).strip().lower()


def _try_parse(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def make_verifier(expected: Any) -> Callable[[Any], bool]:
    """Build a lenient equality check against ground truth."""
    exp_norm = _normalise(expected)

    def verify(answer: Any) -> bool:
        if answer is None:
            return False
        text = str(answer).strip()
        if not text:
            return False
        parsed = _try_parse(text)
        if parsed is not None and _normalise(parsed) == exp_norm:
            return True
        return _normalise(text) == exp_norm

    return verify


@dataclass
class BenchTask:
    """One benchmark problem, plus the unskilled-agent behaviour on it."""

    task_id: str
    prompt: str
    expected: Any
    args: List[str] = field(default_factory=list)
    match: str = ""
    naive_turns: int = 3
    naive_code: str = "# turn {i}: look at the data\nd = {a0}\nprint(d)\n"
    naive_final: str = "result = {a0}\nfinish(result)\n"
    warned_final: str = "result = {a0}\nfinish(result)\n"
    skill_marker: str = ""
    failure_mode: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.match:
            self.match = re.escape(self.task_id)
        if not self.skill_marker:
            self.skill_marker = _name_from(self.prompt)

    @property
    def skill_name(self) -> str:
        return _name_from(self.prompt)

    def verify(self, answer: Any) -> bool:
        return make_verifier(self.expected)(answer)

    def render_naive(self, turn: int) -> str:
        return self.naive_code.format(i=turn, **self._fmt())

    def render_final(self, warned: bool) -> str:
        template = self.warned_final if warned else self.naive_final
        return template.format(**self._fmt())

    def _fmt(self) -> dict:
        d = {f"a{i}": a for i, a in enumerate(self.args)}
        d.setdefault("a0", "[]")
        d.setdefault("a1", "[]")
        d.setdefault("a2", "[]")
        return d


BENCH_TASKS: List[BenchTask] = [
    BenchTask(
        task_id="second_largest",
        prompt="Find the second largest value in [5, 5, 4, 3].",
        expected=4,
        args=["[5, 5, 4, 3]"],
        match=r"second largest",
        naive_turns=3,
        naive_code="# turn {i}: sort and inspect\nd = {a0}\nprint('sorted', sort_desc(d))\n",
        naive_final="result = nth(sort_desc({a0}), 1)\nfinish(result)\n",
        warned_final="result = nth(dedup_sorted_desc({a0}), 1)\nfinish(result)\n",
        skill_marker="second_largest",
        failure_mode="Sorting without de-duplicating returns a duplicate as the "
                     "'second largest' value.",
        tags=["ranking", "dedup-trap"],
    ),
    BenchTask(
        task_id="most_frequent_word",
        prompt='Find the most frequent word in "the The THE cat cat dog" (answer lowercase).',
        expected="the",
        args=['"the The THE cat cat dog"'],
        match=r"most frequent word",
        naive_turns=3,
        naive_code="# turn {i}: count words\nd = {a0}\nprint(count_words(d))\n",
        naive_final="result = max_key(count_words({a0}))\nfinish(result)\n",
        warned_final="result = max_key(count_words_normalized({a0}))\nfinish(result)\n",
        skill_marker="frequent_word",
        failure_mode="Counting words without normalising case splits 'the'/'The'/'THE' "
                     "into separate keys, so a genuinely less frequent word wins.",
        tags=["text", "normalisation-trap"],
    ),
    BenchTask(
        task_id="sum_even_squares",
        prompt="Compute the sum of squares of the even numbers in [1, 2, 3, 4, 5, 6].",
        expected=56,
        args=["[1, 2, 3, 4, 5, 6]"],
        match=r"sum of squares",
        naive_turns=3,
        naive_code="# turn {i}: filter\nd = {a0}\nprint('parts', d)\n",
        naive_final="result = total(squares(odds({a0})))\nfinish(result)\n",
        warned_final="result = total(squares(evens({a0})))\nfinish(result)\n",
        skill_marker="squares_even",
        failure_mode="Filtering with `odds` instead of `evens` sums the wrong subset.",
        tags=["math", "filter-trap"],
    ),
    BenchTask(
        task_id="merge_dicts",
        prompt="Merge {'a': 1, 'b': 2} with {'b': 3, 'c': 4}, adding values of shared keys.",
        expected={"a": 1, "b": 5, "c": 4},
        args=["{'a': 1, 'b': 2}", "{'b': 3, 'c': 4}"],
        match=r"[Mm]erge",
        naive_turns=3,
        naive_code="# turn {i}: inspect both\nd = {a0}\nprint(sorted(d.items()))\n",
        naive_final="result = merge_overwrite({a0}, {a1})\nfinish(result)\n",
        warned_final="result = merge_sum({a0}, {a1})\nfinish(result)\n",
        skill_marker="merge",
        failure_mode="Using dict update semantics overwrites shared keys instead of "
                     "adding their values.",
        tags=["dict", "merge-trap"],
    ),
    BenchTask(
        task_id="flatten_deep",
        prompt="Fully flatten [[1, [2, 3]], [4, [5, [6]]]] into a single flat list.",
        expected=[1, 2, 3, 4, 5, 6],
        args=["[[1, [2, 3]], [4, [5, [6]]]]"],
        match=r"[Ff]latten",
        naive_turns=3,
        naive_code="# turn {i}: try one level\nd = {a0}\nprint(flatten1(d))\n",
        naive_final="result = flatten1({a0})\nfinish(result)\n",
        warned_final="result = deep_flatten({a0})\nfinish(result)\n",
        skill_marker="flatten",
        failure_mode="A single-level flatten leaves nested lists intact when the input "
                     "is doubly nested.",
        tags=["list", "recursion-trap"],
    ),
    BenchTask(
        task_id="moving_average",
        prompt="Compute the 3-period moving average of [1, 2, 3, 4, 5].",
        expected=[2, 3, 4],
        args=["[1, 2, 3, 4, 5]"],
        match=r"moving average",
        naive_turns=3,
        naive_code="# turn {i}: look at windows\nd = {a0}\nprint(windows_truncated(d, 3))\n",
        naive_final="result = [mean(w) for w in windows_truncated({a0}, 3)]\nfinish(result)\n",
        warned_final="result = [mean(w) for w in windows({a0}, 3)]\nfinish(result)\n",
        skill_marker="moving_average",
        failure_mode="An off-by-one in the windowing helper silently drops the final "
                     "window, shortening the result.",
        tags=["series", "off-by-one-trap"],
    ),
    BenchTask(
        task_id="palindrome",
        prompt='Is "Level" a palindrome after normalising case and punctuation?',
        expected=True,
        args=['"Level"'],
        match=r"palindrome",
        naive_turns=2,
        naive_code="# turn {i}: compare\nd = {a0}\nprint(d, rev(d))\n",
        naive_final="result = {a0} == rev({a0})\nfinish(result)\n",
        warned_final="result = norm({a0}) == rev(norm({a0}))\nfinish(result)\n",
        skill_marker="palindrome",
        failure_mode="Comparing raw strings to their reverse fails whenever case or "
                     "punctuation differs.",
        tags=["text", "normalisation-trap"],
    ),
    BenchTask(
        task_id="group_and_sum",
        prompt="Group [{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}, {'k': 'a', 'v': 3}] by "
               "key 'k' and sum 'v' within each group.",
        expected={"a": 4, "b": 2},
        args=["[{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}, {'k': 'a', 'v': 3}]"],
        match=r"[Gg]roup",
        naive_turns=3,
        naive_code="# turn {i}: bucket rows\nd = {a0}\nprint(group_by(d, 'k'))\n",
        naive_final="result = group_by({a0}, 'k')\nfinish(result)\n",
        # Braces are doubled because this template goes through str.format.
        warned_final="result = {{k: sum_field(v, 'v') for k, v in group_by({a0}, 'k').items()}}\n"
                     "finish(result)\n",
        skill_marker="group",
        failure_mode="Returning the grouped buckets without aggregating leaves lists "
                     "where totals were asked for.",
        tags=["aggregation", "pipeline-trap"],
    ),
    BenchTask(
        task_id="total_all",
        prompt="Total all the numbers in [10, 20, 30, 40].",
        expected=100,
        args=["[10, 20, 30, 40]"],
        match=r"[Tt]otal all",
        naive_turns=3,
        naive_code="# turn {i}: inspect\nd = {a0}\nprint(len(d))\n",
        naive_final="result = total({a0})\nfinish(result)\n",
        warned_final="result = total({a0})\nfinish(result)\n",
        skill_marker="total",
        failure_mode="",
        tags=["math", "easy"],
    ),
    BenchTask(
        task_id="maximum_number",
        prompt="Find the maximum number in [3, 9, 2, 7].",
        expected=9,
        args=["[3, 9, 2, 7]"],
        match=r"maximum number",
        naive_turns=2,
        naive_code="# turn {i}: sort\nd = {a0}\nprint(sort_desc(d))\n",
        naive_final="result = nth(sort_desc({a0}), 0)\nfinish(result)\n",
        warned_final="result = nth(sort_desc({a0}), 0)\nfinish(result)\n",
        skill_marker="maximum",
        failure_mode="",
        tags=["ranking", "easy"],
    ),
]


def find_task(text: str) -> Optional[BenchTask]:
    """Match free text (a task prompt) back to its spec."""
    for t in BENCH_TASKS:
        if re.search(t.match, text, re.IGNORECASE):
            return t
    return None


def by_id(task_id: str) -> Optional[BenchTask]:
    for t in BENCH_TASKS:
        if t.task_id == task_id:
            return t
    return None
