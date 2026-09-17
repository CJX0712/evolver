"""Tests for the deployment budget planner.

These exist because of a specific, expensive mistake: a 45 GB model was
downloaded onto a machine whose process was capped at 8 GB, and the failure
surfaced only as a silent death during load. ``free`` said 123 GB, so nothing
looked wrong. The planner is the fix -- it reads the cgroup before deciding,
and these tests pin both the arithmetic and the reason it exists.
"""

from __future__ import annotations

import unittest

from examples.plan_local import (
    Budget,
    GIB,
    current_budget,
    kv_cache_bytes,
    plan,
)


class BudgetTests(unittest.TestCase):
    def test_cgroup_limit_overrides_host_ram(self):
        """The host number must not be allowed to mask a tighter cgroup cap."""
        b = Budget(cgroup_max=8 * GIB, cgroup_swap=8 * GIB,
                   host_total=123 * GIB)
        # 8 GB hard + half of 8 GB swap = 12 GB, nowhere near the 123 GB host.
        self.assertEqual(b.effective_bytes, 12 * GIB)
        self.assertLess(b.effective_bytes, b.host_total / 5)

    def test_no_cgroup_means_host_ram_is_the_budget(self):
        b = Budget(cgroup_max=None, cgroup_swap=None, host_total=64 * GIB)
        self.assertEqual(b.effective_bytes, 64 * GIB)

    def test_describe_flags_a_misleading_host_number(self):
        b = Budget(cgroup_max=8 * GIB, cgroup_swap=0, host_total=123 * GIB)
        self.assertIn("misleading", b.describe())

    def test_describe_is_quiet_when_host_and_cgroup_agree(self):
        b = Budget(cgroup_max=100 * GIB, cgroup_swap=0, host_total=123 * GIB)
        self.assertNotIn("misleading", b.describe())

    def test_swap_is_discounted(self):
        """Swap counts, but not at face value -- it means disk-speed inference."""
        with_swap = Budget(cgroup_max=8 * GIB, cgroup_swap=8 * GIB,
                           host_total=123 * GIB)
        without = Budget(cgroup_max=8 * GIB, cgroup_swap=0, host_total=123 * GIB)
        self.assertEqual(without.effective_bytes, 8 * GIB)
        self.assertEqual(with_swap.effective_bytes - without.effective_bytes, 4 * GIB)


class PlanTests(unittest.TestCase):
    def test_oversized_model_is_rejected(self):
        """The exact model that failed in production must be rejected."""
        b = Budget(cgroup_max=8 * GIB, cgroup_swap=8 * GIB, host_total=123 * GIB)
        original = current_budget
        try:
            import examples.plan_local as m
            m.current_budget = lambda: b
            r = plan(45.17, context=8192, model_key="qwen3-30b-a3b")
            self.assertFalse(r["fits"])
            self.assertLess(r["headroom_gb"], 0)
        finally:
            m.current_budget = original

    def test_model_that_fits_is_accepted(self):
        import examples.plan_local as m
        original = m.current_budget
        try:
            m.current_budget = lambda: Budget(cgroup_max=8 * GIB, cgroup_swap=8 * GIB,
                                              host_total=123 * GIB)
            r = plan(8.38, context=8192, model_key="qwen3-14b")
            self.assertTrue(r["fits"])
            self.assertGreater(r["headroom_gb"], 0)
        finally:
            m.current_budget = original

    def test_kv_cache_grows_with_context(self):
        small = kv_cache_bytes("qwen3-14b", 2048)
        large = kv_cache_bytes("qwen3-14b", 8192)
        self.assertEqual(large, small * 4)

    def test_larger_context_needs_more_memory(self):
        import examples.plan_local as m
        original = m.current_budget
        try:
            m.current_budget = lambda: Budget(cgroup_max=None, cgroup_swap=0,
                                              host_total=64 * GIB)
            a = plan(8.38, context=2048, model_key="qwen3-14b")
            b = plan(8.38, context=32768, model_key="qwen3-14b")
            self.assertLess(a["total_gb"], b["total_gb"])
        finally:
            m.current_budget = original

    def test_response_reports_a_usable_max_context_when_rejected(self):
        """A rejection must be actionable -- it should say what *would* work."""
        import examples.plan_local as m
        original = m.current_budget
        try:
            # 8 GB cap, no swap, 8.38 GB of weights: no room for any context.
            m.current_budget = lambda: Budget(cgroup_max=8 * GIB, cgroup_swap=0,
                                              host_total=123 * GIB)
            too_big = plan(8.38, context=8192, model_key="qwen3-14b")
            self.assertFalse(too_big["fits"])
            self.assertEqual(too_big["max_context_that_fits"], 0)

            # Drop to a 6.0 GB quant. At 8192 context it is still 0.45 GB over
            # an 8 GB cap, so it correctly fails -- but the planner must also
            # report the largest context that *would* fit (5242 tokens), because
            # a bare "no" leaves the caller with nowhere to go.
            workable = plan(6.0, context=8192, model_key="qwen3-14b")
            self.assertGreater(workable["max_context_that_fits"], 0)
            self.assertLess(workable["max_context_that_fits"], 8192)

            # At a context the planner says fits, it must actually fit.
            ok = plan(6.0, context=workable["max_context_that_fits"],
                      model_key="qwen3-14b")
            self.assertTrue(ok["fits"], ok)
        finally:
            m.current_budget = original

    def test_max_context_zero_when_weights_exceed_budget(self):
        import examples.plan_local as m
        original = m.current_budget
        try:
            m.current_budget = lambda: Budget(cgroup_max=4 * GIB, cgroup_swap=0,
                                              host_total=123 * GIB)
            r = plan(6.82, context=8192, model_key="qwen3-14b")
            self.assertFalse(r["fits"])
            self.assertEqual(r["max_context_that_fits"], 0)
        finally:
            m.current_budget = original

    def test_weights_dominate_at_small_context(self):
        import examples.plan_local as m
        original = m.current_budget
        try:
            m.current_budget = lambda: Budget(cgroup_max=None, cgroup_swap=0,
                                              host_total=64 * GIB)
            r = plan(30.0, context=1024, model_key="qwen3-14b")
            self.assertGreater(r["weights_gb"], r["kv_cache_gb"] * 10)
        finally:
            m.current_budget = original


class LiveEnvironmentTest(unittest.TestCase):
    def test_current_budget_is_readable(self):
        """Whatever the sandbox reports, it must not raise."""
        b = current_budget()
        self.assertGreater(b.host_total, 0)
        self.assertGreater(b.effective_bytes, 0)

    def test_effective_budget_never_exceeds_host(self):
        b = current_budget()
        self.assertLessEqual(b.effective_bytes, b.host_total)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
