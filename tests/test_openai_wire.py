"""Wire-level tests for the OpenAI-compatible (Kimi / OpenRouter) path.

These matter because Evolver's hosted-model story rests on one assumption:
that a ``kimi`` config produces bytes Moonshot and OpenRouter will accept.
The mock server lets CI assert that with no key, no GPU, and no network.

These are **plumbing** tests, not quality tests: the responses are synthetic,
so a green run means "the request is well-formed and the pipeline converges",
never "Kimi K3 is good at this".
"""

from __future__ import annotations

import os
import unittest

from evolver.bench.runner import run_benchmark
from evolver.bench.tasks import BENCH_TASKS
from evolver.core.config import Config
from evolver.llm.mock_server import MockOpenAIServer
from evolver.llm.vendors import OpenAIAdapter

MOCK_KEY_ENV = "MOCK_API_KEY"


class WireTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ[MOCK_KEY_ENV] = "sk-mock-local"
        self.server = MockOpenAIServer(model="kimi-k3").start()
        # register_secret leaks a `serve_forever` thread and eventually trips
        # `RuntimeError: can't start new thread`. The cleanup closure holds the
        # instance, so replacing self.server elsewhere cannot orphan it.
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        server, self.server = self.server, None
        if server is not None:
            server.stop()

    def _adapter(self, **kw) -> OpenAIAdapter:
        from evolver.core.config import LLMConfig

        cfg = LLMConfig(provider="kimi", model="kimi-k3",
                        base_url=self.server.base_url, api_key_env=MOCK_KEY_ENV)
        for k, v in kw.items():
            setattr(cfg, k, v)
        return OpenAIAdapter(config=cfg)

    # -- request shape ---------------------------------------------------
    def test_request_carries_kimi_model_id(self):
        self._adapter().complete("sys", "TASK:\nFind the largest of [1,2].")
        self.assertEqual(self.server.models_called, ["kimi-k3"])

    def test_request_is_openai_shaped(self):
        self._adapter().complete("system text", "user text")
        body = self.server.last_body
        self.assertEqual(sorted(body), ["max_tokens", "messages", "model", "temperature"])
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["system", "user"])
        self.assertEqual(body["messages"][0]["content"], "system text")
        self.assertEqual(body["messages"][1]["content"], "user text")

    def test_bearer_auth_header_is_sent(self):
        self._adapter().complete("sys", "usr")
        auth = self.server.requests[-1]["authorization"] or ""
        self.assertTrue(auth.startswith("Bearer "), auth)
        self.assertIn("sk-mock-local", auth)

    def test_temperature_and_max_tokens_are_forwarded(self):
        self._adapter().complete("sys", "usr", temperature=0.7, max_tokens=123)
        body = self.server.last_body
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["max_tokens"], 123)

    # -- response parsing ------------------------------------------------
    def test_response_text_and_usage_are_parsed(self):
        resp = self._adapter().complete("sys", "TASK:\nFind the largest of [1,2].")
        self.assertTrue(resp.ok, resp.error)
        self.assertIn("```python", resp.text)
        self.assertGreater(resp.tokens_in, 0)
        self.assertGreater(resp.tokens_out, 0)
        self.assertEqual(resp.model, "kimi-k3")

    def test_server_error_surfaces_not_raises(self):
        # Complete against a closed port: the adapter must degrade, not explode.
        self.server.stop()
        resp = self._adapter().complete("sys", "usr")
        self.assertFalse(resp.ok)
        self.assertIsNotNone(resp.error)
        # Bring a live server back for any later use, replacing the dead one.
        # The cleanup installed in setUp holds the *old* instance, so this
        # swap cannot orphan its thread.
        self.server = MockOpenAIServer(model="kimi-k3").start()

    # -- end-to-end over TCP ---------------------------------------------
    def test_benchmark_converges_over_http(self):
        cfg = Config.for_provider("kimi", model="kimi-k3")
        cfg.llm.base_url = self.server.base_url
        cfg.llm.api_key_env = MOCK_KEY_ENV
        cfg.store_path = ".evolver/skills-wire-test.json"

        with MockOpenAIServer(model="kimi-k3", tasks=list(BENCH_TASKS)) as srv:
            cfg.llm.base_url = srv.base_url
            result = run_benchmark(cfg, tasks=list(BENCH_TASKS), epochs=3, seed=42)

        self.assertGreater(len(srv.requests), 0)
        self.assertEqual(set(srv.models_called), {"kimi-k3"})
        first, last = result.epochs[0], result.epochs[-1]
        self.assertGreaterEqual(last.success_rate, first.success_rate)
        self.assertLess(last.avg_tokens, first.avg_tokens)

    def test_wire_result_matches_replay_result(self):
        """HTTP transport must not change behaviour -- only how bytes travel."""
        tasks = list(BENCH_TASKS)

        replay_cfg = Config.for_provider("replay")
        replay_cfg.store_path = ".evolver/skills-wire-replay.json"
        replay = run_benchmark(replay_cfg, tasks=tasks, epochs=2, seed=42)

        with MockOpenAIServer(model="kimi-k3", tasks=tasks) as srv:
            kimi_cfg = Config.for_provider("kimi", model="kimi-k3")
            kimi_cfg.llm.base_url = srv.base_url
            kimi_cfg.llm.api_key_env = MOCK_KEY_ENV
            kimi_cfg.store_path = ".evolver/skills-wire-kimi.json"
            kimi = run_benchmark(kimi_cfg, tasks=tasks, epochs=2, seed=42)

        self.assertEqual([e.success_rate for e in replay.epochs],
                         [e.success_rate for e in kimi.epochs])
        self.assertEqual(sorted(replay.skill_names), sorted(kimi.skill_names))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
