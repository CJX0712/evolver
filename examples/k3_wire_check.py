"""Prove the Kimi K3 path works -- without a K3, a key, or a GPU.

Run it::

    python3.11 examples/k3_wire_check.py

What it does
------------
Stands up a local OpenAI-compatible server (the same wire format Moonshot and
the OpenRouter K3 providers speak), points the ``kimi`` provider at it, and
runs the *full* benchmark over TCP: CodeAct execution, the 4-layer sandbox,
skill injection, distillation, evolution. Then it prints the exact request the
server received.

What it proves -- and what it does not
--------------------------------------
It proves the plumbing: correct model id, correct auth header, correct
chat-completions shape, and an end-to-end run that converges. It does **not**
prove anything about K3 quality, because the responses here are synthetic.
Swap ``base_url`` for ``https://api.moonshot.cn/v1`` and set
``MOONSHOT_API_KEY``, and the identical code path talks to the real 2.8T model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolver.bench.runner import run_benchmark  # noqa: E402
from evolver.bench.tasks import BENCH_TASKS  # noqa: E402
from evolver.core.config import Config  # noqa: E402
from evolver.llm.mock_server import MockOpenAIServer  # noqa: E402

EPOCHS = 3


def main() -> int:
    os.environ.setdefault("MOCK_API_KEY", "sk-mock-local")

    # tasks= lets the server-side replay adapter recognise each benchmark task
    # from the prompt text, so it can answer at the right competence level.
    with MockOpenAIServer(model="kimi-k3", tasks=list(BENCH_TASKS)) as srv:
        cfg = Config.for_provider("kimi", model="kimi-k3")
        cfg.llm.base_url = srv.base_url
        cfg.llm.api_key_env = "MOCK_API_KEY"
        cfg.store_path = ".evolver/skills-wire-check.json"

        print(f"endpoint : {srv.base_url}")
        print(f"provider : {cfg.llm.provider}   model: {cfg.llm.model}")
        print(f"tasks    : {len(BENCH_TASKS)} x {EPOCHS} epochs\n")

        result = run_benchmark(cfg, tasks=list(BENCH_TASKS), epochs=EPOCHS, seed=42)

        print("\n--- wire ---")
        print(f"requests received      : {len(srv.requests)}")
        print(f"distinct model ids sent: {sorted(set(srv.models_called))}")
        first = srv.requests[0]
        print(f"authorization header   : {(first['authorization'] or '')[:24]}...")
        print(f"first request keys      : {sorted(first['body'].keys())}")

        print("\n--- first request payload (truncated) ---")
        preview = dict(first["body"])
        for msg in preview.get("messages", []):
            content = msg.get("content", "")
            if len(content) > 220:
                msg["content"] = content[:220] + f" ...[{len(content)} chars total]"
        print(json.dumps(preview, indent=2, ensure_ascii=False)[:1600])

        print("\n--- benchmark over HTTP ---")
        for k, v in result.summary().items():
            print(f"  {k:22s} {v}")

        print("\n--- per-epoch ---")
        for e in result.epochs:
            print(f"  epoch {e.epoch}: success={e.success_rate:.0%}  "
                  f"steps={e.avg_steps:.2f}  tokens={e.avg_tokens:.0f}")

        print("\nskills distilled:", ", ".join(result.skill_names) or "(none)")

        converged = (
            len(result.epochs) >= 2
            and result.epochs[-1].success_rate >= result.epochs[0].success_rate
        )
        print("\nwire check:", "PASS" if converged and srv.requests else "FAIL")
        return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
