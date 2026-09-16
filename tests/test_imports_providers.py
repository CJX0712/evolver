"""Import hygiene and provider wiring.

Two regressions worth guarding:

1. A circular import once made ``from evolver.llm import build_adapter`` fail
   outright (llm -> bench -> core.engine -> llm). Each layer must import on its
   own, verified here in a fresh interpreter.
2. Provider presets are easy to typo and the failure only shows up at the first
   API call, so assert the endpoint wiring directly.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from evolver.core.config import Config

MODULES = [
    "evolver",
    "evolver.core.types",
    "evolver.core.config",
    "evolver.core.engine",
    "evolver.act.sandbox",
    "evolver.act.codeact",
    "evolver.memory.context",
    "evolver.evolve.distill",
    "evolver.evolve.curator",
    "evolver.llm",
    "evolver.llm.replay",
    "evolver.bench.tasks",
    "evolver.bench.runner",
    "evolver.cli",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_isolation(module):
    """`import X` must work from a cold interpreter, not just after a package import."""
    r = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"import {module} failed:\n{r.stderr}"


def test_kimi_preset_points_at_moonshot():
    cfg = Config.for_provider("kimi")
    assert cfg.llm.provider == "kimi"
    assert "moonshot" in (cfg.llm.base_url or "")
    assert cfg.llm.api_key_env == "MOONSHOT_API_KEY"


def test_openrouter_preset_uses_kimi_k3_slug():
    cfg = Config.for_provider("openrouter")
    assert "openrouter.ai" in (cfg.llm.base_url or "")
    assert "kimi" in cfg.llm.model.lower()
    assert cfg.llm.api_key_env == "OPENROUTER_API_KEY"


def test_openai_compatible_providers_reuse_the_openai_adapter():
    """Kimi K3 is served over an OpenAI-compatible wire format."""
    from evolver.llm.vendors import OpenAIAdapter

    import os
    env = {"MOONSHOT_API_KEY": "test-key", "OPENROUTER_API_KEY": "test-key",
           "OPENAI_API_KEY": "test-key"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        from evolver.llm import build_adapter

        for provider in ("kimi", "openrouter", "openai"):
            adapter = build_adapter(Config.for_provider(provider))
            assert isinstance(adapter, OpenAIAdapter), provider
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_unknown_provider_raises():
    from evolver.llm import build_adapter

    with pytest.raises(ValueError, match="unknown provider"):
        build_adapter(Config.for_provider("nope-provider"))


def test_missing_key_raises_actionable_error():
    """A missing key should say which env var to set, not blow up obscurely."""
    import os

    os.environ.pop("MOONSHOT_API_KEY", None)
    with pytest.raises(ValueError, match="MOONSHOT_API_KEY"):
        from evolver.llm import build_adapter

        build_adapter(Config.for_provider("kimi"))


def test_replay_adapter_has_no_bench_import_dependency():
    """The adapter layer must not import the benchmark layer at module scope."""
    import inspect

    from evolver.llm import replay

    src = inspect.getsource(replay)
    assert "from evolver.bench" not in src.replace(
        "from evolver.bench.tasks import find_task  # deferred", ""
    )


def test_config_roundtrip_preserves_base_url():
    cfg = Config.for_provider("openrouter")
    cfg.llm.base_url = "http://localhost:11434/v1"
    restored = Config.from_dict(cfg.to_dict())
    assert restored.llm.base_url == "http://localhost:11434/v1"
