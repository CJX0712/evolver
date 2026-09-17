"""Configuration for Evolver.

All tunables live here so an experiment is a single reproducible object.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    """Which brain to use, and how to reach it."""

    provider: str = "replay"          # replay | openai | anthropic
    model: str = "replay-simulated"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 60.0

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env) or None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextConfig:
    """Context budget.

    Default target is 30K tokens -- deliberately an order of magnitude below
    the 200K-1M windows most agents run at. A smaller, curated window is
    cheaper and produces fewer hallucinations, because noise is what derails
    reasoning, not missing context.
    """

    max_tokens: int = 30_000
    compact_threshold: float = 0.80    # compact once we hit 80% of budget
    keep_recent_steps: int = 4         # always keep the last N steps verbatim
    archive_summary_tokens: int = 400  # budget for each compression summary
    reserve_for_answer: int = 2_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SandboxConfig:
    """Limits for executing model-generated code."""

    timeout_s: float = 5.0
    max_output_chars: int = 8_000
    max_memory_mb: int = 256
    allow_imports: tuple = ("math", "json", "re", "datetime", "itertools", "collections", "textwrap")
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["allow_imports"] = list(self.allow_imports)
        return d


@dataclass
class EvolveConfig:
    """Distillation and curation behaviour."""

    distill_enabled: bool = True
    min_steps_to_distill: int = 2       # trivial runs aren't worth a skill
    max_skill_tokens: int = 1_200
    curate_every_n_runs: int = 5
    min_score_to_keep: float = 0.05
    max_skills: int = 200
    max_pitfalls: int = 100
    merge_similarity: float = 0.75      # Jaccard threshold for skill merging
    skill_match_threshold: float = 0.34
    top_k_skills: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    evolve: EvolveConfig = field(default_factory=EvolveConfig)
    store_path: str = ".evolver/skills.json"
    # Whether an existing store at ``store_path`` should be loaded on startup.
    #
    # Defaults to False, and that default is load-bearing. When it was implicit,
    # simply running the benchmark twice produced a second run that started
    # already trained on the first run's skills -- so the "epoch 0" baseline was
    # no longer a baseline, and a curve that measures learning measured nothing.
    # Loading a prior library is a deliberate act; make the caller ask.
    load_existing_store: bool = False
    verbose: bool = False

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.to_dict(),
            "context": self.context.to_dict(),
            "sandbox": self.sandbox.to_dict(),
            "evolve": self.evolve.to_dict(),
            "store_path": self.store_path,
            "load_existing_store": self.load_existing_store,
            "verbose": self.verbose,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        if "llm" in d:
            cfg.llm = LLMConfig(**d["llm"])
        if "context" in d:
            cfg.context = ContextConfig(**d["context"])
        if "sandbox" in d:
            sb = dict(d["sandbox"])
            if "allow_imports" in sb:
                sb["allow_imports"] = tuple(sb["allow_imports"])
            cfg.sandbox = SandboxConfig(**sb)
        if "evolve" in d:
            cfg.evolve = EvolveConfig(**d["evolve"])
        cfg.store_path = d.get("store_path", cfg.store_path)
        cfg.load_existing_store = bool(d.get("load_existing_store", False))
        cfg.verbose = bool(d.get("verbose", False))
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_provider(cls, provider: str, model: Optional[str] = None, **kw) -> "Config":
        presets = {
            "openai": LLMConfig(provider="openai", model=model or "gpt-4o-mini",
                                api_key_env="OPENAI_API_KEY"),
            "anthropic": LLMConfig(provider="anthropic",
                                   model=model or "claude-sonnet-4-20250514",
                                   api_key_env="ANTHROPIC_API_KEY"),
            # Kimi K3 (2.8T MoE) is not self-hostable on ordinary hardware --
            # roughly 1.3 TB of weights at INT4 and ~1.7 TB VRAM to serve. It is
            # reached through an OpenAI-compatible endpoint instead.
            "kimi": LLMConfig(provider="kimi", model=model or "kimi-k3",
                              api_key_env="MOONSHOT_API_KEY",
                              base_url="https://api.moonshot.cn/v1"),
            "openrouter": LLMConfig(provider="openrouter",
                                    model=model or "moonshotai/kimi-k3",
                                    api_key_env="OPENROUTER_API_KEY",
                                    base_url="https://openrouter.ai/api/v1"),
            "replay": LLMConfig(provider="replay", model=model or "replay-simulated"),
        }
        if provider not in presets:
            raise ValueError(f"unknown provider {provider!r}; pick one of {list(presets)}")
        cfg = cls(**kw)
        cfg.llm = presets[provider]
        return cfg
