# Evolver

<p align="center">
  <a href="https://github.com/CJX0712/evolver/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/evolver/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/evolver/releases"><img src="https://img.shields.io/github/v/release/CJX0712/evolver?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/evolver/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/evolver" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

**The self-evolving layer for AI agents.**

Most agents start every task from zero. Explain the same thing tomorrow and
they re-derive it, at full token cost, with the same odds of getting it wrong.
Evolver is the layer that fixes that: it turns each run into reusable
capability, then *measures* whether the agent actually got better.

```
pip install -e .
evolver bench                    # offline, deterministic, ~0.05s
evolver bench --provider openai  # or against a real model
```

---

## The claim, and the evidence

An agent that evolves should show improving curves, not just a growing skill
folder. Here is the built-in benchmark — 10 tasks × 6 epochs, offline:

| epoch | 成功率 | 平均步数 | 平均 Token | 技能数 | 技能命中率 |
|------:|-------:|---------:|-----------:|-------:|-----------:|
| 0     | 30%    | 2.80     | 1536       | 0      | 0%         |
| 1     | 100%   | 2.30     | 1354       | 10     | 30%        |
| 2     | 100%   | 1.00     | 613        | 10     | 100%       |
| 3–5   | 100%   | 1.00     | 613        | 10     | 100%       |

**Net: 成功率 30% → 100%，Token −60%，步数 −64%.**

### Control: is that actually the mechanism, or just repetition?

Run the same benchmark with evolution switched off (`--no-evolve`):

| | 成功率 | 平均步数 | 平均 Token |
|---|---:|---:|---:|
| 进化关闭 | 20% → **20%** | 2.80 → **2.80** | 1522 → **1522** |
| 进化开启 | 30% → **100%** | 2.80 → **1.00** | 1536 → **613** |

Flat without it. The gain is the learning machinery, not exposure.

### The two learning signals are separable

This is the part worth paying attention to:

- **epoch 0 → 1**: 成功率 30% → 100%, 步数 barely moves (2.80 → 2.30).
  That is **Pitfalls** — learning from *failures*. Warned about a trap, the
  agent stops walking into it, but still reasons it out step by step.
- **epoch 1 → 2**: 步数 2.30 → 1.00, Token 1354 → 613.
  That is **Skills** — learning from *successes*. A distilled function
  collapses a multi-turn derivation into one call.

If success rises but steps don't fall, pitfalls work and skills don't. That is
far more diagnosable than "the agent improved".

> **Honesty about these numbers.** The default `replay` adapter is a *scripted*
> agent, not a language model. These results validate the **mechanism**. They
> say nothing about how GPT/Claude/Gemini behave — for that, plug in a real
> provider and rerun. The report says this too, in a box, so nobody quotes it
> wrong.

---

## Four mechanisms

### 1. CodeAct — one program, not N round trips

The usual loop is a ping-pong match: pick a tool, wait, observe, repeat. Each
arrow is a full model turn that re-sends the whole history.

CodeAct inverts it. The model writes **one Python block** that calls the tools
it needs; it runs once in a sandbox; one consolidated result comes back.

### 2. Skill distillation — successes become functions

A successful run is lifted into a callable: the final expression is extracted,
data literals are promoted to parameters, and the result is a real function.

```python
# distilled from `nth(dedup_sorted_desc([5, 5, 4, 3]), 1)`
def second_largest(xs):
    return nth(dedup_sorted_desc(xs), 1)
```

It generalises — distilled on `[5, 5, 4, 3]`, reused on `[9, 9, 8, 2, 7]`:

```
Find the second largest value in [5, 5, 4, 3].    success  3 steps  1616 tok
Find the second largest value in [9, 9, 8, 2, 7]. success  1 step    561 tok  ← reused
Find the second largest value in [4, 4, 4, 1].    success  1 step    560 tok  ← reused
```

### 3. Pitfall distillation — failures become warnings

Most frameworks only learn from success. That discards the cheaper, more
specific half of the signal. A failure yields a `Pitfall` that is surfaced as a
warning on the next similar task.

### 4. Layered context + autonomous curation

A **30K token budget** (not 200K–1M) with automatic compaction across three
tiers — resident goal, recent steps verbatim, older steps archived. A smaller
curated window is cheaper *and* more accurate, because noise is what derails
reasoning.

An unattended library rots three ways, and each has a remedy: **dead weight**
(prune by score), **fragmentation** (merge near-duplicates, pool their stats),
**bloat** (hard cap).

---

## Security

Model-generated code is untrusted — a prompt-injected document can steer the
model into emitting hostile code. Four independent layers:

| Layer | Mechanism |
|---|---|
| Static | AST allowlist; dunder attributes, `eval`/`exec`/`open`/`__import__` rejected |
| Runtime | Restricted builtins + an `__import__` that re-enforces the allowlist |
| Time | `SIGALRM` wall clock kills runaway loops |
| Memory | `RLIMIT_AS` caps the address space |

```python
sandbox.run("print(().__class__.__bases__)")  # SecurityViolation
sandbox.run("import os")                      # SecurityViolation
sandbox.run("while True: pass")               # timeout
```

Any single layer has known bypasses. The combination makes escape impractical
rather than merely unlikely.

---

## Quick start

```bash
pip install -e .
evolver bench --epochs 6 --out report.html
evolver skills                       # inspect the learned library
evolver run "Find the second largest value in [5, 5, 4, 3]."
```

Programmatic use:

```python
from evolver import Evolver, Config
from evolver.bench.tools import TOOLS

engine = Evolver(config=Config(), tools=list(TOOLS))
traj = engine.run("Fully flatten [[1, [2, 3]], [4, [5, [6]]]] into a single flat list.")
print(traj.status, traj.final_answer, traj.n_steps, traj.total_tokens)
engine.save()
```

### Moving a learned library between machines

Running the benchmark writes what it learned to `--store`. A finished run is
therefore a distributable artifact — one file, no weights:

```bash
evolver bench --epochs 6 --store alice.json          # learn
evolver pack export --store alice.json --name bench-v1 --author alice --out bench-v1.evp
evolver pack inspect bench-v1.evp                    # verify before shipping
evolver pack import bench-v1.evp --store bob.json    # merge into someone else
```

Import is **additive** — it can raise a local counter, never lower one — and
every skill's source is re-vetted through the same static sandbox check as any
other generated code, so a pack cannot smuggle in `os.system`. Imported
credibility is *discounted*: inherited wins are a prior, not a promotion, and a
skill with a single off-task win loses it in transit rather than arriving
pre-proven.

```
$ evolver pack import bench-v1.evp --store fresh.json
10 added, 0 merged, 0 skipped, 7 pitfalls
```

### Loading an existing library

Loading is **opt-in** via `--resume`, and that is deliberate. When it was
implicit, running the benchmark twice produced a second run whose epoch-0
baseline was already trained on the first run's skills — a curve measuring
nothing. `--no-save` runs an experiment without touching the store.

```bash
evolver bench --store alice.json --resume    # continue from a saved library
evolver bench --no-save                      # measure without writing
```

### With a real model

```bash
export OPENAI_API_KEY=sk-...
evolver bench --provider openai --model gpt-4o-mini

export ANTHROPIC_API_KEY=sk-ant-...
evolver bench --provider anthropic --model claude-sonnet-4-20250514
```

LLM-driven distillation activates automatically with a live provider — a model
generalises the trajectory instead of the heuristic lifting the final
expression, and models the *reason* a failure happened.

### Kimi K3

Kimi K3 (2.8T MoE, 896 experts / 16 active, 1M context) cannot be self-hosted
here — or on most hardware:

| | size |
|---|---:|
| BF16 weights | 5,178 GB |
| FP8 weights | 2,589 GB |
| INT4 weights (smallest) | **1,295 GB** |
| Minimum VRAM to serve | **~1,680 GB** |
| Minimum deployment | 6× B300 or 8× H200 |

It is reachable through an OpenAI-compatible endpoint instead, which is what
these presets wire up:

```bash
# OpenRouter — 16 providers host K3; $3/$15 per 1M in/out
export OPENROUTER_API_KEY=sk-or-v1-...
evolver bench --provider openrouter            # moonshotai/kimi-k3

# Moonshot's own platform — $0.30/M on cache hits (90% off)
export MOONSHOT_API_KEY=sk-...
evolver bench --provider kimi

# Any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, ...)
evolver bench --provider openai --base-url http://localhost:11434/v1 --model qwen3:30b
```

Self-hosting only breaks even above roughly **2 billion output tokens/month**,
against ~$23k–37k/month for an 8×H200 node. Below that, or with bursty demand,
the API is cheaper and involves no cluster to babysit.

---

## Architecture

```
evolver/
├── core/       types · config · engine (the loop)
├── act/        sandbox (4-layer isolation) · codeact (one-shot execution)
├── memory/     tokens · context (3-tier compaction)
├── evolve/     distill (skill + pitfall) · store · curator
├── llm/        base · replay (offline) · vendors (OpenAI / Anthropic)
└── bench/      tasks · tools · runner · report (single-file HTML)
```

Zero required dependencies in core. `openai`, `anthropic`, `tiktoken` are
optional extras.

The benchmark is deliberately built on **low-level primitives** — `evens` and
`odds`, `flatten1` and `deep_flatten`, `merge_sum` and `merge_overwrite`. The
*composition* is what must be learned. If the tools were high-level, there
would be nothing to learn and the benchmark would measure nothing.

---

## Limitations

Stated plainly, because a benchmark that hides its caveats is marketing:

1. **Retrieval is lexical.** Skills and pitfalls are matched on keyword
   overlap, not embeddings. It is fast and dependency-free, but it will miss
   paraphrases. Embedding-based retrieval is the obvious next step.
2. **Cross-task transfer is currently ~zero, and we measured it.** Splitting
   the suite in half — train on five tasks, then evaluate on the five never
   seen — a cold agent and a pack-seeded agent both land on **20%**. Seeding
   buys nothing on held-out tasks, in either success or tokens.

   Getting to that number required fixing two retrieval bugs that had been
   *manufacturing* a fake transfer signal:

   - `sum_squares_even` carries the trigger `"even squares sum"`. The task
     `group_and_sum` says "…and **sum** 'v' within each group". Sharing one
     generic word was enough to inject the skill, and any perturbation to a
     run that happens to succeed was booked as a generalisation win.
   - The same flaw in `Pitfall.matches` meant a warning about *even squares*
     fired on *group and sum*, which flipped that task from failure to
     success.

   Both are now gated on a **specific** (non-generic) term matching, and
   off-task credit additionally requires a shared specific term. The old
   scoreboard read "+20% transfer"; the true figure is 0%. See
   `tests/test_retrieval_specificity.py`.

   This is the honest state of the art for a 10-task suite: skills transfer
   within a task family (shown above, `[5, 5, 4, 3]` → `[9, 9, 8, 2, 7]`) but
   not across families. A library that generalises needs either real semantic
   retrieval or enough tasks that families overlap.
3. **`replay` results are mechanism validation, not model evaluation.**
4. **Heuristic distillation lifts the final expression.** It cannot invent
   control flow the trajectory did not contain. The LLM path is stronger; the
   heuristic path exists so the pipeline runs with no API key.
5. **Ten tasks is a small suite.** Enough to show the mechanism and its
   ablation, not enough to rank models. Five held-out tasks means a single
   flip moves the score 20 points — which is exactly why the transfer study
   above reports per-task outcomes and repeated seeds, not one aggregate.

---

## Testing

```bash
pytest -q              # 152 tests, ~15s
python3.11 scripts/check.py   # full local gate, ~20s
```

Includes a control-group test (`test_control_run_does_not_improve`) that fails
if the benchmark's improvement could be explained by repetition alone, and
`tests/test_retrieval_specificity.py`, which pins the two retrieval bugs above
so the fake-transfer signal cannot come back.

### The local gate

`scripts/check.py` runs three stages, cheapest first, and stops at the first
failure:

| Stage | What it proves |
|---|---|
| `import` | the package imports and its public surface is intact |
| `tests` | the suite passes |
| `regression` | **evolution still works** — success rises, token cost falls, and both beat a no-evolve control |

The third stage exists because the first two cannot cover it. A test suite
proves the code does what the tests say; it cannot notice that the *learning*
quietly stopped. Verified by breaking it: making skill selection return `[]`
left the mechanism dead but was caught as `token cost did not fall:
1522 -> 1641` — success rate was still climbing, so nothing else complained.

Use it as a pre-push hook if you want it enforced:

```bash
printf '#!/bin/sh\npython3.11 scripts/check.py || exit 1\n' > .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

---

## License

MIT
