"""Plan a local model deployment against the machine's *real* limits.

The lesson behind this module
-----------------------------
The first attempt at this served a 45 GB Qwen3-Next-80B and watched it die on
load with no error message. The machine reported 123 GB of RAM, so the model
looked comfortable. It was not: the process lived under a cgroup with an 8 GB
``memory.max``. ``free`` reports the host; the cgroup is what actually applies.
Half an hour of download and two failed loads went into discovering a number
that one file read would have shown.

So this module reads the limit first, then decides what fits::

    python3.11 examples/plan_local.py --model-size 12.8 --context 8192
    python3.11 examples/plan_local.py --list

It is deliberately conservative. The failure mode being guarded against is not
"slightly suboptimal quantisation" -- it is "downloaded 45 GB and nothing
loads", which costs far more than picking a smaller model.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

GIB = 1073741824


def cgroup_memory_limit() -> tuple[int | None, int | None]:
    """Return (max_bytes, swap_max_bytes) from cgroup v2, if present.

    ``None`` means unlimited or not applicable. This is the number that matters;
    ``free`` reports the host and will happily overstate what a container can use.
    """
    def read(name: str) -> int | None:
        p = Path(f"/sys/fs/cgroup/{name}")
        if not p.exists():
            return None
        raw = p.read_text().strip()
        if raw == "max":
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    return read("memory.max"), read("memory.swap.max")


@dataclass
class Budget:
    """What the machine will actually let a process use."""

    cgroup_max: int | None
    cgroup_swap: int | None
    host_total: int

    @property
    def effective_bytes(self) -> int:
        """Memory a model may occupy.

        Weights are resident in RAM (mmap'd pages count once touched), so the
        ceiling is the cgroup limit plus whatever swap we are willing to lean
        on. Swap is counted at half weight: it works, but a model that spills
        into it runs at disk speed, so planning to use it is planning to be slow.
        """
        if self.cgroup_max is None:
            return self.host_total
        return self.cgroup_max + int((self.cgroup_swap or 0) * 0.5)

    def describe(self) -> str:
        lines = [
            f"host RAM        : {self.host_total / GIB:.1f} GB",
            f"cgroup memory   : "
            + ("unlimited" if self.cgroup_max is None
               else f"{self.cgroup_max / GIB:.1f} GB"),
            f"cgroup swap     : "
            + ("none" if not self.cgroup_swap else f"{self.cgroup_swap / GIB:.1f} GB"),
            f"effective budget: {self.effective_bytes / GIB:.1f} GB",
        ]
        if self.cgroup_max is not None and self.cgroup_max < self.host_total * 0.8:
            lines.append("  ** host RAM is misleading here -- the cgroup is the real cap **")
        return "\n".join(lines)


def host_memory_total() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            if hasattr(os, "sysconf") else 0)


def current_budget() -> Budget:
    mx, sw = cgroup_memory_limit()
    return Budget(cgroup_max=mx, cgroup_swap=sw, host_total=host_memory_total())


# KV cache cost per token, fp16, summed over attention layers.
#
# Values are *bytes per token*. Derivation, so the next person can check rather
# than trust:
#
#     per_token = n_layers x n_kv_heads x head_dim x 2 (K and V) x 2 (fp16)
#
# A 14B Qwen3 with 40 layers, 8 KV heads, head_dim 128:
#     40 x 8 x 128 x 2 x 2 = 163,840 bytes/token = 160 KB/token
#     at 8192 context -> 1.25 GB
#
# Earlier revisions of this table carried bare integers like "40 * 1024" with a
# comment calling them "rough constants", which is how a units bug survived: the
# numbers were neither KB nor bytes, so the planner produced verdicts that were
# wrong by three orders of magnitude and confidently rejected models that load
# fine. A planner that launders a guess into a verdict is worse than no planner.
KV_BYTES_PER_TOKEN = {
    "qwen3-30b-a3b": 36 * 1024,      # 48 layers, 4 KV heads, head_dim 128
    "gpt-oss-20b": 72 * 1024,        # 24 layers, wider KV heads
    "qwen3-14b": 160 * 1024,         # 40 layers, 8 KV heads, head_dim 128
    "qwen3-8b": 144 * 1024,          # 36 layers, 8 KV heads, head_dim 128
    "qwen3-4b": 144 * 1024,          # 36 layers, 8 KV heads, head_dim 128
    "default": 128 * 1024,
}


def kv_cache_bytes(model_key: str, context: int) -> int:
    return KV_BYTES_PER_TOKEN.get(model_key, KV_BYTES_PER_TOKEN["default"]) * context


def plan(model_size_gb: float, context: int, model_key: str = "default",
         backend_overhead_gb: float = 1.2) -> dict:
    """Decide whether a GGUF of ``model_size_gb`` fits."""
    b = current_budget()
    weights = model_size_gb * GIB
    kv = kv_cache_bytes(model_key, context)
    overhead = backend_overhead_gb * GIB
    total = weights + kv + overhead
    fits = total <= b.effective_bytes
    return {
        "budget_gb": round(b.effective_bytes / GIB, 2),
        "weights_gb": round(weights / GIB, 2),
        "kv_cache_gb": round(kv / GIB, 2),
        "overhead_gb": round(overhead / GIB, 2),
        "total_gb": round(total / GIB, 2),
        "headroom_gb": round((b.effective_bytes - total) / GIB, 2),
        "fits": fits,
        "max_context_that_fits": int(
            max(0, b.effective_bytes - weights - overhead)
            / KV_BYTES_PER_TOKEN.get(model_key, KV_BYTES_PER_TOKEN["default"])
        ),
    }


CANDIDATES = [
    ("gpt-oss-20b (native MXFP4, 3 shards)", 12.81, "gpt-oss-20b"),
    ("Qwen3-4B-Instruct-2507 Q4_K_M", 2.33, "qwen3-4b"),
    ("Qwen3-8B Q4_K_M", 4.68, "qwen3-8b"),
    ("Qwen3-14B Q3_K_M", 6.82, "qwen3-14b"),
    ("Qwen3-14B Q4_K_M", 8.38, "qwen3-14b"),
    ("Qwen3-30B-A3B-Instruct-2507 Q4_K_M", 17.28, "qwen3-30b-a3b"),
    ("Qwen3-Next-80B-A3B Q4_K_M", 45.17, "qwen3-30b-a3b"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plan a local model deployment")
    ap.add_argument("--model-size", type=float, help="GGUF size in GB")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--model-key", default="default")
    ap.add_argument("--list", action="store_true", help="evaluate known candidates")
    args = ap.parse_args()

    b = current_budget()
    print(b.describe())
    print()

    if args.list:
        print(f"{'candidate':40s} {'weights':>8s} {'total':>8s} {'verdict':>9s}")
        print("-" * 70)
        for name, size, key in CANDIDATES:
            r = plan(size, args.context, key)
            verdict = "FITS" if r["fits"] else "TOO BIG"
            print(f"{name:40s} {size:7.2f}G {r['total_gb']:7.2f}G {verdict:>9s}")
        print("\n(assumes context=%d; KV cache and backend overhead included)"
              % args.context)
        return 0

    if args.model_size is None:
        raise SystemExit("pass --model-size, or --list to see candidates")

    r = plan(args.model_size, args.context, args.model_key)
    for k, v in r.items():
        print(f"  {k:24s} {v}")
    if not r["fits"]:
        print(f"\n  VERDICT: does not fit. Largest workable context for this "
              f"model: {r['max_context_that_fits']} tokens.")
        return 1
    print("\n  VERDICT: fits. Headroom "
          f"{r['headroom_gb']} GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
