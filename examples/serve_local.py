"""Serve a local llama.cpp model over the OpenAI wire format.

Why this is the point of the whole local-model exercise
------------------------------------------------------
Evolver's benchmark has always been able to produce numbers. Until a real
model sits behind the adapter those numbers only validate the *machinery* --
the deterministic replay agent does not reason, so "success rate 30% -> 100%"
proves the evolution loop converges, not that it helps a model.

Pointing ``--provider kimi --base-url http://127.0.0.1:8080/v1`` at a locally
served Qwen3-Next-80B turns the same benchmark into a real measurement. The
model is real, the reasoning is real, and any improvement is the model's.

Usage::

    python3.11 examples/serve_local.py --model /workspace/models/.../x.gguf
    # then, in another shell:
    evolver bench --provider kimi --base-url http://127.0.0.1:8080/v1 \\
                  --api-key-env MOONSHOT_API_KEY --epochs 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_BIN = "/tmp/llama.cpp/build/bin/llama-server"


def find_server(explicit: str | None) -> str:
    if explicit:
        return explicit
    if Path(DEFAULT_BIN).exists():
        return DEFAULT_BIN
    # Fall back to whatever is on PATH.
    from shutil import which

    found = which("llama-server")
    if not found:
        raise SystemExit(
            "llama-server not found. Build llama.cpp first, or pass --binary."
        )
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve a local GGUF model")
    ap.add_argument("--model", default=None, help="path to a .gguf file")
    ap.add_argument("--binary", default=None, help="path to llama-server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=8192, help="context size")
    ap.add_argument("--threads", type=int, default=0, help="0 = auto")
    ap.add_argument("--ngl", type=int, default=0, help="GPU layers (0 = CPU only)")
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    args = ap.parse_args()

    model = args.model or os.getenv("EVOLVER_MODEL_PATH")
    if not model:
        raise SystemExit("pass --model <path.gguf> or set $EVOLVER_MODEL_PATH")
    if not Path(model).exists():
        raise SystemExit(f"model not found: {model}")

    threads = args.threads or os.cpu_count() or 8
    cmd = [
        find_server(args.binary),
        "-m", model,
        "--host", args.host,
        "--port", str(args.port),
        "-c", str(args.ctx),
        "-t", str(threads),
        "-ngl", str(args.ngl),
        "--jinja",              # honour the model's own chat template
    ]

    size_gb = Path(model).stat().st_size / 1073741824
    print(f"model   : {Path(model).name}  ({size_gb:.1f} GB)")
    print(f"threads : {threads}")
    print(f"endpoint: http://{args.host}:{args.port}/v1")
    print(f"command : {' '.join(cmd)}\n")

    if args.dry_run:
        return 0

    # llama-server exposes an OpenAI-compatible API at /v1, so Evolver's
    # `kimi`/`openai-compatible` providers work against it unchanged.
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
