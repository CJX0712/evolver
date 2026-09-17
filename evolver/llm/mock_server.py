"""A local OpenAI-compatible endpoint for wire-level verification.

Why this exists
---------------
Kimi K3 cannot be hosted on a normal machine: INT4 weights alone are ~1.3 TB
and serving wants ~1.7 TB of VRAM. But every hosted K3 provider -- Moonshot,
and the OpenRouter providers reselling it -- speaks the OpenAI wire format. So
the only thing standing between "wired up" and "actually works" is whether the
bytes Evolver puts on the wire are correct.

This module answers that with no API key. It stands up a real HTTP server
implementing ``POST /v1/chat/completions``, delegates generation to
:class:`ReplayAdapter`, and records every request it receives::

    evolver bench --provider kimi --base-url http://127.0.0.1:8791/v1 \\
                  --api-key-env MOCK_KEY

The full pipeline -- CodeAct execution, the 4-layer sandbox, skill injection,
distillation, evolution -- then runs over TCP exactly as it would against
Moonshot, and you can inspect the exact payload that would have been sent.

Two caveats, stated plainly:

* It is a **wire** check, not a **quality** check. The responses are synthetic,
  so a passing run proves the plumbing, not that K3 is any good.
* It uses the stdlib only, so it runs in CI on a machine with no GPU, no
  network, and no credentials.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

from evolver.core.config import LLMConfig
from evolver.llm.replay import ReplayAdapter


class _MockHTTPServer(HTTPServer):
    """Single-threaded HTTP server -- deliberately not the threading variant.

    A benchmark run issues ~90 requests. ``ThreadingHTTPServer`` spawns a fresh
    thread per request and -- with HTTP/1.1 keep-alive -- each thread then sits
    blocked reading the next request, so threads accumulate faster than they
    retire. Past a few thousand the process dies with::

        RuntimeError: can't start new thread
        MemoryError

    which surfaced as an infuriating "test just hangs" symptom. One request at
    a time is plenty for a mock, and it removes the failure mode entirely
    rather than papering over it with a thread cap. It also makes the
    ``server_close()`` deadlock (the base ``ThreadingMixIn`` joins every
    handler thread) structurally impossible.
    """

    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    """Answers one request per connection, then hangs up.

    ``protocol_version`` stays HTTP/1.1 so clients exercise the same framing
    they would against a real provider, but ``close_connection`` is forced
    true. Without that, a client that reuses its connection leaves the
    single-threaded server parked in ``handle_one_request() -> readinto()``
    waiting for a request that never comes -- and since this is used from a
    context manager, teardown then blocks forever on ``shutdown()``.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep pytest output clean
        pass

    # -- helpers ---------------------------------------------------------
    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {
                "object": "list",
                "data": [{"id": self.server.model, "object": "model",
                          "created": int(time.time()), "owned_by": "mock"}],
            })
            return
        self._send(404, {"error": {"message": f"no route: {self.path}",
                                   "type": "invalid_request_error"}})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        body = self._read_json()
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"no route: {self.path}",
                                       "type": "invalid_request_error"}})
            return

        record = {
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "content_type": self.headers.get("Content-Type"),
            "body": body,
        }
        self.server.requests.append(record)

        messages = body.get("messages") or []
        system = next((m.get("content", "") for m in messages
                       if m.get("role") == "system"), "")
        user = next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"), "")
        model = body.get("model") or self.server.model

        try:
            resp = self.server.adapter.complete(system, user)
            text = resp.text
            tokens_out = resp.tokens_out
            error: Optional[str] = resp.error
        except Exception as exc:  # noqa: BLE001 - a broken mock must not hang CI
            text, tokens_out, error = "", 0, f"{type(exc).__name__}: {exc}"

        if error:
            self._send(500, {"error": {"message": error, "type": "server_error"}})
            return

        tokens_in = max(1, len(system + user) // 4)
        self._send(200, {
            "id": f"chatcmpl-mock-{len(self.server.requests)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": tokens_in,
                      "completion_tokens": tokens_out,
                      "total_tokens": tokens_in + tokens_out},
        })


class MockOpenAIServer:
    """A throwaway OpenAI-compatible server, safe for tests.

    Binds port 0 by default so the OS picks a free port -- no fixed-port
    collisions between parallel test workers.
    """

    def __init__(
        self,
        adapter: Optional[ReplayAdapter] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        model: str = "kimi-k3",
        tasks: Optional[List[Any]] = None,
    ) -> None:
        self.model = model
        self.requests: List[Dict[str, Any]] = []
        self.adapter = adapter or ReplayAdapter(
            config=LLMConfig(provider="replay", model=model), tasks=tasks
        )
        self._httpd = _MockHTTPServer((host, port), _Handler)
        # The handler reaches its server through self.server; stash extra state.
        self._httpd.model = model
        self._httpd.adapter = self.adapter
        self._httpd.requests = self.requests
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -------------------------------------------------------
    @property
    def host(self) -> str:
        return self._httpd.server_address[0]

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> "MockOpenAIServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "MockOpenAIServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- inspection ------------------------------------------------------
    def clear(self) -> None:
        self.requests.clear()

    @property
    def models_called(self) -> List[str]:
        return [r["body"].get("model") for r in self.requests]

    @property
    def last_body(self) -> Dict[str, Any]:
        return self.requests[-1]["body"] if self.requests else {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MockOpenAIServer url={self.base_url!r} requests={len(self.requests)}>"


def smoke_check(port: int = 0) -> Dict[str, Any]:
    """Ping the mock once and report what a provider would have received."""
    # LLMConfig.api_key is a read-only property over os.environ, so a key can
    # only ever arrive through an env var -- never as a literal in a config
    # file that might get committed. The mock honours that instead of
    # special-casing a client.
    os.environ.setdefault("MOCK_API_KEY", "sk-mock-local")
    with MockOpenAIServer(port=port) as srv:
        from evolver.llm.vendors import OpenAIAdapter

        adapter = OpenAIAdapter(
            config=LLMConfig(provider="kimi", model="kimi-k3",
                             base_url=srv.base_url, api_key_env="MOCK_API_KEY"),
        )
        resp = adapter.complete("You are a test.", "Say hello.")
        return {
            "ok": resp.ok,
            "error": resp.error,
            "text": resp.text,
            "tokens_in": resp.tokens_in,
            "tokens_out": resp.tokens_out,
            "requests": srv.requests,
        }


__all__ = ["MockOpenAIServer", "smoke_check"]
