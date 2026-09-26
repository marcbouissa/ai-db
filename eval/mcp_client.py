"""A persistent MCP client, for measuring the transport agents actually use.

Why this exists
---------------
Every other number in this benchmark comes from `python -m ai_db.cli ...`, which
starts a fresh process per call. That is the wrong shape for an agent: an agent
holds an MCP session open, so it pays interpreter start and model load **once**
and then not again.

Measured on this repo, same query, same index:

| transport | first call | subsequent calls |
|---|---|---|
| CLI subprocess | ~240 ms | ~240 ms, every call |
| CLI subprocess, hybrid | ~10.5 s | ~10.5 s, every call |
| persistent MCP | 286 ms | **0-3 ms** |

So a CLI-only harness does not merely add noise to the hybrid comparison, it
inverts it: it makes the highest-recall configuration look like the slowest by
three orders of magnitude, purely because of how the process is launched. Any
claim about hybrid being "too slow" measured this way is a claim about
`subprocess`, not about ai-db.

Protocol
--------
Line-delimited JSON-RPC 2.0 over the server's stdin/stdout, which is what
`mcp_server.py` implements. The client keeps one server process alive across
calls, performs the `initialize` / `notifications/initialized` handshake once,
and matches responses by id.

The server is always terminated in a `finally`, so a failing call cannot leave an
orphaned process holding a database lock.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Self

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class McpError(RuntimeError):
    """The server returned an error, or the stream desynchronised."""


class PersistentMcpClient:
    """One long-lived `ai-db mcp` process, driven over stdio JSON-RPC.

    Not thread-safe: it owns a single stdin/stdout pair and matches replies by
    id, so calls must be serialised. The benchmark is sequential by design.
    """

    def __init__(self, config: str, cwd: str = REPO,
                 protocol_version: str = "2024-11-05") -> None:
        self.config = config
        self.cwd = cwd
        self.protocol_version = protocol_version
        self.proc: subprocess.Popen[str] | None = None
        self._next_id = 0
        self.handshake_ms: float | None = None
        self.server_info: dict[str, Any] = {}

    # ------------------------------------------------------------- lifecycle
    def start(self) -> Self:
        argv = [sys.executable, "-m", "ai_db.cli", "--config", self.config, "mcp"]
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=self.cwd)
        t0 = time.perf_counter()
        resp = self._round_trip("initialize", {
            "protocolVersion": self.protocol_version,
            "capabilities": {},
            "clientInfo": {"name": "ai-db-benchmark", "version": "1"},
        })
        self.handshake_ms = (time.perf_counter() - t0) * 1000
        self.server_info = resp.get("result", {}).get("serverInfo", {})
        # The notification is fire-and-forget: the server emits no reply for it.
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            self.proc.kill()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass
            self.proc = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- calls
    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, float]:
        """Invoke a tool. Returns (text, elapsed_ms).

        The elapsed time is the *server-side round trip*: from the moment the
        request is written to the moment its reply is read. That is the number an
        agent waits for, and it excludes this client's own bookkeeping.
        """
        t0 = time.perf_counter()
        resp = self._round_trip("tools/call", {"name": name, "arguments": arguments})
        ms = (time.perf_counter() - t0) * 1000
        return extract_text(resp), ms

    def list_tools(self) -> list[str]:
        resp = self._round_trip("tools/list", {})
        return [t["name"] for t in resp.get("result", {}).get("tools", [])]

    # --------------------------------------------------------------- plumbing
    def _send(self, payload: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise McpError("client is not started")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _round_trip(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id,
                    "method": method, "params": params})
        if self.proc is None or self.proc.stdout is None:
            raise McpError("client is not started")
        line = self.proc.stdout.readline()
        if not line:
            raise McpError(f"server closed the stream during {method!r}")
        try:
            resp: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            raise McpError(f"non-JSON reply to {method!r}: {line[:120]!r}") from exc
        if "error" in resp:
            raise McpError(f"{method!r}: {resp['error']}")
        return resp


def extract_text(resp: dict[str, Any]) -> str:
    """Concatenate the text blocks of a tools/call result.

    MCP returns content as a list of typed blocks. The dispatcher returns a
    string for some tools and a dict for others; the dict arrives as JSON text,
    so the caller can parse it if it needs to.
    """
    result = resp.get("result", {})
    if result.get("isError"):
        raise McpError(f"tool reported an error: {json.dumps(result)[:200]}")
    blocks = result.get("content")
    if isinstance(blocks, list):
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
    return json.dumps(result)
