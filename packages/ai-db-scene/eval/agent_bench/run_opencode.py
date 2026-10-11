"""Run one OpenCode headless session (``opencode run``) with a free OpenCode model.

Every run gets its own empty HOME (no global config, MCP servers, AGENTS.md or skills leak in;
the free models need no login), a project config with exactly the arm's MCP servers, and
OpenCode's built-in tools switched off. Usage comes from the run's own session store:
tokens per model call (input, output, reasoning, cache read/write) and per-call timings.
OpenCode 2 calls MCP tools through "Code Mode" (one ``execute`` tool running JS that calls
``tools["<server>"].<tool>(...)``), so the inner tool names are read from that code.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
from typing import Any

import run_claude

OPENCODE = os.path.expanduser("~/.opencode/bin/opencode")
SANDBOX = os.path.expanduser("~/.cache/ai-db-bench/cwd")
# OpenCode 1 and 2 names of the built-in tools: agents get only their MCP servers
_OFF = ("bash", "edit", "write", "read", "glob", "grep", "list", "patch", "webfetch", "websearch",
        "task", "todowrite", "todoread", "skill")
# OpenCode's free tier refuses requests that do not offer a usable `shell` tool ("free tier can
# only be used from within OpenCode"; a permission deny hides it too). So shell stays: the task
# prompt forbids it, and every shell command is recorded and flagged in the report.
_SHELL_FLAGS = ("blender", "ai-db", "aidb", "bpy", "9877", "110", "/dev/tcp", "socket", "python")


class ModelUnavailable(RuntimeError):
    """The provider refused or failed the session: no result to record.
    ``started`` is True when the model had already produced output (a truncated run)."""

    def __init__(self, message: str, started: bool = False) -> None:
        super().__init__(message)
        self.started = started


def _provider_error(events_path: str) -> str | None:
    try:
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                if '"type":"error"' in line and '"provider.' in line:
                    return json.loads(line).get("error", {}).get("message", "provider error")
    except (OSError, json.JSONDecodeError):
        return "unreadable events"
    return None


def _env(home: str) -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "HOME": home, "XDG_CONFIG_HOME": f"{home}/.config",
            "XDG_DATA_HOME": f"{home}/.local/share", "XDG_CACHE_HOME": f"{home}/.cache",
            "XDG_STATE_HOME": f"{home}/.local/state", "LANG": "C.UTF-8"}


def _home(run_dir: str) -> str:
    home = os.path.join(run_dir, "oc_home")
    os.makedirs(os.path.join(home, ".cache", "opencode"), exist_ok=True)
    cache = os.path.expanduser("~/.cache/opencode/models.json")
    if os.path.exists(cache):  # the model catalogue: saves a fetch per run
        shutil.copy(cache, os.path.join(home, ".cache", "opencode", "models.json"))
    return home


def _config(servers: dict[str, Any]) -> dict[str, Any]:
    mcp = {name: {"type": "local", "enabled": True, "command": [s["command"], *s["args"]],
                  "environment": s.get("env", {})} for name, s in servers.items()}
    return {"$schema": "https://opencode.ai/config.json", "mcp": mcp,
            "tools": dict.fromkeys(_OFF, False),
            "permission": {"*": "allow"}}


def run(model: str, prompt: str, port: int, run_dir: str, with_aidb: bool, aidb_tools: str,
        timeout_s: float, files: list[str] | None = None, servers: dict[str, Any] | None = None) -> dict[str, Any]:
    """One session; a rate limit hit before the model did anything is waited out (2-16 min)."""
    delay = 120.0
    while True:
        try:
            return _run_once(model, prompt, port, run_dir, with_aidb, aidb_tools, timeout_s, files, servers)
        except ModelUnavailable as e:
            if "rate limit" not in str(e).lower() or e.started or delay > 960:
                raise
            time.sleep(delay)
            delay *= 2


def _run_once(model: str, prompt: str, port: int, run_dir: str, with_aidb: bool, aidb_tools: str,
              timeout_s: float, files: list[str] | None, servers: dict[str, Any] | None) -> dict[str, Any]:
    home = _home(run_dir)
    # OpenCode treats the enclosing git repository as the project: it snapshots it every step
    # and reads its instruction files. bench/ lives in the ai-db repo, so the session's working
    # folder must be outside any repository.
    work = os.path.join(SANDBOX, os.path.relpath(run_dir, "/").replace(os.sep, "__"))
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    if servers is None:
        servers = run_claude.mcp_servers(port, run_dir, with_aidb, aidb_tools)
    with open(os.path.join(work, "opencode.json"), "w", encoding="utf-8") as f:
        json.dump(_config(servers), f, indent=1)
    cmd = [OPENCODE, "run", "--standalone", "--auto", "-m", f"opencode/{model}", "--format", "json"]
    for path in files or []:
        cmd += ["-f", path]
    cmd.append(prompt)
    t0 = time.time()
    with open(os.path.join(run_dir, "opencode_events.jsonl"), "w", encoding="utf-8") as out, \
            open(os.path.join(run_dir, "opencode_stderr.txt"), "w", encoding="utf-8") as err:
        try:
            proc = subprocess.run(cmd, cwd=work, env=_env(home), stdin=subprocess.DEVNULL, stdout=out,
                                  stderr=err, timeout=timeout_s)
            rc, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
    wall = time.time() - t0
    res = summarize(os.path.join(home, ".local", "share", "opencode", "opencode.db"), wall, rc, timed_out,
                     os.path.join(run_dir, "transcript.jsonl"))
    err = _provider_error(os.path.join(run_dir, "opencode_events.jsonl"))
    if err and not res["timed_out"] and "DONE" not in (res["final_text"] or "").upper():
        # the provider ended the session (rate limit, outage), not the model: no result
        res["provider_error"] = err
        raise ModelUnavailable(f"opencode/{model}: {err}", started=bool(res["tokens"]["output"]))
    return res


def summarize(db_path: str, wall: float, rc: int, timed_out: bool, transcript_path: str) -> dict[str, Any]:
    per_call: list[dict[str, Any]] = []
    tools: dict[str, dict[str, float]] = {}
    final, outcome = "", None
    shell_cmds: list[dict[str, Any]] = []
    if os.path.exists(db_path):
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = c.execute("select type, data from session_message order by time_created, seq").fetchall()
        c.close()
    else:
        rows = []
    t_first = None
    with open(transcript_path, "w", encoding="utf-8") as tr:
        for typ, data in rows:
            d = json.loads(data)
            tr.write(json.dumps({"type": typ, **d}) + "\n")
            if typ == "idle":
                outcome = d.get("outcome")
            if typ != "assistant":
                continue
            tm, tok = d.get("time", {}), d.get("tokens") or {}
            t_first = t_first or tm.get("created")
            # model time: until the stream ended (completed also covers the tools it ran)
            start, end = tm.get("created") or 0, tm.get("streamed") or tm.get("completed") or tm.get("created") or 0
            cache = tok.get("cache") or {}
            per_call.append({"t": round(((start or 0) - (t_first or 0)) / 1000, 2), "seconds": round((end - start) / 1000, 2),
                             "input": tok.get("input", 0), "output": tok.get("output", 0),
                             "reasoning": tok.get("reasoning", 0), "cache_read": cache.get("read", 0),
                             "cache_write": cache.get("write", 0)})
            for item in d.get("content", []):
                if item.get("type") == "text":
                    final = item.get("text") or final
                if item.get("type") != "tool":
                    continue
                if item.get("name") in ("shell", "bash"):
                    command = str((item.get("state") or {}).get("input", {}).get("command", ""))
                    shell_cmds.append({"command": command[:500],
                                       "flagged": any(f in command.lower() for f in _SHELL_FLAGS)})
                st = item.get("state") or {}
                names = []
                for call in (st.get("metadata") or {}).get("toolCalls") or []:
                    name = str(call.get("tool", "?"))
                    inner = (call.get("input") or {}).get("tool")
                    names.append(f"{name.split('.')[0]}:{inner}" if name.endswith("invoke_tool") and inner
                                 else name.replace(".", ":"))
                tt = item.get("time") or {}
                secs = ((tt.get("completed") or 0) - (tt.get("ran") or tt.get("created") or 0)) / 1000
                label = "+".join(sorted(set(names))) or item.get("name", "?")
                s = tools.setdefault(label, {"calls": 0, "seconds": 0.0, "errors": 0})
                s["calls"] += 1
                s["seconds"] += max(secs, 0.0)
                s["errors"] += int(st.get("status") == "error")
    model_s = sum(p["seconds"] for p in per_call)
    tool_s = sum(v["seconds"] for v in tools.values())
    return {
        "ok": rc == 0 and not timed_out and "DONE" in final.upper(), "returncode": rc, "timed_out": timed_out,
        "outcome": outcome, "wall_s": round(wall, 1), "model_s": round(model_s, 1),
        "tool_s": round(tool_s if tool_s else max(wall - model_s, 0.0), 1), "api_s": round(model_s, 1),
        "turns": len(per_call), "cost_usd": 0.0,
        "tokens": {"input": sum(p["input"] for p in per_call), "output": sum(p["output"] for p in per_call),
                   "reasoning": sum(p["reasoning"] for p in per_call),
                   "cache_write": sum(p["cache_write"] for p in per_call),
                   "cache_read": sum(p["cache_read"] for p in per_call)},
        "per_call": per_call, "shell_commands": shell_cmds,
        "shell_flagged": sum(1 for c in shell_cmds if c["flagged"]),
        "tools": {k: {"calls": int(v["calls"]), "seconds": round(v["seconds"], 1), "errors": int(v["errors"])}
                  for k, v in sorted(tools.items(), key=lambda kv: -kv[1]["seconds"])},
        "final_text": final[-1500:],
    }
