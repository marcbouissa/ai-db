"""Run one Claude Code headless session (``claude -p``) on a brief and measure it.

The session sees exactly the arm's MCP servers (``--strict-mcp-config``), no built-in
tools, and none of the user's settings, plugins or daemon. Every stream-json line is
timestamped as it arrives, which gives model time vs tool time and a per-tool breakdown.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any

CLAUDE = os.environ.get(
    "BENCH_CLAUDE", os.path.expanduser("~/.config/Claude/claude-code/2.1.293/25786da347c3/claude"))
MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BLENDER_MCP = os.path.expanduser("~/GitRepos/blender_mcp")
_BUILTIN = ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch", "Task", "Agent",
            "NotebookEdit", "TodoWrite", "Skill"]


def blender_mcp_module() -> str:
    """The addon checkout's MCP server package (the one ``*_mcp`` package with a ``__main__``),
    overridable with BENCH_BLENDER_MCP_MODULE: the name is not hard-coded in this repo."""
    if os.environ.get("BENCH_BLENDER_MCP_MODULE"):
        return os.environ["BENCH_BLENDER_MCP_MODULE"]
    found = sorted(d for d in os.listdir(BLENDER_MCP) if d.endswith("_mcp")
                   and os.path.isfile(os.path.join(BLENDER_MCP, d, "__main__.py")))
    if len(found) != 1:
        raise RuntimeError(f"cannot pick the MCP package in {BLENDER_MCP}: {found}; set BENCH_BLENDER_MCP_MODULE")
    return found[0]


def mcp_servers(port: int, run_dir: str, with_aidb: bool, aidb_tools: str) -> dict[str, Any]:
    servers: dict[str, Any] = {"blender": {
        "command": os.path.join(BLENDER_MCP, ".venv", "bin", "python"),
        "args": ["-m", blender_mcp_module(), "--transport", "stdio", "--mode", "omni", "--blender-port", str(port)],
        "env": {"PYTHONPATH": BLENDER_MCP},
    }}
    if with_aidb:
        servers["aidb"] = {
            "command": os.path.join(REPO, ".venv", "bin", "ai-db"),
            "args": ["mcp"],
            "env": aidb_env(port, run_dir, aidb_tools),
        }
    return servers


def aidb_env(port: int, run_dir: str, aidb_tools: str) -> dict[str, str]:
    """An ai-db with its own index, config, scene memory and Blender port: never the user's."""
    return {"AI_DB_PATH": os.path.join(run_dir, "aidb", "code.db"),
            "AI_DB_CONFIG": os.path.join(run_dir, "aidb", "config.json"),
            "AI_DB_SCENE_MEMORY": os.path.join(run_dir, "aidb", "scene_memory.db"),
            "AI_DB_BLENDER_PORT": str(port), "AI_DB_MCP_TOOLS": aidb_tools,
            "AI_DB_SKILL_DIRS": "", "HF_HUB_OFFLINE": "1"}


def run(model: str, prompt: str, port: int, run_dir: str, with_aidb: bool, aidb_tools: str,
        timeout_s: float) -> dict[str, Any]:
    cfg_path = os.path.join(run_dir, "mcp.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"mcpServers": mcp_servers(port, run_dir, with_aidb, aidb_tools)}, f, indent=1)
    allowed = ["mcp__blender"] + (["mcp__aidb"] if with_aidb else [])
    cmd = [CLAUDE, "-p", prompt, "--model", MODELS[model], "--output-format", "stream-json", "--verbose",
           "--mcp-config", cfg_path, "--strict-mcp-config", "--setting-sources", "project",
           "--no-session-persistence", "--allowedTools", *allowed, "--disallowedTools", *_BUILTIN]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AI_DB_", "CLAUDE_CODE_PLUGIN")) and k != "ANTHROPIC_API_KEY"}
    work = os.path.join(run_dir, "cwd")
    os.makedirs(work, exist_ok=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    lines: list[tuple[float, dict[str, Any]]] = []
    killed = threading.Timer(timeout_s, proc.kill)
    killed.start()
    with open(os.path.join(run_dir, "transcript.jsonl"), "w", encoding="utf-8") as tr:
        for raw in proc.stdout:  # type: ignore[union-attr]
            t = time.monotonic() - t0
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lines.append((t, msg))
            tr.write(json.dumps({"t": round(t, 3), **msg}) + "\n")
    proc.wait()
    killed.cancel()
    stderr = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
    return summarize(lines, time.monotonic() - t0, proc.returncode, stderr, timed_out=not killed.is_alive()
                     and proc.returncode not in (0, 1))


def summarize(lines: list[tuple[float, dict[str, Any]]], wall: float, rc: int, stderr: str,
              timed_out: bool) -> dict[str, Any]:
    """Tokens (total, per call), model vs tool time, per-tool calls/seconds/errors."""
    calls: dict[str, dict[str, Any]] = {}
    tools: dict[str, dict[str, float]] = {}
    turns: list[dict[str, Any]] = []
    seen_msg: set[str] = set()
    result: dict[str, Any] = {}
    for t, msg in lines:
        kind = msg.get("type")
        if kind == "assistant":
            m = msg.get("message", {})
            mid = m.get("id")
            if mid and mid not in seen_msg and m.get("usage"):
                seen_msg.add(mid)
                u = m["usage"]
                turns.append({"t": round(t, 2), "input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
                              "cache_write": u.get("cache_creation_input_tokens", 0),
                              "cache_read": u.get("cache_read_input_tokens", 0)})
            for block in m.get("content", []):
                if block.get("type") == "tool_use":
                    name = block.get("name", "?").removeprefix("mcp__")
                    inner = (block.get("input") or {}).get("tool") or (block.get("input") or {}).get("name")
                    if name.endswith("invoke_tool") and inner:
                        name = f"blender:{inner}"
                    calls[block["id"]] = {"name": name, "t": t}
        elif kind == "user":
            for block in msg.get("message", {}).get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    c = calls.pop(block.get("tool_use_id"), None)
                    if c is None:
                        continue
                    st = tools.setdefault(c["name"], {"calls": 0, "seconds": 0.0, "errors": 0})
                    st["calls"] += 1
                    st["seconds"] += t - c["t"]
                    st["errors"] += 1 if block.get("is_error") else 0
        elif kind == "result":
            result = msg
    tool_s = sum(v["seconds"] for v in tools.values())
    usage = result.get("usage") or {}
    return {
        "ok": rc == 0 and not result.get("is_error", False), "returncode": rc, "timed_out": timed_out,
        "stderr_tail": stderr[-800:],
        "wall_s": round(wall, 1), "tool_s": round(tool_s, 1), "model_s": round(max(wall - tool_s, 0.0), 1),
        "api_s": round((result.get("duration_api_ms") or 0) / 1000, 1),
        "turns": result.get("num_turns"), "cost_usd": result.get("total_cost_usd"),
        "tokens": {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0),
                   "cache_write": usage.get("cache_creation_input_tokens", 0),
                   "cache_read": usage.get("cache_read_input_tokens", 0)},
        "model_usage": result.get("modelUsage"),
        "per_call": turns,
        "tools": {k: {"calls": int(v["calls"]), "seconds": round(v["seconds"], 1), "errors": int(v["errors"])}
                  for k, v in sorted(tools.items(), key=lambda kv: -kv[1]["seconds"])},
        "final_text": (result.get("result") or "")[-1500:],
    }
