"""A small, general tool-using agent for the Strata model (OpenAI-compatible server).

New for this benchmark (the robot agent in stage_ov is not used or changed). It speaks
MCP over stdio to the same servers the Claude arms get, exposes their tools as OpenAI
function tools, and records per-call tokens/latency and per-tool calls/seconds/errors.
The model's context is 32k tokens, so old tool results are shortened as the history grows.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.environ.get("STRATA_BASE_URL", "http://10.160.30.12:20000/v1")
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-iq3_s")
RESULT_CHARS = 2500  # a tool result as the model first sees it
OLD_RESULT_CHARS = 300  # tool results older than the last KEEP_FULL turns
KEEP_FULL = 4
TOOL_TIMEOUT_S = 180.0  # a single MCP tool call (Blender can hang on a bad request)
SERVER_WAIT_S = 3 * 3600  # how long to wait out a Strata server outage before giving up
MAX_TURNS = 400  # effectively unbounded: the 30 min wall-clock cap applies to every arm
MAX_MALFORMED = 3  # tool calls in a row the server could not parse before the session ends
MAX_TOKENS = 4096  # one reply, thinking included
MAX_CONTINUE = 3  # replies in a row cut off while thinking (empty, no call) that get a "continue" nudge
CONTINUE_NUDGE = ("Your reply ran out of room while thinking, before any answer or tool call. Continue: keep the "
                  "thinking short and call the next tool now, or reply DONE with the summary if the scene is complete.")


IMAGE_MAX_SIDE = 1280  # a reference photo's long side as the (vision) model gets it


def image_part(path: str, max_side: int = IMAGE_MAX_SIDE) -> dict[str, Any]:
    """An OpenAI ``image_url`` content part: the image as a base64 JPEG data URL, scaled down so
    its long side is at most ``max_side`` px (never up)."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max_side / max(im.size)
        if scale < 1:
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def user_content(prompt: str, image: str | None) -> str | list[dict[str, Any]]:
    """The first user message's content: the prompt, after the image when there is one."""
    return prompt if image is None else [image_part(image), {"type": "text", "text": prompt}]


class ServerUnavailable(RuntimeError):
    """The model server stayed down: the scene has no result and must be run again."""


def _inherited_env() -> dict[str, str]:
    """This process's environment without ai-db settings: an inherited AI_DB_DAEMON_URL would
    send the run's ai-db calls to the user's daemon (its code, scene memory and Blender port)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("AI_DB_")}


class MCPClient:
    """Minimal MCP stdio client: initialize, tools/list, tools/call (line-delimited JSON-RPC)."""

    def __init__(self, command: str, args: list[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen([command, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                     env={**_inherited_env(), **env})
        self._id = 0
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "strata-bench", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _rpc(self, method: str, params: dict[str, Any], timeout: float = TOOL_TIMEOUT_S) -> Any:
        """One request; never waits longer than ``timeout`` (a late reply is ignored by id)."""
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{method} gave no reply within {timeout:.0f} s")
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f"MCP server closed during {method}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"].get("message", str(msg["error"])))
                return msg.get("result")

    def tools(self) -> list[dict[str, Any]]:
        return self._rpc("tools/list", {}).get("tools", [])

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        res = self._rpc("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        return text, bool(res.get("isError"))

    def close(self) -> None:
        self.proc.terminate()


class RequestRejected(RuntimeError):
    """The server refused the request itself (HTTP 4xx, e.g. the context is full): not an outage."""


def _chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]], base_url: str = BASE_URL,
          model: str = MODEL) -> tuple[dict[str, Any], dict[str, Any]]:
    """The reply message and the usage, with the choice's ``finish_reason`` added to the usage."""
    body = json.dumps({"model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
                       "temperature": 0.2, "max_tokens": MAX_TOKENS}).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500 and e.code != 429:
            raise RequestRejected(f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}") from e
        raise
    choice = data["choices"][0]
    return choice["message"], {**(data.get("usage") or {}), "finish_reason": choice.get("finish_reason")}


def _call_error(content: str) -> str:
    """Why the text of a <tool_call> block is not a usable call."""
    body = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    try:
        call = json.loads(body)
    except json.JSONDecodeError as e:
        return f"invalid JSON ({e})."
    if not isinstance(call, dict) or "name" not in call:
        return 'it needs {"name": ..., "arguments": {...}}.'
    return "the call was not recognised."


def _salvage_call(content: str, n: int) -> dict[str, Any] | None:
    """A <tool_call> block whose JSON is complete but followed by surplus closing braces (a common
    slip of small models on deeply nested arguments) as an OpenAI tool call; None otherwise."""
    body = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    try:
        call, end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError:
        return None
    if body[end:].strip(" \n\t}") or not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    args = call.get("arguments", {})
    return {"id": f"salvaged-{n}", "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(args)}}


def _trim(messages: list[dict[str, Any]]) -> None:
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[:-KEEP_FULL]:
        c = messages[i]["content"]
        if len(c) > OLD_RESULT_CHARS:
            messages[i]["content"] = c[:OLD_RESULT_CHARS] + " …[trimmed]"


def run(prompt: str, servers: dict[str, dict[str, Any]], transcript_path: str, timeout_s: float,
        base_url: str | None = None, model: str | None = None, image: str | None = None) -> dict[str, Any]:
    """One session against ``base_url``/``model`` (default: the Strata server). ``image``: a
    picture (path) sent with the prompt in the first user message (a vision model's reference)."""
    base_url, model = base_url or BASE_URL, model or MODEL
    clients = {name: MCPClient(s["command"], s["args"], s.get("env", {})) for name, s in servers.items()}
    route: dict[str, tuple[str, str]] = {}
    tools: list[dict[str, Any]] = []
    for sname, cl in clients.items():
        for t in cl.tools():
            fname = f"{sname}__{t['name']}"[:64]
            route[fname] = (sname, t["name"])
            tools.append({"type": "function", "function": {
                "name": fname, "description": (t.get("description") or "")[:1024],
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a Blender scene-building agent. Use the tools; call them, do not describe them."},
        {"role": "user", "content": user_content(prompt, image)}]
    per_call: list[dict[str, Any]] = []
    stats: dict[str, dict[str, float]] = {}
    t0 = time.monotonic()
    llm_s = tool_s = outage_s = 0.0
    outages = 0
    final, ok, timed_out, rejected = "", False, False, None
    malformed = bad_streak = salvaged = 0
    cut_streak = continued = 0
    with open(transcript_path, "w", encoding="utf-8") as tr:
        for _turn in range(MAX_TURNS):
            if time.monotonic() - t0 > timeout_s:
                timed_out = True
                break
            _trim(messages)
            msg = usage = None
            delay = 10.0
            while msg is None:
                a = time.monotonic()
                try:
                    msg, usage = _chat(messages, tools, base_url, model)
                except RequestRejected as e:  # e.g. the context is full: the session ends here
                    rejected = str(e)[:500]
                    tr.write(json.dumps({"t": round(a - t0, 2), "rejected": rejected}) + "\n")
                    break
                except (OSError, http.client.HTTPException) as e:
                    # Server outage (refused, reset, 5xx): wait it out, and keep the waiting out
                    # of every timing, so an outage pauses the scene instead of failing it.
                    if outage_s > SERVER_WAIT_S:
                        # Not a result of the model: the harness must retry this scene later.
                        for cl in clients.values():
                            cl.close()
                        raise ServerUnavailable(f"LLM server unavailable for {outage_s:.0f} s: {e}") from e
                    tr.write(json.dumps({"t": round(a - t0, 2), "outage": str(e)[:200], "wait_s": delay}) + "\n")
                    time.sleep(delay)
                    outage_s += time.monotonic() - a
                    t0 += time.monotonic() - a
                    outages += 1
                    delay = min(delay * 2, 120.0)
            if msg is None or usage is None:
                break
            dt = time.monotonic() - a
            llm_s += dt
            per_call.append({"t": round(a - t0, 2), "seconds": round(dt, 2),
                             "input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)})
            tr.write(json.dumps({"t": round(a - t0, 2), "assistant": msg, "usage": usage}) + "\n")
            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            if not calls and "<tool_call>" in content:
                fixed = _salvage_call(content, len(per_call))
                if fixed:  # only surplus closing braces were wrong: run it as the call it is
                    calls, content = [fixed], content.split("<tool_call>", 1)[0]
                    salvaged += 1
            messages.append({"role": "assistant", "content": content,
                             **({"tool_calls": calls} if calls else {})})
            if not calls and "<tool_call>" in content and bad_streak < MAX_MALFORMED:
                # a tool call the server's parser could not read (e.g. unbalanced JSON): say so,
                # as a harness does for a bad call, instead of taking it as the final answer
                bad_streak += 1
                malformed += 1
                messages.append({"role": "user", "content": "Your last tool call could not be parsed: "
                                 + _call_error(content) + " Call the tool again with valid JSON."})
                continue
            cut_off = usage.get("finish_reason") == "length" or usage.get("completion_tokens", 0) >= MAX_TOKENS
            if not calls and not content.strip() and cut_off and cut_streak < MAX_CONTINUE:
                # the thinking used the whole reply budget: not an answer, so nudge instead of ending
                cut_streak += 1
                continued += 1
                messages.append({"role": "user", "content": CONTINUE_NUDGE})
                tr.write(json.dumps({"t": round(time.monotonic() - t0, 2), "continue_nudge": cut_streak}) + "\n")
                continue
            cut_streak = 0
            if not calls:
                final = content
                ok = "DONE" in final.upper()
                break
            bad_streak = 0
            for call in calls:
                if time.monotonic() - t0 > timeout_s:  # the cap holds inside a turn too
                    timed_out = True
                    messages.append({"role": "tool", "tool_call_id": call.get("id", "?"),
                                     "content": "not run: the time limit was reached"})
                    continue
                fname = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    bad_args = None if isinstance(args, dict) else "arguments must be a JSON object"
                    args = args if isinstance(args, dict) else {}
                except json.JSONDecodeError as e:
                    args, bad_args = {}, f"arguments are not valid JSON ({e})"
                sname, tname = route.get(fname, ("?", fname))
                label = f"{sname}:{tname}"
                inner = args.get("tool") or args.get("name")
                if tname == "invoke_tool" and isinstance(inner, str):
                    label = f"blender:{inner}"
                b = time.monotonic()
                if sname == "?":
                    text, err = f"unknown tool {fname}", True
                elif bad_args:
                    text, err = f"not run: {bad_args}", True
                else:
                    try:
                        text, err = clients[sname].call(tname, args)
                    except Exception as e:  # noqa: BLE001 - tool failures go back to the model
                        text, err = f"tool error: {e}", True
                dt = time.monotonic() - b
                tool_s += dt
                st = stats.setdefault(label, {"calls": 0, "seconds": 0.0, "errors": 0})
                st["calls"] += 1
                st["seconds"] += dt
                st["errors"] += int(err)
                tr.write(json.dumps({"t": round(b - t0, 2), "tool": label, "seconds": round(dt, 2),
                                     "error": err, "result": text[:2000]}) + "\n")
                messages.append({"role": "tool", "tool_call_id": call.get("id", fname),
                                 "content": text[:RESULT_CHARS]})
    for cl in clients.values():
        cl.close()
    wall = time.monotonic() - t0
    return {
        "ok": ok, "timed_out": timed_out, "wall_s": round(wall, 1), "tool_s": round(tool_s, 1),
        "model_s": round(llm_s, 1), "api_s": round(llm_s, 1), "turns": len(per_call), "cost_usd": 0.0,
        "tokens": {"input": sum(c["input"] for c in per_call), "output": sum(c["output"] for c in per_call),
                   "cache_write": 0, "cache_read": 0},
        "per_call": per_call, "server_outages": outages, "rejected": rejected,
        "malformed_calls": malformed, "salvaged_calls": salvaged, "continue_nudges": continued, "outage_s_excluded": round(outage_s, 1),
        "tools": {k: {"calls": int(v["calls"]), "seconds": round(v["seconds"], 1), "errors": int(v["errors"])}
                  for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["seconds"])},
        "final_text": final[-1500:],
    }
