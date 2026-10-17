"""The OpenCode runner waits for its MCP servers before it sends the brief (aidb7, 2026-10-17).

Every nemotron-3.5-lightning-free-aidb7 scene began with "No tool named aidb_build_scene is
currently available": ``opencode run --standalone`` sent the first model request before the MCP
servers had registered. The runner now starts a private ``opencode serve`` and polls its
``/api/mcp`` until every server is connected. A fake ``opencode`` stands in for the real one.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "agent_bench"))
import run_opencode

_FAKE = textwrap.dedent('''\
    #!{python}
    import base64, json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    want = "Basic " + base64.b64encode(("opencode:" + os.environ["OPENCODE_SERVER_PASSWORD"]).encode()).decode()
    polls = {{"n": 0}}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def _send(self, code, body):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
        def do_GET(self):
            if self.headers.get("Authorization") != want:
                return self._send(401, {{}})
            polls["n"] += 1
            with open(os.path.join(os.getcwd(), "..", "polls.txt"), "w") as f:
                f.write(str(polls["n"]))
            n = polls["n"]
            data = [] if n < 2 else [{{"name": "aidb", "status": {{"status": "failed" if n < 4 else "connected"}}}},
                                     {{"name": "blender", "status": {{"status": "connected"}}}}]
            self._send(200, {{"data": data}})
        def do_POST(self):
            self._send(200, {{}})
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


@pytest.fixture
def fake_opencode(tmp_path, monkeypatch):
    exe = tmp_path / "opencode"
    exe.write_text(_FAKE.format(python=sys.executable))
    exe.chmod(0o755)
    monkeypatch.setattr(run_opencode, "OPENCODE", str(exe))
    return tmp_path


def test_serve_returns_once_every_mcp_server_is_connected(fake_opencode):
    work = fake_opencode / "work"
    work.mkdir()
    home = fake_opencode / "home"
    home.mkdir()
    servers = {"aidb": {}, "blender": {}}
    proc, url, password, ready = run_opencode._serve(str(home), str(work), str(fake_opencode), servers)
    try:
        assert url.startswith("http://127.0.0.1:") and len(password) >= 24
        assert ready["servers"] == {"aidb": "connected", "blender": "connected"}
        assert ready["reconnects"] >= 1  # the failed aidb was asked to connect again
        assert int((fake_opencode / "polls.txt").read_text()) >= 4
    finally:
        run_opencode._stop(proc)
    assert proc.poll() is not None


def test_serve_gives_up_and_stops_the_server(fake_opencode, monkeypatch):
    monkeypatch.setattr(run_opencode, "MCP_READY_TIMEOUT_S", 1.5)
    work = fake_opencode / "work"
    work.mkdir()
    with pytest.raises(RuntimeError, match="not connected"):
        run_opencode._serve(str(fake_opencode), str(work), str(fake_opencode), {"aidb": {}, "blender": {}, "other": {}})


def test_lockdown_config_is_unchanged():
    cfg = run_opencode._config({"aidb": {"command": "x", "args": []}, "blender": {"command": "y", "args": []}})
    assert cfg["permission"]["*"] == "deny"
    assert cfg["permission"]["read"] == {"*": "deny", run_opencode._READ_DECOY: "allow"}
    assert cfg["permission"]["aidb_*"] == cfg["permission"]["blender_*"] == "allow"
    assert all(not s["codemode"] for s in cfg["mcp"]["servers"].values())
    assert os.path.basename(run_opencode.OPENCODE) == "opencode"
