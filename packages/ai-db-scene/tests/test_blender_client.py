"""Blender client (blender_ox socket protocol), scene tool table, and optional live test."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from ai_db_scene.blender_client import BlenderClient, BlenderError, BlenderUnavailable, default_port
from ai_db_scene.tools import scene_tools

from ai_db.dispatcher import ServiceDispatcher


class FakeBlender:
    """TCP server speaking the blender_ox protocol; `responder(payload) -> bytes|None`."""

    def __init__(self, responder):
        self.responder = responder
        self.received: list[Any] = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                payload = json.loads(buf)
                self.received.append(payload)
                out = self.responder(payload)
                if out is not None:
                    conn.sendall(out)

    def close(self):
        self.sock.close()


def _json_line(d: dict) -> bytes:
    return json.dumps(d).encode() + b"\n"


@pytest.fixture
def fake():
    servers = []

    def make(responder):
        f = FakeBlender(responder)
        servers.append(f)
        return f, BlenderClient(port=f.port, timeout=2)

    yield make
    for s in servers:
        s.close()


def test_single_request(fake):
    f, c = fake(lambda p: _json_line({"ok": 1, "status": "success", "result": {"name": "Box"}}))
    assert c.request("create_primitive", {"primitive": "cube"}) == {"name": "Box"}
    assert f.received == [{"type": "create_primitive", "params": {"primitive": "cube"}}]


def test_batch_partial(fake):
    results = [{"ok": 1, "status": "success", "result": 1}, {"ok": 0, "status": "error", "message": "x"}]
    f, c = fake(lambda p: _json_line({"ok": 0, "status": "partial", "results": results}))
    out = c.batch([{"type": "a"}, {"type": "b"}])
    assert [r["ok"] for r in out] == [1, 0]
    assert f.received[0] == {"batch": [{"type": "a"}, {"type": "b"}]}


def test_error_is_structured(fake):
    _, c = fake(
        lambda p: _json_line(
            {"ok": 0, "status": "error", "message": "Object not found: Bx", "error_type": "ObjectNotFound", "hint": "Box"}
        )
    )
    with pytest.raises(BlenderError) as ei:
        c.request("delete_object", {"name": "Bx"})
    assert not isinstance(ei.value, BlenderUnavailable)
    assert ei.value.error_type == "ObjectNotFound" and ei.value.hint == "Box"


def test_unavailable_and_is_available():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    c = BlenderClient(port=port, timeout=0.5)
    assert c.is_available() is False
    with pytest.raises(BlenderUnavailable):
        c.request("get_scene_info")


def test_empty_response_and_timeout(fake):
    _, c = fake(lambda p: b"")
    with pytest.raises(BlenderUnavailable):
        c.request("x")
    _, c2 = fake(lambda p: time.sleep(1.5))
    c2.timeout = 0.3
    with pytest.raises(BlenderUnavailable):
        c2.request("x")


def test_port_env_override(monkeypatch):
    assert default_port() == 9877
    monkeypatch.setenv("AI_DB_BLENDER_PORT", "12345")
    assert default_port() == 12345
    monkeypatch.setenv("AI_DB_BLENDER_PORT", "junk")
    assert default_port() == 9877


def test_every_scene_tool_has_handler_and_registers():
    table = scene_tools.scene_tool_table()
    names = [n for n, *_ in table]
    assert len(names) == len(set(names))
    assert all(callable(h) for *_, h in table)
    for spatial in (
        "get_bounds", "raycast", "check_collision", "measure_distance", "find_objects_in_region",
        "get_camera_view", "get_spatial_context", "get_object_transform", "validate_scene", "get_scene_info",
    ):
        assert spatial in names
    assert "plan_3d_scene" not in names and "plan_3d_asset" not in names
    d = ServiceDispatcher.__new__(ServiceDispatcher)
    d._tools = {}
    scene_tools.register_scene_tools(d)
    assert set(d._tools) == set(names)


def test_forward_goes_to_blender(fake, monkeypatch):
    f, _ = fake(lambda p: _json_line({"ok": 1, "status": "success", "result": {"min": [0, 0, 0]}}))
    monkeypatch.setenv("AI_DB_BLENDER_PORT", str(f.port))
    handler = {n: h for n, _, _, h in scene_tools.scene_tool_table()}["get_bounds"]
    assert handler({"name": "A"}) == {"status": "success", "result": {"min": [0, 0, 0]}}
    assert f.received[0] == {"type": "get_bounds", "params": {"name": "A"}}
    code = {n: h for n, _, _, h in scene_tools.scene_tool_table()}["execute_blender_code"]
    code({"code": "result=1", "user_prompt": "hi"})
    assert f.received[1] == {"type": "run_code", "params": {"code": "result=1"}}


def test_forward_error_structured(monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("AI_DB_BLENDER_PORT", str(port))
    handler = {n: h for n, _, _, h in scene_tools.scene_tool_table()}["raycast"]
    res = handler({"origin": [0, 0, 0], "direction": [0, 0, -1]})
    assert res["status"] == "error" and res["error_type"] == "BlenderUnavailable"


_BLENDER = Path.home() / "bin" / "blender"
# Checkout of the blender_ox addon; the live test is skipped when it is not set.
_ADDON = Path(os.environ.get("AI_DB_BLENDER_ADDON", "~/GitRepos/blender_mcp")).expanduser()

_LIVE_SCRIPT = """
import os, sys, time
sys.path.insert(0, os.environ["OX_ROOT"])
from blender_ox.socket_server import MCPServerState
MCPServerState.start(int(os.environ["OX_PORT"]))
stop = os.environ["OX_STOP"]
end = time.time() + 60
while time.time() < end and not os.path.exists(stop):
    MCPServerState.itc.process_pending()
    time.sleep(0.01)
MCPServerState.stop()
os._exit(0)
"""


@pytest.mark.blender
@pytest.mark.skipif(not (_BLENDER.exists() or shutil.which("blender")) or not _ADDON.exists(), reason="Blender/addon not available")
def test_live_blender(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    stop = tmp_path / "stop"
    script = tmp_path / "serve.py"
    script.write_text(_LIVE_SCRIPT)
    exe = str(_BLENDER) if _BLENDER.exists() else "blender"
    env = {**os.environ, "OX_ROOT": str(_ADDON), "OX_PORT": str(port), "OX_STOP": str(stop)}
    proc = subprocess.Popen([exe, "-b", "--factory-startup", "--python", str(script)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        c = BlenderClient(port=port, timeout=20)
        deadline = time.time() + 30
        while time.time() < deadline and not c.is_available(timeout=1):
            time.sleep(0.3)
        assert c.is_available(timeout=1)
        info = c.get_scene_info()
        assert info is not None
        res = c.batch([
            {"type": "create_primitive", "params": {"primitive": "cube", "name": "LiveBox"}},
            {"type": "delete_object", "params": {"name": "Nope"}},
        ])
        assert [r["ok"] for r in res] == [1, 0]
        assert any(o.get("name") == "LiveBox" for o in c.scene_snapshot())
    finally:
        stop.write_text("1")
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
