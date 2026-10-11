"""Daemon server: tool calls, client cwd, progress, auth, rate limit, locks, reload.

Runs the FastAPI app in-process (TestClient / ASGI transport); no real socket.
"""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_db.config import load_config
from ai_db.daemon import server
from ai_db.daemon.client import DaemonClient


def _write_config(tmp_path, **daemon):
    cfg = {
        "version": 2,
        "storage": {"provider": "sqlite",
                    "options": {"path": str(tmp_path / "auto.db"), "vector_index": "exact"}},
        "retrieval": {"mode": "lexical"},
        "embedding": {"provider": "none"},
        "rerank": {"provider": "none"},
        "access": {"cross_project": {}},
        "auto_sync_paths": [],
        "trace": {"wait_patterns": None, "wait_patterns_extend": None},
        "daemon": {"auto_discover": False,
                   "projects": {"proj": {"db_path": str(tmp_path / "proj.db"), "auto_start": False}},
                   **daemon},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "pkg" / "mod.py").write_text("def daemon_target():\n    return 1\n")
    return root


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_DB_CONFIG", str(_write_config(tmp_path)))
    monkeypatch.delenv("AI_DB_API_KEYS", raising=False)
    monkeypatch.delenv("AI_DB_ADMIN_KEY", raising=False)
    server.rate_limiter._buckets.clear()
    with TestClient(server.app) as client:
        yield client


def _call(client, tool, arguments, project="proj", headers=None, meta=None):
    body = {"name": tool, "arguments": arguments}
    if meta is not None:
        body["_meta"] = meta
    resp = client.post(f"/projects/{project}/tools/call", json=body, headers=headers or {})
    return resp


def _text(resp):
    data = resp.json()
    return data["isError"], data["content"][0]["text"]


# --- tool calls and the client's cwd (H1) ------------------------------------


def test_relative_paths_resolve_against_the_clients_cwd(daemon, repo, monkeypatch):
    monkeypatch.chdir("/")  # like a daemonized server
    is_error, text = _text(_call(daemon, "sync", {"path": ".", "cwd": str(repo)}))
    assert not is_error, text
    assert json.loads(text)["added"] == 1
    is_error, text = _text(_call(daemon, "symbol", {"name": "daemon_target", "cwd": str(repo / "pkg")}))
    assert not is_error and "daemon_target" in text


def test_unknown_tool_and_unknown_project(daemon):
    is_error, text = _text(_call(daemon, "no_such_tool", {}))
    assert is_error and "Unknown tool" in text
    resp = _call(daemon, "status", {}, project="not-registered")  # auto_discover is off
    assert resp.status_code == 404


def test_handler_keyerror_is_not_unknown_tool(daemon, monkeypatch):
    _call(daemon, "status", {})  # load the project
    dispatcher = server.get_pool().dispatchers["proj"]
    monkeypatch.setattr(dispatcher._tools["status"], "handler", lambda args: {}["boom"])
    is_error, text = _text(_call(daemon, "status", {}))
    assert is_error and "Unknown tool" not in text and "boom" in text


# --- progress (H2) -------------------------------------------------------------


def test_meta_progress_token_is_parsed():
    req = server.ToolCallRequest.model_validate(
        {"name": "sync", "arguments": {}, "_meta": {"progressToken": "t1"}})
    assert req.meta == {"progressToken": "t1"}


def test_progress_events_reach_a_listener(tmp_path, repo):
    async def scenario():
        pool = server.ProjectPool(load_config(str(_write_config(tmp_path))))
        queue = pool.attach_listener("tok")
        await pool.run_tool("proj", "sync", {"path": str(repo)}, progress_token="tok")
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        await pool.shutdown()
        return events

    events = asyncio.run(scenario())
    assert any(e["type"] == "progress" for e in events)
    assert events[-1]["type"] == "done"


def test_health_answers_while_a_tool_call_runs(tmp_path):
    async def scenario():
        pool = server.ProjectPool(load_config(str(_write_config(tmp_path))))
        server._state.pool = pool
        _, dispatcher = await pool.get_or_create("proj")
        dispatcher._tools["status"].handler = lambda args: time.sleep(1.0) or {"ok": True}
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://d") as client:
            slow = asyncio.create_task(client.post(
                "/projects/proj/tools/call", json={"name": "status", "arguments": {}}))
            await asyncio.sleep(0.1)
            t0 = time.perf_counter()
            health = await client.get("/health")
            health_s = time.perf_counter() - t0
            await slow
        await pool.shutdown()
        server._state.pool = None
        return health.status_code, health_s

    status, seconds = asyncio.run(scenario())
    assert status == 200
    assert seconds < 0.5  # not stuck behind the 1 s tool call


# --- auth and rate limit (H3) --------------------------------------------------


@pytest.fixture
def secured(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_DB_CONFIG", str(_write_config(tmp_path)))
    monkeypatch.setenv("AI_DB_API_KEYS", "proj=projkey")
    monkeypatch.setenv("AI_DB_ADMIN_KEY", "adminkey")
    server.rate_limiter._buckets.clear()
    with TestClient(server.app) as client:
        yield client


def _bearer(key):
    return {"Authorization": f"Bearer {key}"}


def test_project_routes_need_the_project_or_admin_key(secured):
    assert _call(secured, "status", {}).status_code == 401
    assert _call(secured, "status", {}, headers=_bearer("wrong")).status_code == 403
    assert _call(secured, "status", {}, headers=_bearer("projkey")).status_code == 200
    assert _call(secured, "status", {}, headers=_bearer("adminkey")).status_code == 200


def test_management_routes_need_the_admin_key(secured):
    for method, path in [("post", "/daemon/reload"), ("post", "/daemon/shutdown"),
                         ("get", "/projects"), ("get", "/metrics"),
                         ("post", "/query/cross-project"), ("delete", "/projects/proj")]:
        resp = getattr(secured, method)(path, headers=_bearer("projkey"))
        assert resp.status_code == 403, (method, path, resp.status_code)
        assert getattr(secured, method)(path).status_code == 401, (method, path)
    assert secured.get("/projects", headers=_bearer("adminkey")).status_code == 200
    assert secured.get("/health").status_code == 200  # always open


def test_register_with_arbitrary_db_path_needs_admin(secured, tmp_path):
    body = {"name": "evil", "db_path": str(tmp_path / "evil.db")}
    assert secured.post("/projects", json=body).status_code == 401
    assert not (tmp_path / "evil.db").exists()


def test_rate_limit_is_per_project(daemon, monkeypatch):
    monkeypatch.setattr(server, "DAEMON_RATE_LIMIT", 2)
    codes = [_call(daemon, "status", {}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# --- pool locks and reload (H6) -----------------------------------------------


def test_warm_unloaded_project_does_not_deadlock(tmp_path):
    async def scenario():
        pool = server.ProjectPool(load_config(str(_write_config(tmp_path))))
        await asyncio.wait_for(pool.warm_project("proj"), timeout=10)
        warmed = "proj" in pool.warmed and "proj" in pool.projects
        await pool.remove("proj")
        await asyncio.wait_for(pool.get_or_create("proj"), timeout=10)  # lock still usable
        await pool.shutdown()
        return warmed

    assert asyncio.run(scenario())


def test_reload_keeps_runtime_registered_projects(daemon, tmp_path):
    body = {"name": "extra", "db_path": str(tmp_path / "extra.db"), "auto_start": False}
    assert daemon.post("/projects", json=body).status_code == 200
    assert daemon.post("/daemon/reload").status_code == 200
    names = {p["name"] for p in daemon.get("/projects").json()}
    assert {"proj", "extra"} <= names


def test_daemon_reads_the_config_named_by_ai_db_config(daemon, tmp_path):
    # The fixture points AI_DB_CONFIG at cfg.json, which registers "proj" with proj.db.
    projects = {p["name"]: p for p in daemon.get("/projects").json()}
    assert projects["proj"]["db_path"] == str(tmp_path / "proj.db")


# --- client (H7) ---------------------------------------------------------------


def test_client_sends_cwd_and_does_not_mutate_arguments(monkeypatch, tmp_path):
    sent = {}

    async def fake_request(self, method, path, admin=False, **kw):
        sent.update(path=path, body=kw["json"])
        return {"isError": False, "content": [{"type": "text", "text": "{\"ok\": true}"}]}

    monkeypatch.setattr(DaemonClient, "_request", fake_request)
    monkeypatch.chdir(tmp_path)
    args = {"project": "proj", "path": "."}
    assert DaemonClient("http://d").execute("sync", args) == {"ok": True}
    assert args == {"project": "proj", "path": "."}
    assert sent["path"] == "/projects/proj/tools/call"
    assert sent["body"]["arguments"] == {"path": ".", "cwd": str(tmp_path)}


def test_checkpoint_skips_a_project_closed_while_waiting(tmp_path):
    async def scenario():
        pool = server.ProjectPool(load_config(str(_write_config(tmp_path))))
        db, _ = await pool.get_or_create("proj")
        db.backend.close()  # what idle cleanup does between the snapshot and the lock
        await pool.checkpoint_all()  # used to raise "Storage backend is closed"
        pool.projects.pop("proj", None)
        await pool.shutdown()

    asyncio.run(scenario())
