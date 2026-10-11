"""Daemon: per-user keys, the status page, and watching auto_sync_paths.

Runs the FastAPI app in-process (TestClient); no real socket, no real file events.
"""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from ai_db.config import load_config, parse_config
from ai_db.config_template import build_template
from ai_db.daemon import server
from ai_db.errors import AiDbConfigError
from ai_db.watcher import group_by_root


def _write_config(tmp_path, auto_sync_paths=(), **daemon):
    cfg = {
        "version": 2,
        "storage": {"provider": "sqlite",
                    "options": {"path": str(tmp_path / "auto.db"), "vector_index": "exact"}},
        "retrieval": {"mode": "lexical"},
        "embedding": {"provider": "none"},
        "rerank": {"provider": "none"},
        "access": {"cross_project": {}},
        "auto_sync_paths": [str(p) for p in auto_sync_paths],
        "trace": {"wait_patterns": None, "wait_patterns_extend": None},
        "daemon": {"auto_discover": False,
                   "projects": {"proj": {"db_path": str(tmp_path / "proj.db"), "auto_start": False},
                                "proj2": {"db_path": str(tmp_path / "proj2.db"), "auto_start": False}},
                   **daemon},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _env(monkeypatch, cfg_path, api_keys=None, user_keys=None, admin_key=None):
    monkeypatch.setenv("AI_DB_CONFIG", str(cfg_path))
    for var, value in (("AI_DB_API_KEYS", api_keys), ("AI_DB_USER_KEYS", user_keys),
                       ("AI_DB_ADMIN_KEY", admin_key)):
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)
    server.rate_limiter._buckets.clear()


def _status(client, project, key=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post(f"/projects/{project}/tools/call",
                       json={"name": "status", "arguments": {}}, headers=headers).status_code


# --- per-user keys -------------------------------------------------------------


@pytest.fixture
def users_daemon(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, users={"alice": {"projects": ["proj"]},
                                         "bob": {"projects": ["*"]}})
    _env(monkeypatch, cfg, user_keys="alice=akey,bob=bkey,mallory=mkey")
    with TestClient(server.app) as client:
        yield client


def test_user_keys_alone_turn_auth_on(users_daemon):
    """Fail closed: with only AI_DB_USER_KEYS set, a request without a key is refused."""
    assert _status(users_daemon, "proj") == 401
    assert users_daemon.get("/health").status_code == 200


def test_a_user_reaches_only_the_projects_listed_for_them(users_daemon):
    assert _status(users_daemon, "proj", "akey") == 200
    assert _status(users_daemon, "proj2", "akey") == 403
    assert _status(users_daemon, "proj", "bkey") == 200   # "*"
    assert _status(users_daemon, "proj2", "bkey") == 200
    assert _status(users_daemon, "proj", "wrong") == 403


def test_a_user_without_a_config_entry_gets_nothing(users_daemon):
    assert _status(users_daemon, "proj", "mkey") == 403


def test_user_keys_never_open_management_routes(users_daemon):
    for key in ("akey", "bkey"):
        resp = users_daemon.get("/projects", headers={"Authorization": f"Bearer {key}"})
        assert resp.status_code == 403


def test_project_and_admin_keys_still_work_next_to_user_keys(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, users={"alice": {"projects": ["proj"]}})
    _env(monkeypatch, cfg, api_keys="proj2=p2key", user_keys="alice=akey", admin_key="adm")
    with TestClient(server.app) as client:
        assert _status(client, "proj2", "p2key") == 200
        assert _status(client, "proj", "p2key") == 403
        assert _status(client, "proj2", "adm") == 200
        assert client.get("/projects", headers={"Authorization": "Bearer adm"}).status_code == 200


def test_without_any_key_the_daemon_stays_open(tmp_path, monkeypatch):
    _env(monkeypatch, _write_config(tmp_path, users={"alice": {"projects": ["proj"]}}))
    with TestClient(server.app) as client:
        assert _status(client, "proj2") == 200


def test_daemon_users_config_is_validated():
    data = build_template()
    data["daemon"] = {"users": {"alice": {"projects": "proj"}}}
    with pytest.raises(AiDbConfigError, match="daemon.users.alice.projects"):
        parse_config(data)
    data["daemon"] = {"users": {"alice": {"projects": ["proj"], "admin": True}}}
    with pytest.raises(AiDbConfigError):
        parse_config(data)
    data["daemon"] = {"users": {"alice": {"projects": ["proj"]}}, "watch_sync_paths": True}
    cfg = parse_config(data)
    assert cfg.daemon.users["alice"].allows("proj") and not cfg.daemon.users["alice"].allows("x")
    assert cfg.daemon.watch_sync_paths is True
    assert cfg.daemon.to_dict()["users"] == {"alice": {"projects": ["proj"]}}


# --- status page -----------------------------------------------------------------


@pytest.fixture
def secured_cfg(tmp_path, monkeypatch):
    _env(monkeypatch, _write_config(tmp_path), api_keys="proj=pkey", admin_key="adm")


def test_status_page_for_loopback_clients(secured_cfg):
    with TestClient(server.app, client=("127.0.0.1", 50000)) as client:
        assert _status(client, "proj", "pkey") == 200  # loads the project
        resp = client.get(server.STATUS_PAGE)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    page = resp.text
    assert "ai-db daemon" in page and "Up " in page
    assert "<td>proj</td><td>loaded</td>" in page and "<td>proj2</td><td>not loaded</td>" in page
    assert "File watching is off" in page
    assert "pkey" not in page and "Bearer" not in page


def test_status_page_elsewhere_needs_the_admin_key(secured_cfg):
    with TestClient(server.app) as client:  # client host "testclient": not loopback
        assert client.get(server.STATUS_PAGE).status_code == 401
        assert client.get(server.STATUS_PAGE, headers={"Authorization": "Bearer pkey"}).status_code == 403
        assert client.get(server.STATUS_PAGE, headers={"Authorization": "Bearer adm"}).status_code == 200


def test_status_page_is_not_public_on_an_open_daemon(tmp_path, monkeypatch):
    _env(monkeypatch, _write_config(tmp_path))
    with TestClient(server.app) as client:
        assert client.get(server.STATUS_PAGE).status_code == 401
    with TestClient(server.app, client=("::1", 50000)) as client:
        assert client.get(server.STATUS_PAGE).status_code == 200


def test_status_shows_the_running_call_and_the_queue(tmp_path):
    async def scenario():
        pool = server.ProjectPool(load_config(str(_write_config(tmp_path))))
        server._state.pool = pool
        try:
            await pool.get_or_create("proj")
            release = asyncio.Event()

            async def hold():
                async with pool._exclusive("proj", "sync"):
                    await release.wait()

            first = asyncio.create_task(hold())
            await asyncio.sleep(0.05)
            second = asyncio.create_task(hold())
            await asyncio.sleep(0.05)
            snap = server.status_snapshot()
            release.set()
            await asyncio.gather(first, second)
            after = server.status_snapshot()
            return snap, after
        finally:
            await pool.shutdown()
            server._state.pool = None

    snap, after = asyncio.run(scenario())
    row = next(p for p in snap["projects"] if p["name"] == "proj")
    assert row["running"]["tool"] == "sync" and row["waiting"] == 1
    assert row["models"] == {"embedding": None, "rerank": None}
    row = next(p for p in after["projects"] if p["name"] == "proj")
    assert row["running"] is None and row["waiting"] == 0


def test_status_page_escapes_everything():
    snap = {"uptime_s": 1.0, "auth": "open", "progress_streams": ["<b>tok</b>"],
            "projects": [], "watch": {"roots": {"/x/<y>": "y"}, "batches": 0, "files_synced": 0,
                                      "last_sync": None, "errors": ["<script>alert(1)</script>"]}}
    page = server.render_status_page(snap)
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<b>tok</b>" not in page and "/x/&lt;y&gt;" in page


# --- watching auto_sync_paths ------------------------------------------------------


def test_group_by_root_picks_the_deepest_root(tmp_path):
    outer, inner = tmp_path / "a", tmp_path / "a" / "b"
    grouped = group_by_root([str(outer), str(inner)],
                            [str(inner / "x.py"), str(outer / "y.py"), str(tmp_path / "z.py")])
    assert grouped == {str(inner): [str(inner / "x.py")], str(outer): [str(outer / "y.py")]}


def test_watched_roots(tmp_path):
    (tmp_path / "repo").mkdir()
    cfg = load_config(str(_write_config(tmp_path, auto_sync_paths=[tmp_path / "repo",
                                                                     tmp_path / "missing"])))
    assert server.watched_roots(cfg) == {str(tmp_path / "repo"): "repo"}


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_watch_syncs_changed_files_into_their_project(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def before_start():\n    return 1\n")
    cfg = _write_config(tmp_path, auto_sync_paths=[repo], watch_sync_paths=True, auto_discover=True)
    _env(monkeypatch, cfg)
    seen = []

    async def fake_awatch(roots, debounce_ms):
        seen.append((list(roots), debounce_ms))
        (repo / "b.py").write_text("def after_change():\n    return 2\n")
        yield {str(repo): [str(repo / "b.py")]}
        await asyncio.Event().wait()  # no more events; cancelled at shutdown

    monkeypatch.setattr(server, "awatch_roots", fake_awatch)
    with TestClient(server.app, client=("127.0.0.1", 50000)) as client:
        def indexed(name):
            db = server._state.pool.projects.get("repo") if server._state.pool else None
            return db is not None and db.backend.get_file(str(repo / name)) is not None

        assert _wait_for(lambda: indexed("a.py")), "catch-up sync at start"
        assert _wait_for(lambda: indexed("b.py")), "changed file synced"
        assert _wait_for(lambda: server._state.watch.files_synced == 1)
        page = client.get(server.STATUS_PAGE).text
    assert seen == [([str(repo)], server.DAEMON_WATCH_DEBOUNCE_MS)]
    assert f"<code>{repo}</code> &rarr; repo" in page and "1 file(s) synced" in page


def test_no_watching_unless_configured(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _env(monkeypatch, _write_config(tmp_path, auto_sync_paths=[repo]))

    async def never(roots, debounce_ms):
        raise AssertionError("must not watch")
        yield {}  # pragma: no cover

    monkeypatch.setattr(server, "awatch_roots", never)
    with TestClient(server.app):
        assert server._state.watch is None and server._state.watch_task is None
