"""Daemon: per-user keys, the status page, and watching auto_sync_paths.

Runs the FastAPI app in-process (TestClient); no real socket, no real file events.
"""

import json

import pytest
from fastapi.testclient import TestClient

from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.daemon import server
from ai_db.errors import AiDbConfigError


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
    data["daemon"] = {"users": {"alice": {"projects": ["proj"]}}}
    cfg = parse_config(data)
    assert cfg.daemon.users["alice"].allows("proj") and not cfg.daemon.users["alice"].allows("x")
    assert cfg.daemon.to_dict()["users"] == {"alice": {"projects": ["proj"]}}
