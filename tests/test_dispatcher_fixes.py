"""Regression tests for dispatcher-level fixes (TODO_AGENT_FIXES.md F1-F5)."""

import os

import pytest

from ai_db import VectorDB
from ai_db.dispatcher import ServiceDispatcher


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".git").mkdir()  # project marker
    (root / "keep.py").write_text("def kept_function():\n    return 'needle_token'\n")
    (root / "gone.py").write_text("def doomed_function():\n    return 'needle_token'\n")
    db = VectorDB(str(tmp_path / "d.db"))
    db.sync(str(root), project="proj", verbose=False)
    yield root, db, ServiceDispatcher(db=db)
    db.close()


def _paths(results):
    return {os.path.basename(r["abs_path"]) for r in results}


def test_prune_invalidates_cached_query_results(project):
    root, _db, disp = project
    args = {"query": "needle_token", "project": "proj"}
    assert "gone.py" in _paths(disp.execute("query", args))  # now cached
    (root / "gone.py").unlink()
    assert disp.execute("prune", {}) == {"pruned": 1}
    assert "gone.py" not in _paths(disp.execute("query", args))


def test_single_file_sync_respects_aidbignore(project):
    root, db, disp = project
    (root / ".aidbignore").write_text("generated/\n")
    (root / "generated").mkdir()
    ignored = root / "generated" / "out.py"
    ignored.write_text("def ignored_function():\n    return 1\n")
    result = disp.execute("sync", {"path": str(ignored), "project": "proj"})
    assert result["added"] == 0
    assert db.backend.get_file(str(ignored)) is None


def test_single_file_sync_indexes_a_new_file(project):
    root, db, disp = project
    new = root / "fresh.py"
    new.write_text("def fresh_function():\n    return 1\n")
    assert disp.execute("sync", {"path": str(new), "project": "proj"})["added"] == 1
    assert db.backend.get_file(str(new)) is not None


def test_route_skill_uses_the_router_threshold_by_default(project, monkeypatch):
    _, db, disp = project
    seen = {}

    def fake_route(prompt, top_k=3, min_confidence=None, **kw):
        seen["min_confidence"] = min_confidence
        return []

    monkeypatch.setattr(db, "route_skills", fake_route)
    disp.execute("route_skill", {"prompt": "deploy the app"})
    # None lets route_skills apply SKILL_MIN_CONFIDENCE; 0.0 used to disable it.
    assert seen["min_confidence"] is None
    disp.execute("route_skill", {"prompt": "deploy the app", "min_confidence": 0.4})
    assert seen["min_confidence"] == 0.4


def test_keyerror_inside_a_handler_is_not_reported_as_unknown_tool(tmp_path, monkeypatch):
    from mcp_server import StdioMCPServer

    server = StdioMCPServer(str(tmp_path / "m.db"))

    def broken(args):
        return {}["missing_key"]  # a handler bug, not an unknown tool

    monkeypatch.setattr(server.dispatcher._tools["status"], "handler", broken)
    resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "status", "arguments": {}}})
    text = resp["result"]["content"][0]["text"]
    assert resp["result"]["isError"] is True
    assert "Unknown tool" not in text and "missing_key" in text


def test_tool_arguments_cannot_choose_the_database(tmp_path):
    disp = ServiceDispatcher(db_path=str(tmp_path / "real.db"))
    other = tmp_path / "attacker.db"
    disp.execute("status", {"db": str(other)})
    disp.close()
    assert not other.exists()


def test_locate_is_scoped_to_the_callers_project(tmp_path):
    parent = tmp_path / "work"
    for name in ("alpha", "beta"):
        d = parent / name
        d.mkdir(parents=True)
        (d / ".git").mkdir()
        (d / "m.py").write_text(f"def {name}_handler():\n    return 'shared_needle'\n")
    db = VectorDB(str(tmp_path / "l.db"))
    db.sync(str(parent / "alpha"), project="alpha", verbose=False)
    db.sync(str(parent / "beta"), project="beta", verbose=False)
    try:
        hits = db.locate_targets("shared_needle", scope=str(parent), project="alpha")
        assert {h["name"] for h in hits} == {"def alpha_handler"}
        both = db.locate_targets("shared_needle", scope=str(parent), project="alpha",
                                 allowed_projects=["beta"])
        assert {h["name"] for h in both} == {"def alpha_handler", "def beta_handler"}
    finally:
        db.close()
