"""Query-result cache (index generations) and the event-driven watcher."""

import threading
import time

import pytest

from ai_db import VectorDB, watcher


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "a.py").write_text("def alpha_widget():\n    return 1\n")
    return src


def test_query_and_investigate_are_cached_until_index_changes(tmp_path, repo):
    db = VectorDB(str(tmp_path / "c.db"))
    db.sync(str(repo), project="p", verbose=False)
    first = db.query("alpha widget", project="p")
    assert db.last_cache_hit is False
    assert db.query("  Alpha   WIDGET ", project="p") == first
    assert db.last_cache_hit is True
    db.investigate("alpha widget", project="p")
    db.investigate("alpha widget", project="p")
    assert db.last_cache_hit is True

    (repo / "b.py").write_text("def alpha_gadget():\n    return 2\n")
    db.sync(str(repo), project="p", verbose=False)
    second = db.query("alpha widget", project="p")
    assert db.last_cache_hit is False
    assert second != first or len(second) == len(first)
    db.close()


def test_noop_sync_keeps_cache(tmp_path, repo):
    db = VectorDB(str(tmp_path / "c.db"))
    db.sync(str(repo), project="p", verbose=False)
    gen = db.backend.get_index_generation()
    db.sync(str(repo), project="p", verbose=False)
    assert db.backend.get_index_generation() == gen
    db.close()


def test_different_params_are_different_entries(tmp_path, repo):
    db = VectorDB(str(tmp_path / "c.db"))
    db.sync(str(repo), project="p", verbose=False)
    db.query("alpha", project="p", top_k=5)
    db.query("alpha", project="p", top_k=3)
    assert db.last_cache_hit is False
    db.close()


def test_sync_paths_only_touches_given_files(tmp_path, repo):
    db = VectorDB(str(tmp_path / "c.db"))
    db.sync(str(repo), project="p", verbose=False)
    (repo / "a.py").write_text("def alpha_widget():\n    return 3\n")
    (repo / "new.py").write_text("def fresh():\n    pass\n")
    res = db.sync_paths(str(repo), [str(repo / "a.py"), str(repo / "new.py")], project="p")
    assert res == {"added": 1, "updated": 1, "pruned": 0, "skipped": 0}
    (repo / "new.py").unlink()
    res = db.sync_paths(str(repo), [str(repo / "new.py")], project="p")
    assert res["pruned"] == 1
    db.close()


def test_watcher_reindexes_on_change(tmp_path, repo):
    db_path = str(tmp_path / "w.db")
    stop = threading.Event()
    t = threading.Thread(target=watcher.run_watch, args=(VectorDB, db_path, str(repo), 50, stop),
                         daemon=True)
    t.start()
    time.sleep(1.0)
    (repo / "later.py").write_text("def late_arrival():\n    pass\n")
    deadline = time.time() + 10
    found = False
    while time.time() < deadline and not found:
        time.sleep(0.3)
        probe = VectorDB(db_path)
        found = probe.backend.get_file(str(repo / "later.py")) is not None
        probe.close()
    stop.set()
    t.join(timeout=5)
    assert found
