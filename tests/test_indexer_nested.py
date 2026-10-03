"""Nested checkouts (git worktrees, sub-repos) are separate projects and are not indexed."""

import os

from ai_db import VectorDB


def _repo_with_worktree(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "pkg" / "core.py").write_text("def real_symbol():\n    return 1\n")
    wt = repo / ".claude" / "worktrees" / "agent-x"
    (wt / "pkg").mkdir(parents=True)
    (wt / ".git").write_text("gitdir: /elsewhere/.git/worktrees/agent-x\n")  # worktrees use a .git *file*
    (wt / "pkg" / "core.py").write_text("def real_symbol():\n    return 1\n")
    return repo, wt


def test_scan_skips_nested_checkout(tmp_path):
    repo, wt = _repo_with_worktree(tmp_path)
    db = VectorDB(str(tmp_path / "n.db"))
    found = db.scan_directory(str(repo))
    assert os.path.abspath(repo / "pkg" / "core.py") in found
    assert not any(p.startswith(str(wt)) for p in found)
    db.close()


def test_sync_paths_skips_nested_checkout(tmp_path):
    repo, wt = _repo_with_worktree(tmp_path)
    db = VectorDB(str(tmp_path / "n.db"))
    db.sync_paths(str(repo), [str(wt / "pkg" / "core.py"), str(repo / "pkg" / "core.py")], project="p")
    defs = db.query_symbol("real_symbol", project="p")
    assert [d["abs_path"] for d in defs] == [os.path.abspath(repo / "pkg" / "core.py")]
    db.close()
