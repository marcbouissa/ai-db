"""A4: staleness signal on read paths; diff mode defaults to HEAD and sees untracked files."""

import os
import subprocess

import pytest


from ai_db import VectorDB
from ai_db.dispatcher import ServiceDispatcher
from ai_db.errors import AiDbConfigError
from ai_db.search import freshness


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def repo(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "a.py").write_text("def alpha():\n    return 1\n")
    (src / "b.py").write_text("def beta():\n    return 2\n")
    _git(src, "init", "-q")
    _git(src, "add", "a.py", "b.py")
    _git(src, "commit", "-q", "-m", "init")
    return src


@pytest.fixture
def disp(tmp_path, repo):
    db = VectorDB(str(tmp_path / "f.db"))
    db.sync(str(repo), project="p", verbose=False)
    d = ServiceDispatcher(db=db)
    yield d
    db.close()


def _touch_change(path, text):
    path.write_text(text)
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 10))


def test_modified_file_is_resynced(disp, repo):
    _touch_change(repo / "a.py", "\n\n\ndef alpha():\n    return 1\n")
    res = disp.execute("symbol", {"name": "alpha", "project": "p"})
    assert res and not any(r.get("stale") for r in res)
    assert res[0]["line"] == 4
    assert "stale_files" not in res


def test_touch_without_content_change_is_fresh(disp, repo):
    p = repo / "a.py"
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 10))
    assert freshness.detect_stale(disp._get_db({}), {str(p)}) == {}


def test_above_bound_is_flagged(disp, repo, monkeypatch):
    monkeypatch.setattr(freshness, "STALE_RESYNC_MAX_FILES", 0)
    _touch_change(repo / "a.py", "\n\n\ndef alpha():\n    return 1\n")
    res = disp.execute("symbol", {"name": "alpha", "project": "p"})
    assert any(r.get("stale") for r in res)
    out = disp.execute("locate", {"query": "alpha", "scope": str(repo), "format": "stub"})
    assert out.startswith("# stale:")
    inv = disp.execute("investigate", {"query": "alpha", "project": "p", "format": "json"})
    assert str(repo / "a.py") in inv["stale_files"]


def test_deleted_file_reported_not_crash(disp, repo, monkeypatch):
    monkeypatch.setattr(freshness, "STALE_RESYNC_MAX_FILES", 0)
    (repo / "b.py").unlink()
    out = disp.execute("locate", {"query": "beta", "scope": str(repo), "format": "stub"})
    assert "stale" in out


def test_deleted_file_resync_prunes(disp, repo):
    (repo / "b.py").unlink()
    res = disp.execute("symbol", {"name": "beta", "project": "p"})
    assert not res


def test_diff_mode_without_since_includes_untracked(disp, repo):
    (repo / "new_mod.py").write_text("def gamma():\n    return 3\n")
    (repo / "a.py").write_text("def alpha():\n    return 99\n")
    disp._get_db({}).sync(str(repo), project="p", verbose=False)
    pack = disp.execute("investigate", {"query": "", "mode": "diff", "root": str(repo),
                                        "format": "json"})
    changed = {c["filepath"] for c in pack["changes"]}
    assert str(repo / "new_mod.py") in changed
    assert str(repo / "a.py") in changed


def test_diff_mode_outside_git_raises(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x.py").write_text("def x():\n    pass\n")
    db = VectorDB(str(tmp_path / "n.db"))
    try:
        db.sync(str(plain), project="p", verbose=False)
        with pytest.raises(AiDbConfigError):
            ServiceDispatcher(db=db).execute(
                "investigate", {"query": "", "mode": "diff", "root": str(plain)})
    finally:
        db.close()
