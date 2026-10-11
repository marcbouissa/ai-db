"""The one diff engine, ``ai_db.analysis.diff`` (K1 step 1, G7)."""
import os
import subprocess

import pytest

from ai_db import VectorDB
from ai_db.analysis.diff import changed_lines, file_diff

ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
           GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, env=ENV, capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.py").write_text("one\ntwo\nthree\nfour\nfive\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def test_unchanged_file_has_no_spans(repo, tmp_path):
    assert file_diff(str(repo / "a.py"), "HEAD") == {"added": [], "removed": [], "changed": []}
    db = VectorDB(str(tmp_path / "d.db"))
    try:
        db.sync(str(repo), project="r", verbose=False)
        assert db.diff_file(str(repo / "a.py"), since="last") == {
            "added": [], "removed": [], "changed": []}
    finally:
        db.close()


def test_file_new_since_the_ref_is_entirely_added(repo):
    (repo / "new.py").write_text("x = 1\ny = 2\n")
    assert file_diff(str(repo / "new.py"), "HEAD")["added"] == [[1, 2]]  # untracked
    _git(repo, "add", "new.py")
    _git(repo, "commit", "-q", "-m", "add")
    # tracked now, absent at HEAD~1: `git show HEAD~1:new.py` used to raise
    assert file_diff(str(repo / "new.py"), "HEAD~1")["added"] == [[1, 2]]


def test_hunk_kinds(repo):
    (repo / "a.py").write_text("one\nTWO\nthree\nfive\nsix\n")
    assert file_diff(str(repo / "a.py"), "HEAD") == {
        "added": [[5, 5]], "removed": [[4, 4]], "changed": [[2, 2]]}


def test_deletion_only_hunk_seeds_the_line_it_followed(repo):
    (repo / "a.py").write_text("one\ntwo\nfive\n")  # three and four deleted
    assert changed_lines("HEAD", str(repo)) == {str(repo / "a.py"): [2]}


def test_diff_mode_indexes_new_untracked_files(repo, tmp_path):
    """G8: a new file is listed under changes *and* has evidence; ignored files stay out."""
    (repo / ".aidbignore").write_text("secret.py\n")
    _git(repo, "add", ".aidbignore")
    _git(repo, "commit", "-q", "-m", "ignore")
    db = VectorDB(str(tmp_path / "d.db"))
    try:
        db.sync(str(repo), project="r", verbose=False)
        (repo / "new.py").write_text("def gamma(x):\n    return x + 1\n")
        (repo / "secret.py").write_text("def hidden():\n    return 1\n")
        pack = db.investigate("", mode="diff", since="HEAD", root=str(repo), project="r")
        assert "gamma" in [e["qualified_name"] for e in pack["evidence"]]
        assert db.backend.get_file(str(repo / "new.py")) is not None
        assert db.backend.get_file(str(repo / "secret.py")) is None
        flags = {c["filepath"]: c.get("unindexed", False) for c in pack["changes"]}
        assert flags[str(repo / "secret.py")] is True and flags[str(repo / "new.py")] is False
        assert pack["query"] == "changes since HEAD"
    finally:
        db.close()


def test_untracked_files_count_as_changed(repo):
    (repo / "b.py").write_text("x\ny\n")
    assert changed_lines("HEAD", str(repo))[str(repo / "b.py")] == [1, 2]
