"""ai_db.history: commit classification, patch parsing, mining, SZZ-lite, conventions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ai_db.history import ConventionOptions, MineOptions, classify, conventions, mine
from ai_db.history.__main__ import main as history_main
from ai_db.history.conventions import name_style
from ai_db.history.gitlog import glob_regex, parse_patch, path_filter


def _git(repo: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin",
           "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo)}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _commit(r, {"src/app.py": "def load(path):\n    return open(path).read()\n",
                "src/util.py": "X = 1\n"}, "feat: add loader")
    _commit(r, {"src/app.py": "def load(path):\n    return open(path).read()\n\n"
                              "def size(data):\n    return len(data) / 0\n",
                "src/util.py": "X = 2\n"}, "feat: add size helper")
    _commit(r, {"src/app.py": "def load(path):\n    with open(path) as fh:\n"
                              "        return fh.read()\n\n"
                              "def size(data):\n    return len(data) / 0\n",
                "src/util.py": "X = 3\n"}, "fix: close the file in load")
    _commit(r, {"src/app.py": "def load(path):\n    with open(path) as fh:\n"
                              "        return fh.read()\n\n"
                              "def size(data):\n    return len(data)\n",
                "src/util.py": "X = 4\n"}, "fix(app): size divided by zero")
    _commit(r, {"README.md": "docs only\n"}, "docs: readme")
    return r


def test_classify_conventional_keywords_and_reverts() -> None:
    assert classify("fix(pkg): update module readers")[0] == "fix"
    assert classify("Hotfix/copybara: Fix failing lfs push")[0] == "fix"
    assert classify("SimHub: Fix kube containers restart policy")[0] == "fix"
    assert classify("feat: add fixture loader")[0] == "feat"  # conventional type wins
    assert classify("Simulation: Reduce shader variants")[0] == "other"
    assert classify("refactor(seed): remove PlcTemplateSeeder")[0] == "refactor"
    kind, sha = classify('Revert "wip"', "This reverts commit abcdef1234567.")
    assert (kind, sha) == ("revert", "abcdef1234567")
    assert classify("prefix handling for paths")[0] != "fix"


def test_parse_patch_tracks_line_numbers() -> None:
    patch = ("diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
             "@@ -2,2 +2,3 @@ def f():\n-    a = 1\n+    a = 2\n+    b = 3\n     return a\n"
             "diff --git a/g.py b/g.py\nnew file mode 100644\n--- /dev/null\n+++ b/g.py\n"
             "@@ -0,0 +1 @@\n+x = 1\n")
    hunks = parse_patch(patch)
    assert [h.file for h in hunks] == ["f.py", "g.py"]
    assert hunks[0].removed == [(2, "    a = 1")]
    assert hunks[0].added == [(2, "    a = 2"), (3, "    b = 3")]
    assert hunks[0].header == "def f():"
    assert hunks[1].added == [(1, "x = 1")]


def test_globs_and_default_excludes() -> None:
    assert glob_regex("*.cs").match("a/b/C.cs")
    assert glob_regex("third_party/**").match("third_party/x/y.cc")
    assert not glob_regex("third_party/**").match("src/third_party.cc")
    sel = path_filter()
    assert sel("SimHub/src/Foo.cs")
    assert not sel("third_party/lib/foo.cc")
    assert not sel("bazel-out/k8/bin/foo.py")
    assert not sel("UI/node_modules/x/index.js")
    assert not sel("Simulation/Assets/Foo.cs.meta")


def test_mine_splits_classifies_and_blames(repo: Path) -> None:
    report = mine(str(repo), MineOptions(cutoff_ratio=0.5, cochange_min_support=2))
    assert report["cutoff"]["train_commits"] == 2  # README commit has no code: dropped
    assert report["cutoff"]["test_commits"] == 2
    test = report["test"]
    assert test["kinds"] == {"fix": 2}
    fixes = [c for c in test["commits"] if c["kind"] == "fix"]
    assert all(c["sha"] in test["hunks"] for c in fixes)
    # SZZ-lite: the "/ 0" line removed by the second fix came from "feat: add size helper".
    feat_sha = report["train"]["commits"][1]["sha"]
    intro = [i for i in test["introduced"] if "/ 0" in i["text"]]
    assert intro and intro[0]["intro"] == feat_sha and intro[0]["file"] == "src/app.py"
    assert intro[0]["line"] == 5
    # app.py and util.py always change together.
    pairs = {(p["a"], p["b"]) for p in report["train"]["cochange"]}
    assert ("src/app.py", "src/util.py") in pairs
    json.dumps(report)  # serialisable


def test_mine_explicit_cutoff_and_hotspots(repo: Path) -> None:
    shas = _git(repo, "log", "--reverse", "--format=%H").split()
    report = mine(str(repo), MineOptions(cutoff=shas[2]))
    assert report["cutoff"]["sha"] == shas[2]
    assert report["cutoff"]["train_commits"] == 3
    spots = {s["file"]: s for s in report["train"]["hotspots"] + report["test"]["hotspots"]}
    assert "src/app.py" not in spots or spots["src/app.py"]["fix_commits"] >= 2


def test_name_style() -> None:
    assert name_style("GetInstanceAsync") == "PascalCase"
    assert name_style("getInstance") == "camelCase"
    assert name_style("get_instance") == "snake_case"
    assert name_style("run") == "lower"
    assert name_style("MAX_SIZE") == "UPPER_CASE"
    assert name_style("_field") == "_lower"
    assert name_style("_my_field") == "_snake_case"
    assert name_style("_myField") == "_camelCase"
    assert name_style("__init__") == "dunder"


def test_conventions_reads_revision(repo: Path) -> None:
    _commit(repo, {"src/more.py": "def read_all(x):\n    return x\n\n"
                                  "def write_all(x):\n    return x\n\n"
                                  "def badName(x):\n    return x\n"}, "feat: more")
    report = conventions(str(repo), ConventionOptions())
    py = next(c for c in report["naming"]
              if c["language"] == "python" and c["kind"] == "function")
    assert py["dominant"] == "snake_case"
    assert py["total"] >= 5
    assert any("badName" in e for e in py["counter_examples"])
    assert report["files_parsed"] >= 3


def test_cli_writes_report(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert history_main(["mine", str(repo), "--cutoff-ratio", "0.5", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["tool"] == "ai-db history mine"
    assert data["cutoff"]["test_commits"] == 2
