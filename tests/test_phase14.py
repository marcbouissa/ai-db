"""Phase 14: agent experience.

14.1 MCP progress notifications
14.2 diff-mode investigation
14.3 cache isolation
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import subprocess
from typing import Any

import pytest

from ai_db import VectorDB
from ai_db.analysis.investigate import MODES, Investigator, _git_changed_spans
from ai_db.errors import AiDbConfigError
from ai_db.parser.ts_graph import extract_graph
from ai_db.storage.models import ChunkRecord, FileRecord
from mcp_server import StdioMCPServer

# ==============================================================================
# helpers
# ==============================================================================

def _source_tree(root: str, n: int = 120) -> str:
    src = os.path.join(root, "src")
    os.makedirs(src, exist_ok=True)
    for i in range(n):
        with open(os.path.join(src, f"m{i}.py"), "w", encoding="utf-8") as fh:
            fh.write(f"def f{i}():\n    return {i}\n")
    return src


def _seed(db: Any, chunks: list[ChunkRecord]) -> None:
    """Insert chunks directly, declaring the files first (FK to the files table)."""
    for path in {c.filepath for c in chunks}:
        db.backend.upsert_file(FileRecord(path, "h", 0.0, 1))
    db.backend.insert_chunks(chunks)


@pytest.fixture()
def lexical_config(tmp_path):
    path = tmp_path / "lex.json"
    path.write_text(json.dumps({
        "version": 2,
        "storage": {"provider": "sqlite",
                    "options": {"path": str(tmp_path / "p.db"),
                                "vector_index": "exact"}},
        "retrieval": {"mode": "lexical"},
        "embedding": {"provider": "none"},
        "rerank": {"provider": "none"},
        "access": {"cross_project": {}},
        "auto_sync_paths": [],
        "trace": {"wait_patterns": None, "wait_patterns_extend": None},
    }), encoding="utf-8")
    return str(path)


# ==============================================================================
# 14.1 progress notifications
# ==============================================================================

def test_progress_token_absent_emits_nothing(tmp_path, lexical_config, monkeypatch):
    """A client that did not ask for progress must not receive any."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    src = _source_tree(str(tmp_path))
    buf = io.StringIO()
    srv = StdioMCPServer(str(tmp_path / "p.db"), stdout=buf)
    res = srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "sync", "arguments": {"path": src}}})
    assert res["result"]["isError"] is False
    assert buf.getvalue() == ""


def test_progress_token_emits_notifications(tmp_path, lexical_config, monkeypatch):
    """With a progressToken, a long sync streams notifications/progress frames."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    src = _source_tree(str(tmp_path))
    buf = io.StringIO()
    srv = StdioMCPServer(str(tmp_path / "p.db"), stdout=buf)
    srv.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "sync", "arguments": {"path": src},
                                   "_meta": {"progressToken": "tok-1"}}})
    frames = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    assert frames, "expected progress notifications"
    for frame in frames:
        assert frame["method"] == "notifications/progress"
        assert frame["params"]["progressToken"] == "tok-1"
        assert frame["params"]["total"] > 0
    # first, every 50th, and the last
    progresses = [f["params"]["progress"] for f in frames]
    assert progresses[0] == 1
    assert progresses[-1] == frames[-1]["params"]["total"]
    assert 50 in progresses


def test_progress_sink_detached_after_call(tmp_path, lexical_config, monkeypatch):
    """The sink is scoped to one call: a later direct call cannot report."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    src = _source_tree(str(tmp_path), n=5)
    srv = StdioMCPServer(str(tmp_path / "p.db"), stdout=io.StringIO())
    srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "sync", "arguments": {"path": src},
                                   "_meta": {"progressToken": "t"}}})
    db = srv.dispatcher._resolve_db({})
    assert db.indexer.progress_sink is None


def test_indexer_reports_nothing_without_sink(tmp_path, lexical_config, monkeypatch):
    """A raw backend has no sink, so indexing is silent rather than an error."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    src = _source_tree(str(tmp_path), n=3)
    db = VectorDB(str(tmp_path / "raw.db"))
    db.indexer.progress_sink = None
    db.sync(src, project="p", verbose=False)
    db.close()


# ==============================================================================
# 14.2 diff mode
# ==============================================================================

def test_diff_is_a_valid_mode():
    assert "diff" in MODES


def test_diff_requires_since(tmp_path, lexical_config, monkeypatch):
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    db = VectorDB(str(tmp_path / "d.db"))
    with pytest.raises(AiDbConfigError, match="requires 'since'"):
        db.investigate("", mode="diff")
    db.close()


def test_diff_outside_a_repo_raises(tmp_path, lexical_config, monkeypatch):
    """An empty pack would read as 'nothing to review'; it must be an error."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    db = VectorDB(str(tmp_path / "d.db"))
    with pytest.raises(AiDbConfigError, match="git repository"):
        db.investigate("", mode="diff", since="HEAD", root=str(outside))
    db.close()


def test_git_changed_spans_parses_hunks(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, env=env,
                              capture_output=True, text=True, check=True)
    git("init", "-q")
    (repo / "sub" / "a.py").write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (repo / "sub" / "a.py").write_text("one\ntwo\nTHREE\nfour\nFIVE\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "edit")

    spans = _git_changed_spans("HEAD~1", str(repo))
    assert set(spans) == {str(repo / "sub" / "a.py")}
    assert spans[str(repo / "sub" / "a.py")] == [3, 5]


def test_diff_mode_seeds_the_changed_symbols(tmp_path, lexical_config, monkeypatch):
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, env=env,
                              capture_output=True, text=True, check=True)
    git("init", "-q")
    (repo / "sub" / "billing.py").write_text(
        "def total(a):\n    return a\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")

    db = VectorDB(str(tmp_path / "d.db"))
    db.sync(str(repo), project="repo", verbose=False)

    (repo / "sub" / "billing.py").write_text(
        "def total(a):\n    return a * 2\n", encoding="utf-8")
    pack = db.investigate("", mode="diff", since="HEAD", root=str(repo), project="repo")
    names = [e["qualified_name"] for e in pack["entry_points"]]
    assert "total" in names

    # a diff pack also answers "what breaks because of this"
    assert "call_graph" in pack
    db.close()


def test_line_ranges_collapses_spans():
    from ai_db.analysis.investigate import _line_ranges
    assert _line_ranges([1, 2, 3, 7, 9, 10]) == "L1-3 L7 L9-10"
    assert _line_ranges([5]) == "L5"
    assert _line_ranges([]) == ""


def test_diff_seeds_prefer_the_edited_symbol_over_its_class(tmp_path, lexical_config,
                                                             monkeypatch):
    """A class_header spans the whole file, so it must not win every seed slot."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    db = VectorDB(str(tmp_path / "seed.db"))
    _seed(db, [
        ChunkRecord("big.py", "class_header", "Widget", 1, 400,
                    "class Widget:\n    pass\n", project="global", qualified_name="Widget"),
        ChunkRecord("big.py", "code", "Widget.render", 10, 40,
                    "def render(self):\n    return 1\n", project="global",
                    qualified_name="Widget.render"),
        ChunkRecord("doc.md", "section", "Design notes", 1, 300, "notes " * 200,
                    project="global", qualified_name="Design notes"),
    ])
    inv = Investigator(db)
    spans = {"big.py": list(range(10, 15)), "doc.md": list(range(1, 20))}
    seeds = inv._diff_seeds(spans, "", ["global"])
    names = [s.qualified_name for s in seeds]
    # The 31-line method is the seed; the 400-line class that merely contains
    # it is dropped entirely rather than ranked below it.
    assert "Widget.render" in names
    assert "Widget" not in names
    # ...and seeds interleave across files rather than one file taking the lot.
    assert len({s.filepath for s in seeds}) == 2
    db.close()


def test_diff_seeds_boost_a_query_match(tmp_path, lexical_config, monkeypatch):
    """The query is the caller narrowing the diff, so it must outrank noise."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    db = VectorDB(str(tmp_path / "seed2.db"))
    _seed(db, [
        ChunkRecord("a.py", "code", "unrelated", 1, 200,
                    "def unrelated():\n" + "    x = 1\n" * 100, project="global",
                    qualified_name="unrelated"),
        ChunkRecord("b.py", "code", "rebuild_centrality", 1, 5,
                    "def rebuild_centrality():\n    return 1\n", project="global",
                    qualified_name="rebuild_centrality"),
    ])
    inv = Investigator(db)
    spans = {"a.py": list(range(1, 150)), "b.py": [2, 3]}
    seeds = inv._diff_seeds(spans, "rebuild centrality", ["global"])
    assert seeds[0].qualified_name == "rebuild_centrality"
    db.close()


def test_diff_mode_is_distinct_from_explain(tmp_path, lexical_config, monkeypatch):
    """The two modes must not share a cache entry."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    repo = tmp_path / "repo"
    repo.mkdir()
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True,
                   capture_output=True)
    (repo / "m.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, env=env, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "b"], cwd=repo, env=env, check=True,
                   capture_output=True)

    db = VectorDB(str(tmp_path / "d.db"))
    db.sync(str(repo), project="repo", verbose=False)
    d1 = db.investigate("helper", mode="diff", since="HEAD", root=str(repo), project="repo")
    e1 = db.investigate("helper", mode="explain", project="repo")
    assert d1["mode"] == "diff"
    assert e1["mode"] == "explain"
    db.close()


# ==============================================================================
# 14.3 cache isolation
# ==============================================================================

def test_cache_key_covers_the_whole_retrieval_stack(tmp_path, lexical_config, monkeypatch):
    """Every field that changes the answer must change the key."""
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    db = VectorDB(str(tmp_path / "sig.db"))
    base = db.retrieval_signature()
    assert set(base) == {"mode", "embedding_model", "rerank_model",
                         "vector_index", "cross_project", "embedder_device"}

    db.config.storage.options["vector_index"] = "vec0"
    assert db.retrieval_signature() != base
    db.config.storage.options["vector_index"] = "exact"
    db.close()

    # A different access policy is a different config file, so compare the
    # signatures two VectorDB instances would actually build.
    db2 = VectorDB(str(tmp_path / "sig2.db"))
    strict = dataclasses.replace(db2.config, cross_project={"allow": ["a", "b"]})
    db2.config = strict
    assert db2.retrieval_signature() != base
    db2.close()


def _write_cfg(path, db_path, **overrides):
    cfg = {
        "version": 2,
        "storage": {"provider": "sqlite",
                    "options": {"path": str(db_path), "vector_index": "exact"}},
        "retrieval": {"mode": "lexical"},
        "embedding": {"provider": "none"},
        "rerank": {"provider": "none"},
        "access": {"cross_project": {}},
        "auto_sync_paths": [],
        "trace": {"wait_patterns": None, "wait_patterns_extend": None},
    }
    cfg.update(overrides)
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


def test_same_db_different_config_does_not_share_cache(tmp_path, monkeypatch):
    """Two configs on one DB file must not read each other's cached results."""
    shared = tmp_path / "shared.db"
    src = _source_tree(str(tmp_path), n=4)

    a = _write_cfg(tmp_path / "a.json", shared)
    monkeypatch.setenv("AI_DB_CONFIG", a)
    db_a = VectorDB(str(shared))
    db_a.sync(src, project="p", verbose=False)
    assert db_a.query("f0", top_k=3, project="p")
    assert db_a.last_cache_hit is False
    db_a.query("f0", top_k=3, project="p")
    assert db_a.last_cache_hit is True, "same config must hit its own cache"
    db_a.close()

    # A second config differing only in cross-project policy.
    b = _write_cfg(tmp_path / "b.json", shared, access={"cross_project": {"allow": ["other"]}})
    monkeypatch.setenv("AI_DB_CONFIG", b)
    db_b = VectorDB(str(shared))
    db_b.query("f0", top_k=3, project="p")
    assert db_b.last_cache_hit is False, "a different access policy must not reuse the cache"
    db_b.close()


def test_same_config_across_processes_hits_cache(tmp_path, monkeypatch):
    """Isolation must not cost the cache its actual purpose."""
    shared = tmp_path / "shared2.db"
    src = _source_tree(str(tmp_path), n=4)
    a = _write_cfg(tmp_path / "c.json", shared)

    monkeypatch.setenv("AI_DB_CONFIG", a)
    db1 = VectorDB(str(shared))
    db1.sync(src, project="p", verbose=False)
    db1.query("f1", top_k=3, project="p")
    assert db1.last_cache_hit is False
    db1.close()

    monkeypatch.setenv("AI_DB_CONFIG", a)
    db2 = VectorDB(str(shared))
    db2.query("f1", top_k=3, project="p")
    assert db2.last_cache_hit is True
    db2.close()


def test_sync_bumps_generation_and_invalidates(tmp_path, lexical_config, monkeypatch):
    monkeypatch.setenv("AI_DB_CONFIG", lexical_config)
    root = tmp_path / "gen"
    root.mkdir()
    db = VectorDB(str(tmp_path / "g.db"))
    db.sync(str(root), project="p", verbose=False)
    gen_before = db.backend.get_index_generation()

    with open(root / "new.py", "w", encoding="utf-8") as fh:
        fh.write("def fresh():\n    return 1\n")
    db.sync(str(root), project="p", verbose=False)
    assert db.backend.get_index_generation() > gen_before

    db.query("fresh", top_k=3, project="p")
    assert db.last_cache_hit is False
    db.close()


# ==============================================================================
# languages remain available (14 depends on the 10.1 repairs)
# ==============================================================================

@pytest.mark.parametrize("filename,code,expected", [
    ("a.go", 'package p\n\nfunc Run() {}\n', {"Run"}),
    ("a.rs", 'fn run() {}\n', {"run"}),
    ("a.cpp", 'class C { void go() {} };\n', {"C"}),
    ("a.java", 'class C { void go() {} }\n', {"C"}),
])
def test_diff_seeds_span_non_python_languages(filename, code, expected):
    """A polyglot change must still produce seedable symbols."""
    symbols, _, errors = extract_graph(filename, code)
    assert errors == []
    assert expected <= {s["name"] for s in symbols}
