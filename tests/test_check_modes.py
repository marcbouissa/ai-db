"""`check`/`lint` validate the file or the index -- two questions, not one.

`check <path>` used to re-index the file it was checking: `_handle_check` called
`prune_file` then `_index_file`, then read the syntax errors back out of the
database it had just rewritten. Three consequences, all measured:

- A read-only command mutated the index.
- It cost a full chunk-and-embed pass (216 `count_tokens` calls, loading
  tiktoken) to answer a question `ast.parse` answers in under a millisecond:
  583 ms against a 229 ms floor for a command that does nothing.
- It reported the *index's* verdict while appearing to check the file, so a
  stale index silently answered for the file on disk.

Splitting the two also exposed a bug in the tree-sitter error collector, which
is why `ts_graph` cases are covered here too.
"""

from __future__ import annotations

import ast
import hashlib
import os
import time

import pytest

from ai_db.errors import AiDbConfigError
from ai_db.parser.linters import parse_source

BROKEN = "def broken(:\n    pass\n"
STRAY = "x = 1\ny = )\n"
CLEAN = "def f():\n    return 1\n"


def _write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------- tree-sitter
class TestSyntaxErrorDetection:
    """A missing node carries the expected token's type, not the string 'MISSING'.

    `collect_errors` compared `node.type` against the literals "ERROR" and
    "MISSING". Tree-sitter sets `is_missing` and types the node as the token it
    expected -- `def broken(:` yields a node typed ")" -- so the comparison never
    matched and these errors were never reported. Verified against `ast.parse`,
    which is the behaviour Python programmers expect.
    """

    @pytest.mark.parametrize("source", [BROKEN, STRAY])
    def test_detects_what_ast_detects(self, tmp_path, source):
        path = _write(tmp_path, "sample.py", source)
        with pytest.raises(SyntaxError):
            ast.parse(source)  # confirms the fixture really is invalid
        assert parse_source(path, source)[2] is not None

    def test_clean_file_reports_nothing(self, tmp_path):
        path = _write(tmp_path, "clean.py", CLEAN)
        assert parse_source(path, CLEAN)[2] is None

    def test_error_message_names_the_problem(self, tmp_path):
        path = _write(tmp_path, "broken.py", BROKEN)
        line, col, message = parse_source(path, BROKEN)[2]
        assert (line, col) == (1, 12)
        assert ")" in message and "syntax" in message

    def test_agrees_with_ast_across_cases(self, tmp_path):
        """The single parse path must not disagree with the language's own parser."""
        cases = {"a.py": BROKEN, "b.py": STRAY, "c.py": CLEAN,
                 "d.py": "class A:\n    def m(self):\n        return {1:2}\n"}
        for name, source in cases.items():
            path = _write(tmp_path, name, source)
            reported = parse_source(path, source)[2] is not None
            try:
                ast.parse(source)
                expected = False
            except SyntaxError:
                expected = True
            assert reported == expected, f"{name}: tree-sitter={reported} ast={expected}"


# ------------------------------------------------------------------- parities
class TestSingleParsePath:
    """The indexer and `check` must reach the same verdict, or `check` lies."""

    def test_indexer_uses_parse_source(self):
        import inspect

        from ai_db.search import indexer

        src = inspect.getsource(indexer.parse_file)
        assert "parse_source(" in src
        # The old inline branch duplicated the decision and could drift.
        assert "extract_graph(" not in src
        assert "get_linter()" not in src

    def test_parse_source_is_pure(self, tmp_path):
        """No storage access: that is what makes file mode read-only."""
        path = _write(tmp_path, "x.py", CLEAN)
        before = sorted(os.listdir(tmp_path))
        parse_source(path, CLEAN)
        assert sorted(os.listdir(tmp_path)) == before


class TestToolCategoryIsReachable:
    """Every tool sets a category, and nothing dropped it.

    `category` was passed to every one of the 21 registrations and then omitted
    from `to_mcp_dict()`, which is what `list_tools()` returns and what MCP
    clients receive. The grouping was write-only: `list_tools()` reported None
    for all 21 tools.
    """

    def test_every_tool_exposes_a_category(self, tmp_path):
        from ai_db.dispatcher import ServiceDispatcher

        d = ServiceDispatcher(db_path=str(tmp_path / "idx.db"))
        try:
            tools = d.list_tools()
            assert tools
            missing = [t["name"] for t in tools if not t.get("category")]
            assert not missing, f"tools with no category: {missing}"
        finally:
            d.close()

    def test_categories_are_a_small_curated_set(self, tmp_path):
        from ai_db.dispatcher import ServiceDispatcher

        d = ServiceDispatcher(db_path=str(tmp_path / "idx.db"))
        try:
            cats = {t["category"] for t in d.list_tools()}
            # Catches a placeholder default leaking through for a new tool.
            assert len(cats) <= 12, cats
            assert "general" not in cats, "every tool should declare a real category"
        finally:
            d.close()

    def test_mcp_dict_includes_category(self):
        from ai_db.dispatcher import ToolDefinition

        tool = ToolDefinition(name="t", description="d",
                              parameters_schema={"type": "object"},
                              handler=lambda a: None, category="search")
        assert tool.to_mcp_dict()["category"] == "search"


# ---------------------------------------------------------------- the split
class TestCheckModes:
    @pytest.fixture()
    def dispatcher(self, tmp_path):
        from ai_db.dispatcher import ServiceDispatcher

        d = ServiceDispatcher(db_path=str(tmp_path / "idx.db"))
        yield d
        d.close()

    def test_file_mode_reports_a_broken_file(self, dispatcher, tmp_path):
        path = _write(tmp_path, "broken.py", BROKEN)
        errors = dispatcher.execute("check", {"path": path})
        assert len(errors) == 1
        assert errors[0]["line"] == 1 and errors[0]["col"] == 12
        assert errors[0]["source"] == "file"

    def test_file_mode_is_clean_for_valid_file(self, dispatcher, tmp_path):
        path = _write(tmp_path, "ok.py", CLEAN)
        assert dispatcher.execute("check", {"path": path}) == []

    def test_file_mode_does_not_touch_the_index(self, dispatcher, tmp_path):
        """The regression that motivated the split: a read command that wrote."""
        path = _write(tmp_path, "broken.py", BROKEN)
        # Force schema creation first: the very first execute() initialises the
        # database file, which would otherwise be mistaken for a write by check.
        dispatcher.execute("status", {})
        db_file = tmp_path / "idx.db"
        before = hashlib.sha256(db_file.read_bytes()).hexdigest()
        dispatcher.execute("check", {"path": path})
        assert hashlib.sha256(db_file.read_bytes()).hexdigest() == before

    def test_no_token_counting_in_file_mode(self, dispatcher, tmp_path, monkeypatch):
        """File mode must not chunk, or it pays tiktoken to answer a syntax question."""
        from ai_db.parser import chunker

        calls = []
        original = chunker.count_tokens

        def counting(text):
            calls.append(text)
            return original(text)

        monkeypatch.setattr(chunker, "count_tokens", counting)
        path = _write(tmp_path, "big.py", CLEAN * 200)
        dispatcher.execute("check", {"path": path})
        assert calls == []

    def test_file_mode_does_not_load_tiktoken(self, tmp_path):
        """End-to-end: the process must finish without importing tiktoken."""
        import subprocess
        import sys

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = _write(tmp_path, "big.py", CLEAN * 200)
        code = (
            "import sys;"
            "from ai_db.dispatcher import ServiceDispatcher;"
            f"d=ServiceDispatcher(db_path={str(tmp_path / 'i.db')!r});"
            f"d.execute('check', {{'path': {path!r}}});"
            "print('tiktoken' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=repo, check=True)
        assert out.stdout.strip() == "False", out.stdout

    def test_file_mode_does_not_index_the_file(self, dispatcher, tmp_path):
        """The broken file must not become a tracked file in the index."""
        path = _write(tmp_path, "broken.py", BROKEN)
        dispatcher.execute("check", {"path": path})
        assert dispatcher.execute("check", {"index": True}) == []

    def test_directory_mode_walks_recursively(self, dispatcher, tmp_path):
        nested = tmp_path / "pkg" / "deep"
        nested.mkdir(parents=True)
        _write(nested, "bad.py", STRAY)
        _write(tmp_path, "good.py", CLEAN)
        errors = dispatcher.execute("check", {"path": str(tmp_path)})
        assert len(errors) == 1
        assert errors[0]["abs_path"].endswith("bad.py")

    def test_index_mode_reads_stored_errors(self, dispatcher, tmp_path):
        db = dispatcher._get_db({})
        path = _write(tmp_path, "broken.py", BROKEN)
        with open(path, encoding="utf-8") as fh:
            db._index_file(path, hashlib.sha256(fh.read().encode()).hexdigest(),
                           project="global")
        errors = dispatcher.execute("check", {"index": True})
        assert len(errors) == 1
        assert errors[0]["line"] == 1

    def test_index_and_file_modes_agree(self, dispatcher, tmp_path):
        """Same file, both modes, same verdict -- or one of them is lying."""
        good = _write(tmp_path, "good.py", CLEAN)
        bad = _write(tmp_path, "bad.py", BROKEN)
        db = dispatcher._get_db({})
        for p in (good, bad):
            with open(p, encoding="utf-8") as fh:
                db._index_file(p, hashlib.sha256(fh.read().encode()).hexdigest(),
                               project="global")
        from_index = {e["abs_path"] for e in dispatcher.execute("check", {"index": True})}
        from_disk = {e["abs_path"] for e in
                     dispatcher.execute("check", {"path": str(tmp_path)})}
        assert from_index == from_disk == {bad}

    def test_index_with_path_is_rejected(self, dispatcher, tmp_path):
        """Not silently resolved one way or the other (rule 1: no fallbacks)."""
        path = _write(tmp_path, "x.py", CLEAN)
        with pytest.raises(AiDbConfigError, match="--index"):
            dispatcher.execute("check", {"path": path, "index": True})

    def test_directory_mode_uses_the_indexers_file_set(self, dispatcher, tmp_path):
        """`check .` must consider exactly the files `sync .` would index.

        Walking the tree by hand instead would try to parse binaries, vendored
        trees and anything .aidbignore excludes.
        """
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "ok.py").write_text(CLEAN, encoding="utf-8")
        (tmp_path / "notes.md").write_text("# not python\n", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00binary")
        (tmp_path / "ignored.py").write_text(BROKEN, encoding="utf-8")
        (tmp_path / ".aidbignore").write_text("ignored.py\n", encoding="utf-8")

        db = dispatcher._get_db({})
        expected = {os.path.abspath(p) for p in db.scan_directory(str(tmp_path))}
        assert expected  # sanity: the walk found something

        errors = dispatcher.execute("check", {"path": str(tmp_path)})
        # The ignored broken file must not be reported.
        assert errors == []

    def test_missing_path_is_rejected(self, dispatcher, tmp_path):
        with pytest.raises(AiDbConfigError, match="not a file or directory"):
            dispatcher.execute("check", {"path": str(tmp_path / "nope.py")})

    def test_bare_check_reads_the_index(self, dispatcher):
        """No path and no --index: the index is the only thing left to report."""
        assert dispatcher.execute("check", {}) == dispatcher.execute(
            "check", {"index": True})


# --------------------------------------------------------------------- perf
class TestCheckOverhead:
    """File mode stays at parity with a command that does nothing.

    The old path cost 583 ms against a 229 ms floor for `status`, the difference
    being a chunk-and-embed pass spent on a syntax question.
    """

    def test_file_mode_is_close_to_the_empty_command(self, tmp_path):
        import statistics
        import subprocess
        import sys

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = _make_config(tmp_path)
        path = _write(tmp_path, "big.py", CLEAN * 200)

        def median_ms(*args):
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                subprocess.run([sys.executable, "-m", "ai_db.cli", "--config", cfg, *args],
                               capture_output=True, cwd=repo, check=False)
                samples.append((time.perf_counter() - t0) * 1000)
            return statistics.median(samples)

        status = median_ms("status")
        check = median_ms("check", path)
        # Before the split this was ~583ms, i.e. ~350ms over the floor.
        assert check < status * 1.6, f"check {check:.0f}ms vs status {status:.0f}ms"


# ----------------------------------------------------------------------- CLI
class TestCheckCLI:
    def _run(self, cfg, *args):
        import subprocess
        import sys

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "--config", cfg, *args],
            capture_output=True, text=True, cwd=repo, check=False,
        )

    def test_index_flag_is_accepted(self, tmp_path):
        cfg = _make_config(tmp_path)
        result = self._run(cfg, "check", "--index")
        assert result.returncode == 0, result.stderr
        assert "No syntax errors" in result.stdout

    def test_lint_alias_supports_index_flag(self, tmp_path):
        cfg = _make_config(tmp_path)
        result = self._run(cfg, "lint", "--index")
        assert result.returncode == 0, result.stderr

    def test_file_mode_reports_a_broken_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        path = _write(tmp_path, "broken.py", BROKEN)
        result = self._run(cfg, "check", path)
        assert result.returncode == 1
        assert "SYNTAX_ERROR" in result.stdout

    def test_watch_without_path_is_rejected(self, tmp_path):
        """Watch validates files as they change; there is nothing to watch otherwise."""
        cfg = _make_config(tmp_path)
        result = self._run(cfg, "check", "--watch")
        assert result.returncode != 0
        assert "--watch needs a path" in (result.stderr + result.stdout)


def _make_config(tmp_path) -> str:
    """A real config, produced by `init` rather than hand-written.

    The CLI rejects a config missing any required section, so a partial dict
    tests nothing but the parser.
    """
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = str(tmp_path / "config.json")
    subprocess.run([sys.executable, "-m", "ai_db.cli", "--config", path, "init", "--force"],
                   capture_output=True, cwd=repo, check=True)
    return path

