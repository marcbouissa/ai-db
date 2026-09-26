"""`.aidbignore` must actually match.

A trailing slash in a gitignore-style pattern means "this directory and
everything under it". The compiler escaped it into a literal `/` that then had
to be followed by end-of-string or another slash, so `build/`, `node_modules/`
and `dist/` matched **nothing at all** -- silently, which is the worst way for an
ignore rule to fail.

This was not hypothetical: `.aidbignore` with `eval/results/` had no effect, so
`eval/results/token_budget.json` (63 KB of generated JSON containing the words
"latency", "percentiles" and "telemetry") stayed in the index and competed with
the source the golden queries are about.
"""
from __future__ import annotations

import pytest

from ai_db.ignorer import AidbIgnore


def ignorer(*patterns: str) -> AidbIgnore:
    return AidbIgnore("/tmp/project", list(patterns))


class TestDirectoryPatterns:
    @pytest.mark.parametrize("pattern", ["build/", "node_modules/", "dist/", "eval/results/"])
    def test_trailing_slash_ignores_the_directory(self, pattern):
        assert ignorer(pattern).should_ignore(f"{pattern}file.js") is True

    @pytest.mark.parametrize("pattern", ["build", "node_modules", "dist"])
    def test_without_slash_behaves_the_same(self, pattern):
        assert ignorer(pattern).should_ignore(f"{pattern}/deep/file.js") is True

    def test_trailing_and_bare_patterns_agree(self):
        assert ignorer("build/").should_ignore("build/a.js") == \
               ignorer("build").should_ignore("build/a.js") is True

    def test_bare_slash_ignores_nothing(self):
        assert ignorer("/").should_ignore("anything.py") is False


class TestGlobPatterns:
    @pytest.mark.parametrize("pattern,path,expected", [
        ("*.pyc", "a/b.pyc", True),
        ("*.pyc", "a/b.py", False),
        ("src/*.ts", "src/a.ts", True),
        ("src/*.ts", "src/deep/a.ts", False),   # * stays within one component
        ("src/**", "src/deep/a.ts", True),
        ("**/generated", "a/b/generated", True),
        ("build/", "buildings/a.js", False),    # prefix must be a whole segment
    ])
    def test_globs(self, pattern, path, expected):
        assert ignorer(pattern).should_ignore(path) is expected


class TestAidbIgnoreFile:
    def test_reads_a_real_file(self, tmp_path):
        (tmp_path / ".aidbignore").write_text(
            "# a comment\n\nbuild/\n*.log\n", encoding="utf-8"
        )
        ig = AidbIgnore(str(tmp_path))
        assert ig.should_ignore("build/out.js") is True
        assert ig.should_ignore("a/debug.log") is True
        assert ig.should_ignore("a/main.py") is False

    def test_comment_and_blank_lines_are_skipped(self, tmp_path):
        (tmp_path / ".aidbignore").write_text("# build/\n\n", encoding="utf-8")
        assert AidbIgnore(str(tmp_path)).should_ignore("build/out.js") is False

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert AidbIgnore(str(tmp_path)).has_rules is False

    def test_extra_patterns_and_file_patterns_combine(self, tmp_path):
        (tmp_path / ".aidbignore").write_text("vendor/\n", encoding="utf-8")
        ig = AidbIgnore(str(tmp_path), ["secrets/"])
        assert ig.should_ignore("vendor/x.js") is True
        assert ig.should_ignore("secrets/x.json") is True
        assert ig.should_ignore("src/x.py") is False
