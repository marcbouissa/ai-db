"""
ai_db.ignorer
Parses .aidbignore files (gitignore-style) and tests paths against patterns.
A project can place a .aidbignore file at its root to exclude files from indexing.
"""
import os
import re


class AidbIgnore:
    """Parses .aidbignore files and tests whether a relative path should be ignored."""

    def __init__(self, root_dir: str, extra_patterns: list[str] | None = None):
        self.root = os.path.abspath(root_dir)
        self.patterns: list[re.Pattern] = []
        self._load(os.path.join(self.root, ".aidbignore"))
        if extra_patterns:
            for pattern in extra_patterns:
                if pattern and not pattern.startswith("#"):
                    self.patterns.append(self._compile(pattern))

    def _load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                self.patterns.append(self._compile(line))

    @staticmethod
    def _compile(pattern: str) -> re.Pattern:
        """Convert a gitignore-style glob pattern to a compiled regex."""
        # A trailing slash means "this directory and everything under it", which
        # is what gitignore means by `build/`. It has to be stripped before
        # escaping, because re.escape turns it into a literal "/" that then has
        # to be followed by end-of-string or another slash -- so `build/` and
        # `node_modules/` matched nothing at all, silently.
        stripped = pattern.rstrip("/")
        if not stripped:
            return re.compile(r"(?!)")  # "/" alone ignores nothing
        # Escape special regex chars first, then translate globs
        p = re.escape(stripped)
        # ** matches any depth of path segments
        p = p.replace(r"\*\*", ".*")
        # * matches within a single path component
        p = p.replace(r"\*", "[^/]*")
        # ? matches a single character (not a slash)
        p = p.replace(r"\?", "[^/]")
        return re.compile(f"(^|/){p}($|/)", re.IGNORECASE)

    def should_ignore(self, rel_path: str) -> bool:
        """Returns True if the given relative path matches any ignore pattern."""
        normalized = rel_path.replace(os.sep, "/")
        return any(pat.search(normalized) for pat in self.patterns)

    @property
    def has_rules(self) -> bool:
        return bool(self.patterns)