"""The one diff engine: git hunks (``git diff --unified=0``) and the index comparison.

- ``changed_lines(since, root)``: every changed line per file of a repository, for
  ``investigate --mode diff``. Untracked files count as entirely added; a pure deletion
  (``+N,0``) seeds line N, the line the deleted block followed, so its enclosing
  definition is still reviewed.
- ``file_diff(path, since, ...)``: ``added``/``removed``/``changed`` spans of one file, for
  ``analyze --since`` and ``diff``. A git ``since`` diffs the working tree against that
  ref (a file new since then is entirely added); ``"last"``/``"db"`` compares with what
  the index holds, at definition granularity.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from ai_db.constants import GIT_TIMEOUT_S
from ai_db.errors import AiDbConfigError

INDEX_SINCE = ("last", "db")  # `since` values meaning "the indexed version"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    """One ``--unified=0`` hunk: ``old_count`` lines at ``old_start`` became ``new_count``
    lines at ``new_start`` (a count of 0 means a pure insertion or deletion)."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int


def _git(args: list[str], cwd: str, what: str) -> str:
    try:
        res = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=GIT_TIMEOUT_S, cwd=cwd, check=False)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise AiDbConfigError(f"{what}: {err}") from err
    if res.returncode != 0:
        raise AiDbConfigError(f"{what}: {res.stderr.strip() or 'git failed'}")
    return res.stdout


def repo_root(path: str) -> str:
    """Top level of the git repository containing ``path`` (a file or directory)."""
    directory = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    return _git(["rev-parse", "--show-toplevel"], directory,
                f"{path} needs to be in a git repository").strip()


def git_hunks(since: str, root: str, paths: Collection[str] | None = None) -> dict[str, list[Hunk]]:
    """``{absolute path: hunks}`` of ``git diff <since>`` (working tree vs ``since``) in the
    repository at ``root``, optionally limited to ``paths``. Untracked files are not listed."""
    out = _git(["diff", "--unified=0", "--no-color", "--no-ext-diff", since, "--", *(paths or [])],
               root, f"git diff against {since!r} failed")
    hunks: dict[str, list[Hunk]] = {}
    current: str | None = None
    for line in out.splitlines():
        if line.startswith("+++ "):
            # "+++ b/path" for a change, "+++ /dev/null" for a deletion
            target = line[4:].strip()
            if target == "/dev/null":
                current = None
            else:
                current = os.path.join(root, target.removeprefix("b/"))
            continue
        if current is None:
            continue
        m = _HUNK.match(line)
        if m is not None:
            hunks.setdefault(current, []).append(Hunk(
                int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)))
    return hunks


def untracked_files(root: str, paths: Collection[str] | None = None) -> list[str]:
    """Absolute paths of the untracked, non-ignored files under ``root`` (or ``paths``)."""
    out = _git(["ls-files", "--others", "--exclude-standard", "-z", "--", *(paths or [])],
               root, "git ls-files failed")
    return [os.path.join(root, rel) for rel in out.split("\0") if rel]


def _line_count(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except (OSError, UnicodeDecodeError):
        return None  # unreadable or binary: nothing line-based to report


def changed_lines(since: str, root: str) -> dict[str, list[int]]:
    """``{absolute path: changed line numbers}`` for the repository containing ``root``.

    Raises AiDbConfigError when git cannot answer: a diff-mode investigation with no
    diff would silently return an empty pack, which reads as "nothing to review" rather
    than "I could not look".
    """
    top = _git(["rev-parse", "--show-toplevel"], root,
               f"mode 'diff' needs a git repository ({root} is not in one)").strip()
    spans: dict[str, list[int]] = {}
    for path, hunks in git_hunks(since, top).items():
        lines = spans.setdefault(path, [])
        for h in hunks:
            if h.new_count:
                lines.extend(range(h.new_start, h.new_start + h.new_count))
            else:  # pure deletion after line new_start: review what surrounds it
                lines.append(max(1, h.new_start))
    for path in untracked_files(top):
        count = _line_count(path)
        if count:
            spans[path] = list(range(1, count + 1))
    return spans


def file_diff(filepath: str, since: str | None, backend: Any = None,
              languages: Collection[str] | None = None,
              current_content: str | None = None) -> dict[str, list[list[int]]]:
    """``{"added", "removed", "changed"}`` 1-based inclusive spans of one file since ``since``.

    Spans of ``added``/``changed`` are in the current file, ``removed`` in the old one.
    ``since`` in ``INDEX_SINCE`` (or None) compares with the index (``backend`` needed).
    """
    path = os.path.abspath(filepath)
    if since is None or since in INDEX_SINCE:
        if backend is None:
            raise ValueError("comparing with the index needs the storage backend")
        if current_content is None:
            with open(path, encoding="utf-8", errors="replace") as fh:
                current_content = fh.read()
        return index_diff(path, current_content, backend, languages)
    top = repo_root(path)
    if path in untracked_files(top, [path]):
        count = _line_count(path) or 0
        return {"added": [[1, count]] if count else [], "removed": [], "changed": []}
    added: list[list[int]] = []
    removed: list[list[int]] = []
    changed: list[list[int]] = []
    for h in git_hunks(since, top, [path]).get(path, []):
        if h.old_count == 0:
            added.append([h.new_start, h.new_start + h.new_count - 1])
        elif h.new_count == 0:
            removed.append([h.old_start, h.old_start + h.old_count - 1])
        else:
            changed.append([h.new_start, h.new_start + h.new_count - 1])
    return {"added": added, "removed": removed, "changed": changed}


def index_diff(filepath: str, current_content: str, backend: Any,
               languages: Collection[str] | None = None) -> dict[str, list[list[int]]]:
    """Spans that changed since the last index, at definition (chunk) granularity.

    The index does not keep a file's raw text, and its chunks are compacted and skip
    the lines between definitions, so joining them never reproduces the file: that
    reported changes in files that had none. Instead: an unchanged hash means no
    change; otherwise the current text is chunked the way the indexer chunks it and
    chunks are matched by content hash. A new chunk whose name was indexed is
    ``changed``, an unknown name is ``added``; an indexed chunk with no match is
    ``removed`` (its old lines).
    """
    from ai_db.parser.chunker import chunk_file
    from ai_db.parser.languages import SUPPORTED_LANGUAGES
    from ai_db.utils import compute_sha256

    record = backend.get_file(filepath)
    old = backend.get_chunks_for_file(filepath)
    lines = current_content.splitlines()
    if record is None or not old:
        return {"added": [[1, len(lines)]] if lines else [], "removed": [], "changed": []}
    if os.path.isfile(filepath) and record.sha256 == compute_sha256(filepath):
        return {"added": [], "removed": [], "changed": []}
    new = chunk_file(filepath, current_content,
                     SUPPORTED_LANGUAGES if languages is None else languages)
    old_hashes = {c.content_hash for c in old}
    new_hashes = {c["content_hash"] for c in new}
    old_names = {c.name for c in old}
    new_names = {c["name"] for c in new}
    added: list[list[int]] = []
    changed: list[list[int]] = []
    for c in new:
        if c["content_hash"] in old_hashes:
            continue
        (changed if c["name"] in old_names else added).append([c["start_line"], c["end_line"]])
    removed = [[c.start_line, c.end_line] for c in old
               if c.content_hash not in new_hashes and c.name not in new_names]
    return {"added": added, "removed": removed, "changed": changed}
