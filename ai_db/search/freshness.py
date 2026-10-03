"""Staleness detection for read paths.

Every result carries line numbers and bodies from the last index. When a file
changed on disk since then, the answer is wrong in ways an agent cannot see.
``resolve`` compares each result file with disk (mtime first, sha256 to
confirm), re-syncs up to ``STALE_RESYNC_MAX_FILES`` stale files and re-runs the
query, and above that bound flags the affected results instead of returning
stale data silently.
"""

import os
from collections.abc import Callable
from typing import Any

from ai_db.constants import STALE_RESYNC_MAX_FILES
from ai_db.utils import compute_sha256

_PATH_KEYS = ("abs_path", "file", "filepath", "path")


def collect_paths(result: Any) -> set[str]:
    """Absolute paths of every indexed-file reference found in a structured result."""
    found: set[str] = set()
    stack = [result]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in _PATH_KEYS:
                val = node.get(key)
                if isinstance(val, str) and val:
                    found.add(os.path.abspath(os.path.expanduser(val)))
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (dict, list)))
    return found


def detect_stale(db: Any, paths: set[str]) -> dict[str, str]:
    """Return {path: project} for indexed files whose disk state differs from the index.

    Paths unknown to the index are ignored. A deleted file counts as stale.
    """
    stale: dict[str, str] = {}
    for path in sorted(paths):
        rec = db.backend.get_file(path)
        if rec is None:
            continue
        try:
            mtime = os.stat(path).st_mtime
        except FileNotFoundError:
            stale[path] = rec.project
            continue
        if mtime <= rec.last_modified:
            continue
        # mtime moved: a touch alone is not a change, so confirm by content.
        if compute_sha256(path) != rec.sha256:
            stale[path] = rec.project
    return stale


def _sync_root(path: str) -> str:
    cur = os.path.dirname(path)
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.dirname(path)
        cur = parent


def _resync(db: Any, stale: dict[str, str]) -> None:
    groups: dict[tuple[str, str], list[str]] = {}
    for path, project in stale.items():
        groups.setdefault((_sync_root(path), project), []).append(path)
    for (root, project), paths in groups.items():
        db.sync_paths(root, paths, project=project)


def resolve(db: Any, run: Callable[[], Any]) -> tuple[Any, list[str]]:
    """Run ``run()``, then make sure its result is fresh. Returns (result, stale_files).

    ``stale_files`` is empty when everything was fresh or was re-synced.
    """
    result = run()
    stale = detect_stale(db, collect_paths(result))
    if not stale:
        return result, []
    if len(stale) <= STALE_RESYNC_MAX_FILES:
        _resync(db, stale)
        result = run()
        stale = detect_stale(db, collect_paths(result))
    return result, sorted(stale)


def annotate(result: Any, stale_files: list[str]) -> Any:
    """Flag stale results: dicts get ``stale_files`` (and per-item ``stale``), lists get
    per-item ``stale``, strings get a leading ``# stale:`` line."""
    if not stale_files:
        return result
    stale = set(stale_files)

    def mark(node: Any) -> None:
        if isinstance(node, dict):
            if collect_paths({k: v for k, v in node.items() if isinstance(v, str)}) & stale:
                node["stale"] = True
            for v in node.values():
                if isinstance(v, (dict, list)):
                    mark(v)
        elif isinstance(node, list):
            for v in node:
                mark(v)

    if isinstance(result, str):
        return f"# stale: file changed since indexing, results may be wrong: {', '.join(stale_files)}\n{result}"
    mark(result)
    if isinstance(result, dict):
        result["stale_files"] = list(stale_files)
    return result
