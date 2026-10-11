"""Event-driven file watching (watchfiles): only the paths that changed are re-indexed.

Used by ``ai-db watch`` (one directory, blocking) and by the daemon's
``daemon.watch_sync_paths`` (every ``auto_sync_paths`` root, asyncio).
"""

import os
from collections.abc import AsyncIterator, Iterable
from typing import Any


def group_by_root(roots: Iterable[str], paths: Iterable[str]) -> dict[str, list[str]]:
    """Changed ``paths`` grouped under the watched root that contains them (the deepest
    one when roots nest); paths outside every root are dropped."""
    ordered = sorted((os.path.abspath(r) for r in roots), key=len, reverse=True)
    grouped: dict[str, list[str]] = {}
    for path in paths:
        path = os.path.abspath(path)
        for root in ordered:
            if path == root or path.startswith(root + os.sep):
                grouped.setdefault(root, []).append(path)
                break
    return {root: sorted(set(ps)) for root, ps in grouped.items()}


async def awatch_roots(roots: Iterable[str], debounce_ms: int) -> AsyncIterator[dict[str, list[str]]]:
    """Batches of changed paths, ``{root: [paths]}``, until cancelled."""
    from watchfiles import awatch

    watched = [os.path.abspath(r) for r in roots]
    async for changes in awatch(*watched, debounce=debounce_ms):
        batch = group_by_root(watched, (p for _, p in changes))
        if batch:
            yield batch


def run_watch(db_cls: Any, db_path: str | None, target_dir: str, debounce_ms: int = 300,
              stop_event: Any = None) -> None:
    from watchfiles import watch

    target_dir = os.path.abspath(target_dir)
    db = db_cls(db_path)
    try:
        print(f"[ai-db watch] Monitoring '{target_dir}' -> DB: '{db.db_path}'")
        db.sync(target_dir)
        for changes in watch(target_dir, debounce=debounce_ms, stop_event=stop_event,
                             raise_interrupt=False):
            paths = group_by_root([target_dir], (p for _, p in changes)).get(target_dir, [])
            res = db.sync_paths(target_dir, paths)
            if res["added"] or res["updated"] or res["pruned"]:
                print(f"[ai-db watch] +{res['added']} ~{res['updated']} -{res['pruned']}")
    finally:
        db.close()
        print("[ai-db watch] Stopped.")
