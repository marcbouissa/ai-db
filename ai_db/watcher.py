"""Event-driven file watcher: re-indexes only the paths that changed (watchfiles)."""

import os
from typing import Any


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
            paths = [p for _, p in changes]
            res = db.sync_paths(target_dir, paths)
            if res["added"] or res["updated"] or res["pruned"]:
                print(f"[ai-db watch] +{res['added']} ~{res['updated']} -{res['pruned']}")
    finally:
        db.close()
        print("[ai-db watch] Stopped.")
