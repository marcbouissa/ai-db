"""Raw command outputs kept for re-filtering (``ai-db filter --id``).

Outputs can contain secrets (tokens in logs, env dumps), so files are created
owner-only (0600) in the user's cache dir, and only the newest OUTPUT_STORE_KEEP are kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from ai_db.constants import OUTPUT_STORE_DIR, OUTPUT_STORE_KEEP
from ai_db.errors import AiDbStorageError


def _paths(output_id: str, store_dir: str) -> tuple[str, str]:
    if not output_id.isalnum():
        raise AiDbStorageError(f"invalid output id {output_id!r}")
    return (os.path.join(store_dir, f"{output_id}.log"),
            os.path.join(store_dir, f"{output_id}.json"))


def _write_private(path: str, data: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(data)


def save(text: str, meta: dict[str, Any], store_dir: str = OUTPUT_STORE_DIR) -> str:
    """Store ``text``; returns its id."""
    os.makedirs(store_dir, mode=0o700, exist_ok=True)
    output_id = hashlib.sha1(f"{time.time_ns()}{text[:4096]}".encode()).hexdigest()[:10]
    log_path, meta_path = _paths(output_id, store_dir)
    _write_private(log_path, text)
    _write_private(meta_path, json.dumps({**meta, "saved_at": time.time()}))
    _prune(store_dir)
    return output_id


def load(output_id: str, store_dir: str = OUTPUT_STORE_DIR) -> tuple[str, dict[str, Any]]:
    log_path, meta_path = _paths(output_id, store_dir)
    if not os.path.exists(log_path):
        raise AiDbStorageError(
            f"no stored output '{output_id}' (only the last {OUTPUT_STORE_KEEP} are kept)")
    with open(log_path, encoding="utf-8") as f:
        text = f.read()
    meta: dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return text, meta


def _prune(store_dir: str) -> None:
    logs = sorted((e for e in os.scandir(store_dir) if e.name.endswith(".log")),
                  key=lambda e: e.stat().st_mtime, reverse=True)
    for entry in logs[OUTPUT_STORE_KEEP:]:
        os.remove(entry.path)
        meta = entry.path[:-4] + ".json"
        if os.path.exists(meta):
            os.remove(meta)
