"""Raw-connection helpers for session state and telemetry (expects an initialized schema)."""

import json
import sqlite3
import time
from typing import Any

from ai_db.errors import AiDbStorageError


def get_session_state(conn: sqlite3.Connection, key: str) -> Any | None:
    cur = conn.cursor()
    cur.execute("SELECT value_json FROM session_state WHERE key = ?", (key,))
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError as exc:
        raise AiDbStorageError(f"corrupt JSON in session_state[{key!r}]: {exc}") from exc


def set_session_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO session_state (key, value_json, updated) VALUES (?, ?, ?)",
        (key, json.dumps(value), time.time()),
    )
    conn.commit()


def get_telemetry_state(conn: sqlite3.Connection, key: str) -> Any | None:
    return get_session_state(conn, key)


def set_telemetry_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    set_session_state(conn, key, value)


def get_telemetry_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    counts: dict[str, int] = {}
    for tbl in ["files", "chunks", "symbols", "skills", "syntax_errors", "contexts"]:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        counts[tbl] = int(cur.fetchone()[0])
    return counts


def get_telemetry_weak_points(conn: sqlite3.Connection) -> dict[str, Any]:
    cur = conn.cursor()
    total_files = int(cur.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    files_with_errors = int(cur.execute("SELECT COUNT(DISTINCT filepath) FROM syntax_errors").fetchone()[0])
    density = round(files_with_errors / total_files * 100.0, 2) if total_files else 0.0
    hotspots = [
        {"filepath": r[0], "symbols_count": int(r[1])}
        for r in cur.execute(
            """SELECT f.filepath, COUNT(s.id) FROM files f
               JOIN symbols s ON f.filepath = s.filepath
               GROUP BY f.filepath ORDER BY COUNT(s.id) DESC LIMIT 10"""
        ).fetchall()
    ]
    stale = [r[0] for r in cur.execute(
        "SELECT filepath FROM files WHERE last_modified IS NULL OR last_modified = 0 LIMIT 10"
    ).fetchall()]
    return {
        "syntax_error_density_pct": density,
        "complexity_hotspots": hotspots,
        "unindexed_or_stale_files": stale,
    }
