import json
import time
import sqlite3
from typing import Any, Optional, Dict, List

def get_session_state(conn: sqlite3.Connection, key: str) -> Optional[Any]:
    try:
        cur = conn.cursor()
        cur.execute("SELECT value_json FROM session_state WHERE key = ?", (key,))
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row["value_json"])
            except Exception:
                return None
    except Exception:
        return None
    return None

def set_session_state(conn: sqlite3.Connection, key: str, value: Any):
    try:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS session_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated REAL NOT NULL
            )"""
        )
        cur.execute(
            """INSERT OR REPLACE INTO session_state (key, value_json, updated)
               VALUES (?, ?, ?)""",
            (key, json.dumps(value), time.time())
        )
        conn.commit()
    except Exception:
        pass


def get_telemetry_state(conn: sqlite3.Connection, key: str) -> Optional[Any]:
    return get_session_state(conn, key)


def set_telemetry_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    set_session_state(conn, key, value)


def get_telemetry_table_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not conn:
        return counts
    try:
        cur = conn.cursor()
        for tbl in ["files", "chunks", "symbols", "skills", "syntax_errors", "contexts"]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                row = cur.fetchone()
                counts[tbl] = int(row[0]) if row else 0
            except Exception:
                counts[tbl] = 0
    except Exception:
        pass
    return counts


def get_telemetry_weak_points(conn: sqlite3.Connection) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "syntax_error_density_pct": 0.0,
        "complexity_hotspots": [],
        "unindexed_or_stale_files": [],
    }
    if not conn:
        return result
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM files")
        row = cur.fetchone()
        total_files = int(row[0]) if row and row[0] else 0

        cur.execute("SELECT COUNT(DISTINCT filepath) FROM syntax_errors")
        row_err = cur.fetchone()
        files_with_errors = int(row_err[0]) if row_err and row_err[0] else 0

        if total_files > 0:
            result["syntax_error_density_pct"] = round((files_with_errors / total_files) * 100.0, 2)

        cur.execute(
            """
            SELECT f.filepath, COUNT(s.id) as sym_count
            FROM files f
            JOIN symbols s ON f.filepath = s.filepath
            GROUP BY f.filepath
            ORDER BY sym_count DESC
            LIMIT 10
            """
        )
        hotspots = []
        for r in cur.fetchall():
            hotspots.append({
                "filepath": r[0],
                "symbols_count": int(r[1]),
            })
        result["complexity_hotspots"] = hotspots

        cur.execute(
            """
            SELECT filepath FROM files
            WHERE mtime IS NULL OR mtime = 0
            LIMIT 10
            """
        )
        result["unindexed_or_stale_files"] = [r[0] for r in cur.fetchall()]
    except Exception:
        pass
    return result
