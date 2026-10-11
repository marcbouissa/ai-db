"""Session state, table counts and telemetry weak points.

A mixin of :class:`ai_db.storage.sqlite_backend.SQLiteBackend`: every repository shares
the backend's connection and ``transaction()`` (see ``_base.SQLiteRepository``)."""

from __future__ import annotations

import json
import time
from typing import Any

from ai_db.storage.sqlite_repos._base import (
    SQLiteRepository,
    _loads,
    _writes,
)


class StateRepository(SQLiteRepository):
    def get_state(self, key: str) -> Any | None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("SELECT value_json FROM session_state WHERE key = ?", (key,))
        row = cur.fetchone()
        if row:
            return _loads(row["value_json"], "session_state.value_json")
        return None

    @_writes
    def set_state(self, key: str, value: Any) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO session_state (key, value_json, updated)
            VALUES (?, ?, ?)
            """,
            (key, json.dumps(value), time.time())
        )

    def get_table_counts(self) -> dict[str, int]:
        self._check_closed()
        cur = self.conn.cursor()
        counts: dict[str, int] = {}
        for tbl in ["files", "chunks", "symbols", "skills", "syntax_errors", "contexts"]:
            cur.execute(f"SELECT COUNT(*) as c FROM {tbl}")
            counts[tbl] = int(cur.fetchone()["c"])
        return counts

    def get_weak_points(self) -> dict[str, Any]:
        self._check_closed()
        cur = self.conn.cursor()
        total_files = int(cur.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        files_with_errors = int(cur.execute(
            "SELECT COUNT(DISTINCT filepath) FROM syntax_errors").fetchone()[0])
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
