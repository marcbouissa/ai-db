"""Query-result cache (index generation), query log, analysis refs, semantic cache.

A mixin of :class:`ai_db.storage.sqlite_backend.SQLiteBackend`: every repository shares
the backend's connection and ``transaction()`` (see ``_base.SQLiteRepository``)."""

from __future__ import annotations

import json
import time
import zlib
from typing import Any

from ai_db.constants import (
    QUERY_LOG_KEEP,
    ZLIB_LEVEL,
)
from ai_db.errors import AiDbStorageError
from ai_db.storage.models import (
    AnalysisRefRecord,
)
from ai_db.storage.sqlite_repos._base import (
    SQLiteRepository,
    _decompress,
    _loads,
    _writes,
)


class CacheRepository(SQLiteRepository):
    def get_index_generation(self) -> int:
        self._check_closed()
        return int(self.get_state("index_gen") or 0)

    @_writes
    def bump_index_generation(self) -> int:
        """Increment the generation and drop cache rows from older generations."""
        self._check_closed()
        gen = self.get_index_generation() + 1
        self.set_state("index_gen", gen)
        self.conn.execute("DELETE FROM query_cache WHERE index_gen != ?", (gen,))
        return gen

    def get_query_cache(self, cache_key: str, index_gen: int) -> Any | None:
        self._check_closed()
        row = self.conn.execute(
            "SELECT result_json FROM query_cache WHERE cache_key = ? AND index_gen = ?",
            (cache_key, index_gen)).fetchone()
        return json.loads(row["result_json"]) if row else None

    @_writes
    def set_query_cache(self, cache_key: str, index_gen: int, result: Any) -> None:
        self._check_closed()
        self.conn.execute(
            "INSERT OR REPLACE INTO query_cache (cache_key, index_gen, result_json, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (cache_key, index_gen, json.dumps(result, ensure_ascii=False), time.time()))

    @_writes
    def log_query(self, entry: dict[str, Any]) -> None:
        """Append one query-log row; keeps the newest QUERY_LOG_KEEP rows."""
        self._check_closed()
        cur = self.conn.execute(
            "INSERT INTO query_log (timestamp, tool, query, mode, total_ms, cache_hit, "
            "stages_json, providers_json, top_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry["timestamp"], entry["tool"], entry["query"], entry.get("mode"),
             float(entry["total_ms"]), int(bool(entry["cache_hit"])),
             json.dumps(entry.get("stages", {})), json.dumps(entry.get("providers", {})),
             json.dumps(entry.get("top", []))))
        self.conn.execute("DELETE FROM query_log WHERE id <= ?",
                          ((cur.lastrowid or 0) - QUERY_LOG_KEEP,))

    @_writes
    def clear_query_log(self) -> None:
        self._check_closed()
        self.conn.execute("DELETE FROM query_log")

    def get_query_log(self, min_total_ms: float = 0.0, limit: int = 50) -> list[dict[str, Any]]:
        self._check_closed()
        cur = self.conn.execute(
            "SELECT * FROM query_log WHERE total_ms >= ? ORDER BY id DESC LIMIT ?",
            (float(min_total_ms), int(limit)))
        return [{
            "timestamp": r["timestamp"], "tool": r["tool"], "query": r["query"], "mode": r["mode"],
            "total_ms": r["total_ms"], "cache_hit": bool(r["cache_hit"]),
            "stages": json.loads(r["stages_json"]), "providers": json.loads(r["providers_json"]),
            "top": json.loads(r["top_json"]),
        } for r in cur.fetchall()]

    @_writes
    def store_analysis_ref(self, ref: AnalysisRefRecord) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        zbody = zlib.compress(ref.body_text.encode("utf-8"), level=ZLIB_LEVEL)
        existing = cur.execute(
            "SELECT filepath, start_line, end_line, zbody FROM analysis_refs WHERE ref_id = ?",
            (ref.ref_id,)).fetchone()
        if existing is not None:
            same = (existing["filepath"] == ref.filepath and existing["start_line"] == ref.start_line
                    and existing["end_line"] == ref.end_line
                    and _decompress(existing["zbody"], f"analysis ref {ref.ref_id}") == ref.body_text)
            if not same:
                # A hash-prefix collision: overwriting would make `expand` return another body.
                raise AiDbStorageError(
                    f"analysis ref id collision: {ref.ref_id} already holds "
                    f"{existing['filepath']}:{existing['start_line']}-{existing['end_line']}")
            cur.execute("UPDATE analysis_refs SET name = ?, kind = ?, timestamp = ? WHERE ref_id = ?",
                        (ref.name, ref.kind, ref.timestamp, ref.ref_id))
            return
        cur.execute(
            """
            INSERT INTO analysis_refs (ref_id, filepath, name, start_line, end_line, kind, zbody, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ref.ref_id, ref.filepath, ref.name, ref.start_line, ref.end_line, ref.kind, zbody, ref.timestamp)
        )

    def get_analysis_ref(self, ref_id: str) -> AnalysisRefRecord | None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT ref_id, filepath, name, start_line, end_line, kind, zbody, timestamp FROM analysis_refs WHERE ref_id = ?",
            (ref_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        body = _decompress(row["zbody"], f"analysis ref {row['ref_id']}")
        return AnalysisRefRecord(
            ref_id=row["ref_id"],
            filepath=row["filepath"],
            name=row["name"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            kind=row["kind"],
            body_text=body,
            timestamp=row["timestamp"]
        )

    @_writes
    def evict_stale_analysis_refs(self, older_than_seconds: float) -> int:
        self._check_closed()
        cur = self.conn.cursor()
        cutoff = time.time() - older_than_seconds
        cur.execute("DELETE FROM analysis_refs WHERE timestamp < ?", (cutoff,))
        count = cur.rowcount
        return count

    def get_semantic_cache(self, cache_key: str, file_hash: str) -> dict[str, Any] | None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT result_json FROM semantic_cache WHERE cache_key = ? AND file_hash = ?",
            (cache_key, file_hash)
        )
        row = cur.fetchone()
        if row:
            cached: dict[str, Any] = _loads(row["result_json"], "semantic_cache.result_json")
            return cached
        return None

    @_writes
    def set_semantic_cache(self, cache_key: str, file_hash: str, result: dict[str, Any]) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO semantic_cache (cache_key, file_hash, result_json, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (cache_key, file_hash, json.dumps(result), time.time())
        )

    @_writes
    def clear_semantic_cache(self) -> None:
        self._check_closed()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM semantic_cache")
