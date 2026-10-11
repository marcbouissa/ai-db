"""
ai_db/telemetry/tracker.py
Telemetry derived from the query log.

Every logged tool call (``query``, ``investigate``, ``analyze``, ...) is one row of the
backend's ``query_log`` (the newest ``QUERY_LOG_KEEP``). The summary is computed from those
rows on demand, so every figure covers the same window, concurrent processes append rows
instead of overwriting one shared blob, and nothing is written per query besides the row:

  - latency: count, min/max/avg and p50/p95/p99 over the window, overall and per tool
  - cache: lookups, hits, misses, hit rate (query cache for retrieval tools, the semantic
    cache for ``analyze``)
  - stages: per-stage latency of computed (non-cached) calls
  - storage: database size and table row counts
  - weak points: syntax error density, complexity hotspots, unindexed/stale files
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ai_db.constants import QUERY_LOG_KEEP
from ai_db.telemetry.stages import stage_stats

if TYPE_CHECKING:
    from ai_db.storage.backend import StorageBackend


def _percentile(ordered: list[float], pct: float) -> float:
    return ordered[min(int(len(ordered) * pct), len(ordered) - 1)]


def _latency(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0,
                "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "avg_ms": round(sum(ordered) / len(ordered), 3),
        "p50_ms": round(_percentile(ordered, 0.50), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "p99_ms": round(_percentile(ordered, 0.99), 3),
    }


class TelemetryTracker:
    """Read-only telemetry over the backend's query log."""

    def __init__(self, backend: StorageBackend, db_path: str | None = None):
        self.backend = backend
        self.db_path = db_path

    def reset(self) -> None:
        """Forget the logged calls (the query log is the telemetry)."""
        self.backend.clear_query_log()

    def get_summary(self) -> dict[str, Any]:
        rows = self.backend.get_query_log(limit=QUERY_LOG_KEEP)
        by_tool: dict[str, list[float]] = {}
        for r in rows:
            by_tool.setdefault(r["tool"], []).append(float(r["total_ms"]))
        latency = _latency([float(r["total_ms"]) for r in rows])
        latency["by_tool"] = {tool: _latency(vals) for tool, vals in sorted(by_tool.items())}
        hits = sum(1 for r in rows if r["cache_hit"])
        return {
            "window": {
                "calls": len(rows),
                "oldest": min((r["timestamp"] for r in rows), default=None),
                "newest": max((r["timestamp"] for r in rows), default=None),
                "keep": QUERY_LOG_KEEP,
            },
            "latency": latency,
            "cache": {
                "lookups": len(rows),
                "hits": hits,
                "misses": len(rows) - hits,
                "hit_rate_pct": round(100.0 * hits / len(rows), 2) if rows else 0.0,
            },
            "stages": stage_stats(rows),
            "storage": self._storage(),
        }

    def _storage(self) -> dict[str, Any]:
        size_kb = 0.0
        if self.db_path and self.db_path != ":memory:" and os.path.exists(self.db_path):
            size_kb = round(os.path.getsize(self.db_path) / 1024.0, 2)
        return {"db_size_kb": size_kb, "table_rows": self.backend.get_table_counts()}

    def compute_weak_points(self) -> dict[str, Any]:
        """Syntax error density, complexity hotspots, and unindexed/stale files."""
        return self.backend.get_weak_points()
