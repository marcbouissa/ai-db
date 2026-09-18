"""
ai_db/telemetry/tracker.py
Performance & Token Telemetry Tracking Subsystem for ai-db.

Measures:
  - Query latency and throughput across storage backends (p50, p95, p99 percentiles)
  - Token compression savings across representations (Stub vs S-Exp vs JSON vs Raw source)
  - Semantic cache hit rates and tokens saved
  - Storage space trends (SQLite file size, table row counts)
  - Codebase weak points (syntax error density, complexity hotspots, unindexed files)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional


from ai_db.storage.state import (
    get_telemetry_state,
    set_telemetry_state,
    get_telemetry_table_counts,
    get_telemetry_weak_points,
)


class TelemetryTracker:
    """Tracks latency, token efficiency, cache performance, and weak points."""

    SESSION_STATE_KEY = "telemetry_metrics_v1"

    def __init__(self, conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None):
        self.conn = conn
        self.db_path = db_path
        self._metrics: Dict[str, Any] = self._initial_metrics()
        self._load_state()

    def _initial_metrics(self) -> Dict[str, Any]:
        return {
            "latency": {},
            "tokens": {
                "raw_tokens": 0,
                "stub_tokens": 0,
                "sexp_tokens": 0,
                "json_tokens": 0,
                "net_saved_tokens": 0,
                "savings_pct": 0.0,
                "compression_ratio": 1.0,
                "estimated_cost_saved_usd": 0.0,
            },
            "cache": {
                "lookups": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate_pct": 0.0,
                "tokens_saved": 0,
            },
        }

    def _load_state(self) -> None:
        if not self.conn:
            return
        saved = get_telemetry_state(self.conn, self.SESSION_STATE_KEY)
        if saved and isinstance(saved, dict):
            self._metrics = saved

    def _save_state(self) -> None:
        if not self.conn:
            return
        set_telemetry_state(self.conn, self.SESSION_STATE_KEY, self._metrics)

    def record_query(self, backend: str = "sqlite_wal", latency_ms: float = 0.0, results_count: int = 0) -> None:
        """Records latency and result counts for a specific backend."""
        lat = self._metrics.setdefault("latency", {})
        backend_stats = lat.setdefault(backend, {
            "count": 0,
            "min_ms": float("inf"),
            "max_ms": 0.0,
            "avg_ms": 0.0,
            "samples": [],
        })

        backend_stats["count"] += 1
        backend_stats["min_ms"] = min(backend_stats["min_ms"], latency_ms)
        backend_stats["max_ms"] = max(backend_stats["max_ms"], latency_ms)
        samples = backend_stats.setdefault("samples", [])
        samples.append(latency_ms)
        if len(samples) > 1000:
            samples.pop(0)

        backend_stats["avg_ms"] = sum(samples) / len(samples)
        self._save_state()

    def record_token_compression(
        self,
        raw_tokens: int = 0,
        stub_tokens: int = 0,
        sexp_tokens: int = 0,
        json_tokens: int = 0,
    ) -> None:
        """Records token counts across representations and updates cumulative savings."""
        tok = self._metrics.setdefault("tokens", {})
        tok["raw_tokens"] = tok.get("raw_tokens", 0) + raw_tokens
        tok["stub_tokens"] = tok.get("stub_tokens", 0) + stub_tokens
        tok["sexp_tokens"] = tok.get("sexp_tokens", 0) + sexp_tokens
        tok["json_tokens"] = tok.get("json_tokens", 0) + json_tokens

        raw = tok["raw_tokens"]
        stub = tok["stub_tokens"]
        if raw > 0:
            saved = max(0, raw - stub)
            tok["net_saved_tokens"] = saved
            tok["savings_pct"] = round((saved / raw) * 100.0, 2)
            tok["compression_ratio"] = round(raw / max(1, stub), 2)
            # Standard $3.00 per 1M tokens saved
            tok["estimated_cost_saved_usd"] = round((saved / 1_000_000) * 3.0, 4)
        else:
            tok["net_saved_tokens"] = 0
            tok["savings_pct"] = 0.0
            tok["compression_ratio"] = 1.0
            tok["estimated_cost_saved_usd"] = 0.0

        self._save_state()

    def record_cache_access(self, hit: bool, tokens_saved: int = 0) -> None:
        """Records cache hits/misses and tokens saved."""
        cache = self._metrics.setdefault("cache", {})
        cache["lookups"] = cache.get("lookups", 0) + 1
        if hit:
            cache["hits"] = cache.get("hits", 0) + 1
            cache["tokens_saved"] = cache.get("tokens_saved", 0) + tokens_saved
        else:
            cache["misses"] = cache.get("misses", 0) + 1

        lookups = cache["lookups"]
        hits = cache.get("hits", 0)
        cache["hit_rate_pct"] = round((hits / lookups) * 100.0, 2) if lookups > 0 else 0.0
        self._save_state()

    def reset(self) -> None:
        """Resets all recorded telemetry metrics."""
        self._metrics = self._initial_metrics()
        self._save_state()

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated metrics summary including storage trends."""
        summary = json.loads(json.dumps(self._metrics))

        # Compute percentiles for latency
        lat_dict = summary.get("latency", {})
        total_count = 0
        all_samples: List[float] = []

        for backend, data in list(lat_dict.items()):
            samples = sorted(data.pop("samples", []))
            total_count += data.get("count", 0)
            all_samples.extend(samples)
            if samples:
                n = len(samples)
                data["p50_ms"] = samples[int(n * 0.50)]
                data["p95_ms"] = samples[min(int(n * 0.95), n - 1)]
                data["p99_ms"] = samples[min(int(n * 0.99), n - 1)]
            else:
                data["p50_ms"] = 0.0
                data["p95_ms"] = 0.0
                data["p99_ms"] = 0.0
                if data.get("min_ms") == float("inf"):
                    data["min_ms"] = 0.0

        if not lat_dict:
            summary["latency"] = {
                "count": 0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "qps": 0.0,
            }
        else:
            # Also populate overall top-level aggregates for backward-compat
            all_samples.sort()
            n = len(all_samples)
            summary["latency"]["count"] = total_count
            if n > 0:
                summary["latency"]["min_ms"] = min(all_samples)
                summary["latency"]["max_ms"] = max(all_samples)
                summary["latency"]["avg_ms"] = sum(all_samples) / n
                summary["latency"]["p50_ms"] = all_samples[int(n * 0.50)]
                summary["latency"]["p95_ms"] = all_samples[min(int(n * 0.95), n - 1)]
                summary["latency"]["p99_ms"] = all_samples[min(int(n * 0.99), n - 1)]
            else:
                summary["latency"]["min_ms"] = 0.0
                summary["latency"]["max_ms"] = 0.0
                summary["latency"]["avg_ms"] = 0.0
                summary["latency"]["p50_ms"] = 0.0
                summary["latency"]["p95_ms"] = 0.0
                summary["latency"]["p99_ms"] = 0.0

        summary["storage"] = self._compute_storage_metrics()
        return summary

    def _compute_storage_metrics(self) -> Dict[str, Any]:
        storage: Dict[str, Any] = {
            "db_size_kb": 0.0,
            "table_rows": {},
        }
        if self.db_path and self.db_path != ":memory:" and os.path.exists(self.db_path):
            try:
                storage["db_size_kb"] = round(os.path.getsize(self.db_path) / 1024.0, 2)
            except OSError:
                storage["db_size_kb"] = 0.0

        if self.conn:
            storage["table_rows"] = get_telemetry_table_counts(self.conn)

        return storage

    def compute_weak_points(self) -> Dict[str, Any]:
        """Calculates syntax error density, complexity hotspots, and unindexed/stale files."""
        if not self.conn:
            return {
                "syntax_error_density_pct": 0.0,
                "complexity_hotspots": [],
                "unindexed_or_stale_files": [],
            }
        return get_telemetry_weak_points(self.conn)
