"""
tests/test_telemetry.py
Telemetry derived from the query log (K5): latency percentiles over one window, cache hit
rates, per-tool and per-stage breakdowns, storage and weak points, CLI/MCP exposure.
"""
import json
import subprocess
import sys
import time

import pytest

from ai_db import VectorDB
from ai_db.telemetry.tracker import TelemetryTracker


@pytest.fixture
def telemetry_db(tmp_path):
    """An isolated database and VectorDB instance."""
    db_file = str(tmp_path / "telemetry.db")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "calc.py").write_text("def add(a, b): return a + b\n")

    vdb = VectorDB(db_file)
    vdb.sync(str(src_dir), project="telemetry_test", verbose=False)
    vdb.backend.clear_query_log()
    yield {"db_file": db_file, "src_dir": src_dir, "vdb": vdb, "tmp_path": tmp_path}
    vdb.close()


def _log(backend, total_ms, tool="query", cache_hit=False, stages=None):
    backend.log_query({"timestamp": time.time(), "tool": tool, "query": "q", "mode": None,
                       "total_ms": total_ms, "cache_hit": cache_hit, "stages": stages or {},
                       "providers": {}, "top": []})


class TestSummary:
    def test_latency_over_logged_calls(self, telemetry_db):
        backend = telemetry_db["vdb"].backend
        for ms in (10.0, 20.0, 30.0):
            _log(backend, ms)
        lat = TelemetryTracker(backend).get_summary()["latency"]
        assert (lat["count"], lat["min_ms"], lat["max_ms"], lat["avg_ms"]) == (3, 10.0, 30.0, 20.0)

    def test_percentiles(self, telemetry_db):
        backend = telemetry_db["vdb"].backend
        for val in range(1, 101):
            _log(backend, float(val))
        lat = TelemetryTracker(backend).get_summary()["latency"]
        assert 48.0 <= lat["p50_ms"] <= 52.0
        assert 94.0 <= lat["p95_ms"] <= 96.0
        assert 98.0 <= lat["p99_ms"] <= 100.0

    def test_per_tool_breakdown(self, telemetry_db):
        backend = telemetry_db["vdb"].backend
        _log(backend, 5.0, tool="query")
        _log(backend, 25.0, tool="investigate")
        by_tool = TelemetryTracker(backend).get_summary()["latency"]["by_tool"]
        assert by_tool["query"]["avg_ms"] == 5.0
        assert by_tool["investigate"]["avg_ms"] == 25.0

    def test_cache_hit_rate(self, telemetry_db):
        backend = telemetry_db["vdb"].backend
        for _ in range(8):
            _log(backend, 1.0, cache_hit=True)
        for _ in range(2):
            _log(backend, 9.0, cache_hit=False)
        cache = TelemetryTracker(backend).get_summary()["cache"]
        assert cache == {"lookups": 10, "hits": 8, "misses": 2, "hit_rate_pct": 80.0}

    def test_count_and_average_cover_the_same_window(self, telemetry_db, monkeypatch):
        """The old blob averaged the last 1000 samples but counted every query ever."""
        from ai_db.storage import sqlite_backend
        monkeypatch.setattr(sqlite_backend, "QUERY_LOG_KEEP", 5)
        backend = telemetry_db["vdb"].backend
        for ms in (100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0):
            _log(backend, ms)
        summary = TelemetryTracker(backend).get_summary()
        assert summary["latency"]["count"] == summary["window"]["calls"] == 5
        assert summary["latency"]["avg_ms"] == 1.0

    def test_stages(self, telemetry_db):
        backend = telemetry_db["vdb"].backend
        _log(backend, 12.0, stages={"bm25": 3.0, "rank": 4.0})
        stages = TelemetryTracker(backend).get_summary()["stages"]
        assert stages["bm25"]["count"] == 1 and stages["total"]["p50_ms"] == 12.0

    def test_storage(self, telemetry_db):
        summary = TelemetryTracker(telemetry_db["vdb"].backend,
                                   db_path=telemetry_db["db_file"]).get_summary()
        assert summary["storage"]["db_size_kb"] > 0.0
        assert summary["storage"]["table_rows"]["chunks"] >= 1

    def test_empty_log_has_no_division_by_zero(self, telemetry_db):
        summary = TelemetryTracker(telemetry_db["vdb"].backend).get_summary()
        assert summary["latency"]["count"] == 0 and summary["latency"]["avg_ms"] == 0.0
        assert summary["cache"]["hit_rate_pct"] == 0.0

    def test_reset_clears_the_log(self, telemetry_db):
        backend = telemetry_db["vdb"].backend
        _log(backend, 15.0, cache_hit=True)
        tracker = TelemetryTracker(backend)
        tracker.reset()
        summary = tracker.get_summary()
        assert summary["latency"]["count"] == 0 and summary["cache"]["hits"] == 0

    def test_in_memory_db_has_no_size(self):
        from ai_db.storage.sqlite_backend import SQLiteBackend
        backend = SQLiteBackend(":memory:")
        backend.initialize()
        assert TelemetryTracker(backend, db_path=":memory:").get_summary()["storage"]["db_size_kb"] == 0.0
        backend.close()

    def test_persists_across_reconnects_and_processes(self, telemetry_db):
        db_file = telemetry_db["db_file"]
        _log(telemetry_db["vdb"].backend, 7.0)
        other = VectorDB(db_file)  # a second process appends rows; nothing is overwritten
        try:
            _log(other.backend, 9.0)
            assert TelemetryTracker(other.backend).get_summary()["latency"]["count"] == 2
        finally:
            other.close()


class TestIntegration:
    def test_query_calls_are_logged(self, telemetry_db):
        vdb = telemetry_db["vdb"]
        vdb.query("add", project="telemetry_test")
        vdb.query("add", project="telemetry_test")
        summary = vdb.telemetry_tracker.get_summary()
        assert summary["latency"]["by_tool"]["query"]["count"] == 2
        assert summary["cache"]["hits"] == 1  # the second call is a query-cache hit

    def test_analyze_calls_log_the_semantic_cache(self, telemetry_db):
        vdb = telemetry_db["vdb"]
        target = str(telemetry_db["src_dir"] / "calc.py")
        vdb.analyze_file(target, depth="structure")  # miss
        vdb.analyze_file(target, depth="structure")  # hit
        summary = vdb.telemetry_tracker.get_summary()
        assert summary["latency"]["by_tool"]["analyze"]["count"] == 2
        assert summary["cache"]["hits"] == 1

    def test_weak_points(self, telemetry_db):
        report = telemetry_db["vdb"].telemetry_tracker.compute_weak_points()
        assert {"syntax_error_density_pct", "complexity_hotspots", "unindexed_or_stale_files"} <= set(report)

    def test_dispatcher_has_no_fabricated_fallback(self, telemetry_db):
        from ai_db.dispatcher import ServiceDispatcher
        out = ServiceDispatcher(db=telemetry_db["vdb"]).execute("telemetry", {})
        assert "token_savings" not in out and out["window"]["calls"] == 0

    def test_mcp_tool(self, telemetry_db):
        from mcp_server import StdioMCPServer
        server = StdioMCPServer(telemetry_db["db_file"])
        resp = server.handle_request({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                                      "params": {"name": "telemetry", "arguments": {}}})
        assert resp["result"]["isError"] is False


class TestCli:
    def _run(self, db_file, *args):
        return subprocess.run([sys.executable, "-m", "ai_db.cli", "telemetry", *args, "--db", db_file],
                              capture_output=True, text=True, check=False)

    def test_dashboard(self, telemetry_db):
        proc = self._run(telemetry_db["db_file"])
        assert proc.returncode == 0, proc.stderr
        assert "Latency" in proc.stdout and "Cache" in proc.stdout

    def test_json(self, telemetry_db):
        proc = self._run(telemetry_db["db_file"], "--format", "json")
        assert proc.returncode == 0, proc.stderr
        assert "latency" in json.loads(proc.stdout)

    def test_reset(self, telemetry_db):
        proc = self._run(telemetry_db["db_file"], "--reset")
        assert proc.returncode == 0, proc.stderr
        assert "reset" in proc.stdout.lower()
