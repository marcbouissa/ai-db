"""
tests/test_telemetry.py
Comprehensive 4-tier test suite for the Performance & Token Telemetry subsystem.

Tiers:
  - Tier 1: Feature Coverage (query latency percentiles, 4-format token savings, cache hit rates, weak points, exposure)
  - Tier 2: Boundary & Corner Cases (zero queries, zero cache lookups, zero raw tokens, metrics reset, :memory: db)
  - Tier 3: Pairwise Combinations (query engine telemetry integration, analyzer cache tracking, session persistence)
  - Tier 4: Real-World Workflows (developer session telemetry dashboard, CI codebase health audit)
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from ai_db import VectorDB

try:
    from ai_db.telemetry.tracker import TelemetryTracker
except ImportError:
    try:
        from ai_db.telemetry import TelemetryTracker
    except ImportError:
        TelemetryTracker = None


# ==============================================================================
# Helpers & Fixtures
# ==============================================================================

@pytest.fixture
def telemetry_db(tmp_path):
    """Provides an isolated database and VectorDB instance for telemetry verification."""
    db_file = str(tmp_path / "telemetry.db")
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "calc.py").write_text("def add(a, b): return a + b\n")

    vdb = VectorDB(db_file)
    vdb.sync(str(src_dir), project="telemetry_test", verbose=False)

    yield {
        "db_file": db_file,
        "src_dir": src_dir,
        "vdb": vdb,
        "tmp_path": tmp_path,
    }
    vdb.close()


def _require_telemetry():
    if TelemetryTracker is None:
        pytest.skip("Telemetry subsystem (ai_db.telemetry) not yet implemented (planned for Milestone 4)")


# ==============================================================================
# Tier 1: Feature Coverage (Happy Path Isolation)
# ==============================================================================

class TestTelemetryTier1FeatureCoverage:
    """Happy-path tests for latency percentiles, token compression, cache metrics, and diagnostics."""

    def test_telemetry_tracker_record_query(self, telemetry_db):
        """TelemetryTracker records query latency and computes count, min, max, and avg."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        tracker.record_query(backend="sqlite_wal", latency_ms=10.0, results_count=5)
        tracker.record_query(backend="sqlite_wal", latency_ms=20.0, results_count=3)
        tracker.record_query(backend="sqlite_wal", latency_ms=30.0, results_count=8)

        summary = tracker.get_summary()
        lat = summary.get("latency", {}).get("sqlite_wal", summary.get("latency", {}))
        assert lat["count"] == 3
        assert lat["min_ms"] == 10.0
        assert lat["max_ms"] == 30.0
        assert lat["avg_ms"] == 20.0

    def test_telemetry_latency_percentiles(self, telemetry_db):
        """Calculates p50, p95, and p99 percentiles from 100 recorded latency samples."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        for val in range(1, 101):
            tracker.record_query(backend="sqlite_wal", latency_ms=float(val), results_count=1)

        summary = tracker.get_summary()
        lat = summary.get("latency", {}).get("sqlite_wal", summary.get("latency", {}))
        assert 48.0 <= lat["p50_ms"] <= 52.0
        assert 94.0 <= lat["p95_ms"] <= 96.0
        assert 98.0 <= lat["p99_ms"] <= 100.0

    def test_telemetry_multi_backend_latency(self, telemetry_db):
        """Maintains independent latency distributions across multiple storage backends."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        tracker.record_query(backend="sqlite_wal", latency_ms=5.0, results_count=2)
        tracker.record_query(backend="mysql_adapter", latency_ms=25.0, results_count=2)

        summary = tracker.get_summary()
        assert "sqlite_wal" in summary["latency"]
        assert "mysql_adapter" in summary["latency"]
        assert summary["latency"]["sqlite_wal"]["avg_ms"] == 5.0
        assert summary["latency"]["mysql_adapter"]["avg_ms"] == 25.0

    def test_telemetry_token_compression_savings(self, telemetry_db):
        """Calculates net tokens saved and compression ratio across Stub, S-Exp, and JSON."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        # 10,000 raw tokens compressed to 2,000 stub tokens
        tracker.record_token_compression(raw_tokens=10000, stub_tokens=2000, sexp_tokens=3500, json_tokens=6000)

        summary = tracker.get_summary()
        tok = summary["tokens"]
        assert tok["raw_tokens"] == 10000
        assert tok["stub_tokens"] == 2000
        assert tok["net_saved_tokens"] == 8000
        assert tok["savings_pct"] == 80.0

    def test_telemetry_cost_savings_calculation(self, telemetry_db):
        """Calculates estimated financial savings based on $3.00 / 1M tokens saved."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        # 1,000,000 saved tokens = $3.00
        tracker.record_token_compression(raw_tokens=2000000, stub_tokens=1000000, sexp_tokens=1200000, json_tokens=1500000)

        summary = tracker.get_summary()
        usd = summary["tokens"]["estimated_cost_saved_usd"]
        assert round(usd, 2) == 3.00

    def test_telemetry_cache_access_metrics(self, telemetry_db):
        """Computes cache hit rate percentage from recorded hits and misses."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        for _ in range(8):
            tracker.record_cache_access(hit=True, tokens_saved=250)
        for _ in range(2):
            tracker.record_cache_access(hit=False, tokens_saved=0)

        summary = tracker.get_summary()
        cache = summary["cache"]
        assert cache["lookups"] == 10
        assert cache["hits"] == 8
        assert cache["misses"] == 2
        assert cache["hit_rate_pct"] == 80.0
        assert cache["tokens_saved"] == 2000

    def test_telemetry_disk_and_memory_trends(self, telemetry_db):
        """Tracks SQLite file size, table row counts, and storage trends."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn, db_path=telemetry_db["db_file"])
        summary = tracker.get_summary()
        storage = summary.get("storage", {})
        assert storage.get("db_size_kb", 0) > 0
        assert "table_rows" in storage

    def test_telemetry_weak_points_syntax_density(self, telemetry_db):
        """Weak points diagnostic calculates syntax error density as % of total files."""
        _require_telemetry()
        vdb = telemetry_db["vdb"]
        src = telemetry_db["src_dir"]

        (src / "clean1.py").write_text("def a(): pass\n")
        (src / "clean2.py").write_text("def b(): pass\n")
        (src / "broken.py").write_text("def c( :\n    pass\n")
        vdb.sync(str(src), project="weak_points", verbose=False)

        tracker = TelemetryTracker(vdb.conn)
        wp = tracker.compute_weak_points()
        assert "syntax_error_density_pct" in wp
        # 1 error in 4 total files (including initial calc.py) = 25%
        assert 20.0 <= wp["syntax_error_density_pct"] <= 35.0

    def test_telemetry_weak_points_complexity_hotspots(self, telemetry_db):
        """Weak points identifies top complexity hotspots by chunk and symbol density."""
        _require_telemetry()
        vdb = telemetry_db["vdb"]
        src = telemetry_db["src_dir"]

        # High complexity file with many functions
        complex_code = "\n".join([f"def func_{i}(): pass" for i in range(20)])
        (src / "complex_module.py").write_text(complex_code)
        vdb.sync(str(src), project="telemetry_test", verbose=False)

        tracker = TelemetryTracker(vdb.conn)
        wp = tracker.compute_weak_points()
        assert "complexity_hotspots" in wp
        hotspots = wp["complexity_hotspots"]
        assert len(hotspots) > 0
        assert "complex_module.py" in hotspots[0]["filepath"]

    def test_telemetry_cli_dashboard_output(self, telemetry_db):
        """CLI telemetry command formats dashboard with Token, Latency, and Cache sections."""
        db_file = telemetry_db["db_file"]
        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "telemetry", "--db", db_file],
            capture_output=True,
            text=True,
            check=False
        )
        if proc.returncode != 0 and "invalid choice: 'telemetry'" in proc.stderr:
            pytest.skip("CLI subcommand 'telemetry' not yet implemented (planned for Milestone 4)")

        assert proc.returncode == 0
        assert "Token" in proc.stdout or "Latency" in proc.stdout or "Cache" in proc.stdout

    def test_telemetry_mcp_tool_exposure(self, telemetry_db):
        """MCP server exposes 'telemetry' tool returning summary metrics."""
        from mcp_server import StdioMCPServer
        server = StdioMCPServer(telemetry_db["db_file"])
        resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "telemetry", "arguments": {}}
        })
        if resp["result"].get("isError") and "Unknown tool" in resp["result"]["content"][0]["text"]:
            pytest.skip("MCP 'telemetry' tool not yet registered (planned for Milestone 4)")

        assert resp["result"]["isError"] is False

    def test_telemetry_http_endpoint(self, telemetry_db):
        """HTTP server exposes GET /telemetry returning full metrics dictionary."""
        from http.server import HTTPServer

        from ai_db.server.http_server import AiDbHandler

        server = HTTPServer(("127.0.0.1", 0), AiDbHandler)
        server.db_path = telemetry_db["db_file"]
        host, port = server.server_address

        import threading
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        url = f"http://{host}:{port}/telemetry"
        try:
            with urllib.request.urlopen(url) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert isinstance(data, dict)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                pytest.skip("HTTP route GET /telemetry not yet implemented (planned for Milestone 4)")
            raise
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=2.0)


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

class TestTelemetryTier2BoundaryAndCorner:
    """Division by zero protection, metric resets, and in-memory databases."""

    def test_telemetry_zero_queries_no_div_zero(self, telemetry_db):
        """ZeroDivisionError is avoided when retrieving summary with zero recorded queries."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        summary = tracker.get_summary()
        lat = summary.get("latency", {})
        assert lat.get("count", 0) == 0
        assert lat.get("avg_ms", 0.0) == 0.0
        assert lat.get("qps", 0.0) == 0.0

    def test_telemetry_zero_cache_access_no_div_zero(self, telemetry_db):
        """Zero cache lookups safely returns 0.0% hit rate without division error."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        summary = tracker.get_summary()
        cache = summary.get("cache", {})
        assert cache.get("lookups", 0) == 0
        assert cache.get("hit_rate_pct", 0.0) == 0.0

    def test_telemetry_zero_raw_tokens_no_div_zero(self, telemetry_db):
        """Recording zero raw tokens defaults compression ratio to 1.0 (0% savings)."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        tracker.record_token_compression(raw_tokens=0, stub_tokens=0, sexp_tokens=0, json_tokens=0)
        summary = tracker.get_summary()
        tok = summary.get("tokens", {})
        assert tok.get("compression_ratio", 1.0) == 1.0
        assert tok.get("savings_pct", 0.0) == 0.0

    def test_telemetry_reset_metrics(self, telemetry_db):
        """Calling reset() clears all accumulated latency, token, and cache counters."""
        _require_telemetry()
        tracker = TelemetryTracker(telemetry_db["vdb"].conn)
        tracker.record_query(backend="sqlite_wal", latency_ms=15.0, results_count=2)
        tracker.record_cache_access(hit=True, tokens_saved=500)
        tracker.reset()

        summary = tracker.get_summary()
        assert summary.get("latency", {}).get("count", 0) == 0
        assert summary.get("cache", {}).get("hits", 0) == 0

    def test_telemetry_cli_reset_flag(self, telemetry_db):
        """CLI telemetry --reset clears metrics in persistent session_state."""
        db_file = telemetry_db["db_file"]
        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "telemetry", "--reset", "--db", db_file],
            capture_output=True,
            text=True,
            check=False
        )
        if proc.returncode != 0 and "invalid choice: 'telemetry'" in proc.stderr:
            pytest.skip("CLI subcommand 'telemetry' not yet implemented (planned for Milestone 4)")

        assert proc.returncode == 0
        assert "reset" in proc.stdout.lower() or "cleared" in proc.stdout.lower()

    def test_telemetry_cli_format_json(self, telemetry_db):
        """CLI telemetry --format json outputs machine-readable JSON dictionary."""
        db_file = telemetry_db["db_file"]
        proc = subprocess.run(
            [sys.executable, "-m", "ai_db.cli", "telemetry", "--format", "json", "--db", db_file],
            capture_output=True,
            text=True,
            check=False
        )
        if proc.returncode != 0 and "invalid choice: 'telemetry'" in proc.stderr:
            pytest.skip("CLI subcommand 'telemetry' not yet implemented (planned for Milestone 4)")

        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert isinstance(data, dict)

    def test_telemetry_in_memory_db_disk_size(self):
        """In-memory SQLite database (:memory:) does not crash file size queries."""
        _require_telemetry()
        from ai_db.storage.sqlite_backend import SQLiteBackend
        backend = SQLiteBackend(":memory:")
        backend.initialize()
        tracker = TelemetryTracker(backend.conn, db_path=":memory:")
        summary = tracker.get_summary()
        storage = summary.get("storage", {})
        assert storage.get("db_size_kb", 0) == 0.0
        backend.close()


# ==============================================================================
# Tier 3: Pairwise Combinations
# ==============================================================================

class TestTelemetryTier3Combinations:
    """Integration between TelemetryTracker and core query/analyzer engines."""

    def test_telemetry_query_engine_integration(self, telemetry_db):
        """QueryEngine executions automatically record latency and results in TelemetryTracker."""
        _require_telemetry()
        vdb = telemetry_db["vdb"]
        # Execute query
        vdb.query("add", project="telemetry_test")

        tracker = getattr(vdb, "telemetry_tracker", None)
        if tracker is None:
            pytest.skip("VectorDB automatic telemetry integration planned for Milestone 4")

        summary = tracker.get_summary()
        assert summary["latency"]["count"] >= 1

    def test_telemetry_analyzer_engine_cache_integration(self, telemetry_db):
        """Subsequent analyze_file calls record cache misses then hits in TelemetryTracker."""
        _require_telemetry()
        vdb = telemetry_db["vdb"]
        target = str(telemetry_db["src_dir"] / "calc.py")

        # 1st call: cache miss
        vdb.analyze_file(target, depth="structure")
        # 2nd call: cache hit
        vdb.analyze_file(target, depth="structure")

        tracker = getattr(vdb, "telemetry_tracker", None)
        if tracker is None:
            pytest.skip("VectorDB automatic telemetry integration planned for Milestone 4")

        cache = tracker.get_summary()["cache"]
        assert cache["lookups"] >= 2
        assert cache["hits"] >= 1

    def test_telemetry_persistence_across_process_restart(self, telemetry_db):
        """Telemetry metrics stored in session_state survive database reconnects."""
        _require_telemetry()
        db_file = telemetry_db["db_file"]

        # Session 1: Record metric
        vdb1 = VectorDB(db_file)
        t1 = TelemetryTracker(vdb1.conn, db_path=db_file)
        t1.record_token_compression(raw_tokens=5000, stub_tokens=1000, sexp_tokens=2000, json_tokens=3000)
        vdb1.close()

        # Session 2: Restore and verify
        vdb2 = VectorDB(db_file)
        t2 = TelemetryTracker(vdb2.conn, db_path=db_file)
        summary = t2.get_summary()
        assert summary["tokens"]["raw_tokens"] == 5000
        vdb2.close()


# ==============================================================================
# Tier 4: Real-World Application Workflows
# ==============================================================================

class TestTelemetryTier4Workflows:
    """Simulated production monitoring and developer audit sessions."""

    def test_workflow_developer_session_telemetry(self, telemetry_db):
        """
        Developer session workflow:
        1. Indexes codebase
        2. Performs AST search and analyses
        3. Retrieves telemetry summary to quantify token and latency savings
        """
        _require_telemetry()
        vdb = telemetry_db["vdb"]
        tracker = TelemetryTracker(vdb.conn, db_path=telemetry_db["db_file"])

        # Simulate queries & token usage
        tracker.record_query("sqlite_wal", latency_ms=12.4, results_count=4)
        tracker.record_token_compression(raw_tokens=8000, stub_tokens=1600, sexp_tokens=2400, json_tokens=4500)
        tracker.record_cache_access(hit=True, tokens_saved=6400)

        summary = tracker.get_summary()
        assert summary["tokens"]["savings_pct"] >= 75.0
        assert summary["cache"]["tokens_saved"] == 6400

    def test_workflow_codebase_health_audit(self, telemetry_db):
        """
        CI codebase health audit workflow:
        Analyzes weak points report to detect syntax errors, hotspots, and stale files.
        """
        _require_telemetry()
        vdb = telemetry_db["vdb"]
        tracker = TelemetryTracker(vdb.conn, db_path=telemetry_db["db_file"])

        report = tracker.compute_weak_points()
        assert "syntax_error_density_pct" in report
        assert "complexity_hotspots" in report
        assert "unindexed_or_stale_files" in report
